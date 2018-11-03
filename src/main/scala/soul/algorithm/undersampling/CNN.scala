package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

/** Condensed Nearest Neighbor decision rule. Original paper: "The Condensed Nearest Neighbor Rule" by P. Hart.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       distance to be used. It should be "HVDM" or a function of the type: (Array[Double], Array[Double]) => Double.
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class CNN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
          dist: Any, val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {

  private[soul] val distance: Distances.Distance = getDistance(dist)

  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the CNN algorithm
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val random: scala.util.Random = new scala.util.Random(seed)

    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    var randomIndex: List[Int] = data.x.indices.toList
    val classesToWorkWith: Array[Any] = if (randomData) {
      // Index to shuffle (randomize) the data
      randomIndex = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val (attrCounter, attrClassesCounter, sds) = if (distance == Distances.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues((_: Array[Double]).length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    logger.whenInfoEnabled {
      logger.info("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
    }

    // Indicate the corresponding group: 1 for store, 0 for unknown, -1 for grabbag
    val location: Array[Int] = List.fill(dataToWorkWith.length)(0).toArray
    var iteration: Int = 0
    // The first element is added to store
    location(0) = 1
    var changed = true

    // Iterate the data, x (except the first instance)
    dataToWorkWith.zipWithIndex.tail.foreach { element: (Array[Double], Int) =>
      // and classify each element with the actual content of store
      val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
      val neighbours: Array[Array[Double]] = index map dataToWorkWith
      val classes: Array[Any] = index map classesToWorkWith
      val label: (Any, Array[Int], Array[Double]) = if (distance == Distances.USER) {
        nnRule(neighbours, element._1, element._2, classes, 1, dist, "nearest")
      } else {
        nnRuleHVDM(neighbours, element._1, element._2, classes, 1, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
      }
      // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
      location(element._2) = if (label._1 != classesToWorkWith(element._2)) 1 else -1
    }

    logger.whenInfoEnabled {
      logger.info("ITERATION %d: GRABBAG SIZE: %d, STORE SIZE: %d.".format(iteration, location.count((z: Int) => z == -1), location.count((z: Int) => z == 1)))
    }

    // After a first pass, iterate grabbag until is exhausted:
    // 1. There is no element in grabbag or
    // 2. There is no data change between grabbag and store after a full iteration
    while (location.count((z: Int) => z == -1) != 0 && changed) {
      iteration += 1
      changed = false

      // Now, instead of iterating x, we iterate grabbag
      location.zipWithIndex.filter((x: (Int, Int)) => x._1 == -1).foreach { element: (Int, Int) =>
        val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
        val neighbours: Array[Array[Double]] = index map dataToWorkWith
        val classes: Array[Any] = index map classesToWorkWith
        val label: Any = if (distance == Distances.USER) {
          nnRule(neighbours, dataToWorkWith(element._2), element._2, classes, 1, dist, "nearest")._1
        } else {
          nnRuleHVDM(neighbours, dataToWorkWith(element._2), element._2, classes, 1, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")._1
        }
        // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
        location(element._2) = if (label != classesToWorkWith(element._2)) {
          changed = true
          1
        } else -1
      }

      logger.whenInfoEnabled {
        logger.info("ITERATION %d: GRABBAG SIZE: %d, STORE SIZE: %d.".format(iteration, location.count((z: Int) => z == -1), location.count((z: Int) => z == 1)))
      }
    }

    // The final data is the content of store
    val storeIndex: Array[Int] = location.zipWithIndex.filter((x: (Int, Int)) => x._1 == 1).collect { case (_, a) => a }
    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (storeIndex map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (storeIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.info("NEW DATA SIZE: %d".format(storeIndex.length))
      logger.info("REDUCTION PERCENTAGE: %s".format(100 - (storeIndex.length.toFloat / dataToWorkWith.length) * 100))
      logger.info("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.info("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
