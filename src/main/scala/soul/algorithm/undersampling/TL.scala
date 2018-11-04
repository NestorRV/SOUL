package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

/** Tomek Link core. Original paper: "Two Modifications of CNN" by Ivan Tomek.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param dist          object of DistanceType representing the distance to be used
  * @param ratio         indicates the instances of the Tomek Links that are going to be remove. "all" will remove all instances,
  *                      "minority" will remove instances of the minority class and "not minority" will remove all the instances
  *                      except the ones of the minority class.
  * @param minorityClass minority class. If set to None, it will be computed
  * @param normalize     normalize the data or not
  * @param randomData    iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class TL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), dist: DistanceType = Distance(euclideanDistance),
         ratio: String = "not minority", val minorityClass: Option[Any] = None, val normalize: Boolean = false,
         val randomData: Boolean = false) extends LazyLogging {

  /** Compute the TL algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = if (minorityClass.isDefined) minorityClass.get else counter.minBy((c: (Any, Int)) => c._2)._1
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

    val (attrCounter, attrClassesCounter, sds) = if (dist.isInstanceOf[HVDM]) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val candidates: Map[Any, Array[Int]] = classesToWorkWith.distinct.map {
      c: Any =>
        c -> classesToWorkWith.zipWithIndex.collect {
          case (a, b) if a != c => b
        }
    }.toMap

    val distances: Array[Array[Double]] = Array.fill[Array[Double]](dataToWorkWith.length)(new Array[Double](dataToWorkWith.length))

    if (dist.isInstanceOf[Distance]) {
      dataToWorkWith.indices.par.foreach { i: Int =>
        dataToWorkWith.indices.drop(i).par.foreach { j: Int =>
          distances(i)(j) = euclideanDistance(dataToWorkWith(i), dataToWorkWith(j))
          distances(j)(i) = distances(i)(j)
        }
      }
    } else {
      dataToWorkWith.indices.par.foreach { i: Int =>
        dataToWorkWith.indices.drop(i).par.foreach { j: Int =>
          distances(i)(j) = HVDM(dataToWorkWith(i), dataToWorkWith(j), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          distances(j)(i) = distances(i)(j)
        }
      }
    }

    // Look for the nearest neighbour in the rest of the classes
    val nearestNeighbour: Array[Int] = distances.zipWithIndex.map((row: (Array[Double], Int)) => row._1.indexOf((candidates(classesToWorkWith(row._2)) map row._1).min))
    // For each instance, I: If my nearest neighbour is J and the nearest neighbour of J it's me, I, I and J form a Tomek link
    val tomekLinks: Array[(Int, Int)] = nearestNeighbour.zipWithIndex.filter((pair: (Int, Int)) => nearestNeighbour(pair._1) == pair._2)
    val targetInstances: Array[Int] = tomekLinks.flatMap((x: (Int, Int)) => List(x._1, x._2)).distinct
    // but the user can choose which of them should be removed
    val removedInstances: Array[Int] = if (ratio == "all") targetInstances else if (ratio == "minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a == untouchableClass => b
      } else if (ratio == "not minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a != untouchableClass => b
      } else
      throw new Exception("Incorrect value of ratio. Possible options: all, minority, not minority")
    val finalIndex: Array[Int] = dataToWorkWith.indices.diff(removedInstances).toArray
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      logger.info("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.info("NEW DATA SIZE: %d".format(finalIndex.length))
      logger.info("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      logger.info("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.info("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.info("REMOVED INSTANCES: %s".format(ratio))
    }

    newData
  }
}
