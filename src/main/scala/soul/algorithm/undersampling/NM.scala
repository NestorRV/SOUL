package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities
import soul.util.Utilities._

import scala.util.Random

/** NearMiss. Original paper: "kNN Approach to Unbalanced Data Distribution: A Case Study involving Information
  * Extraction" by Jianping Zhang and Inderjeet Mani.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param dist        distance to be used. It should be "HVDM" or a function of the type: (Array[Double], Array[Double]) => Double.
  * @param version     version of the core to execute
  * @param nNeighbours number of neighbours to take for each minority example (only used if version is set to 3)
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class NM(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), dist: Any = Utilities.euclideanDistance _,
         version: Int = 1, nNeighbours: Int = 3, ratio: Double = 1.0, val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {

  private[soul] val distance: Distances.Distance = getDistance(dist)
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the NM algorithm.
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

    val majElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }
    val minElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val minNeighbours: Array[Array[Double]] = minElements map dataToWorkWith
    val majNeighbours: Array[Array[Double]] = majElements map dataToWorkWith
    val minClasses: Array[Any] = minElements map classesToWorkWith
    val majClasses: Array[Any] = majElements map classesToWorkWith
    val selectedMajElements: Array[Int] = if (version == 1) {
      majElements.map { i: Int =>
        val result: (Any, Array[Int], Array[Double]) = if (distance == Distances.USER) {
          nnRule(minNeighbours, dataToWorkWith(i), i, minClasses, 3, dist, "nearest")
        } else {
          nnRuleHVDM(minNeighbours, dataToWorkWith(i), i, minClasses, 3, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
        }
        (i, (result._2 map result._3).sum / result._2.length)
      }.sortBy((_: (Int, Double))._2).map((_: (Int, Double))._1)
    } else if (version == 2) {
      majElements.map { i: Int =>
        val result: (Any, Array[Int], Array[Double]) = if (distance == Distances.USER) {
          nnRule(minNeighbours, dataToWorkWith(i), i, minClasses, 3, dist, "farthest")
        } else {
          nnRuleHVDM(minNeighbours, dataToWorkWith(i), i, minClasses, 3, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "farthest")
        }
        (i, (result._2 map result._3).sum / result._2.length)
      }.sortBy((_: (Int, Double))._2).map((_: (Int, Double))._1)
    } else if (version == 3) {
      // We shuffle the data because, at last, we are going to take, at least, minElements.length * ratio elements and if
      // we don't shuffle, we only take majority elements examples that are near to the first minority class examples
      new Random(seed).shuffle(minElements.flatMap { i: Int =>
        if (distance == Distances.USER) {
          nnRule(majNeighbours, dataToWorkWith(i), i, majClasses, nNeighbours, dist, "nearest")._2
        } else {
          nnRuleHVDM(majNeighbours, dataToWorkWith(i), i, majClasses, nNeighbours, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")._2
        }
      }.distinct.toList).toArray
    } else {
      throw new Exception("Invalid argument: version should be: 1, 2 or 3")
    }

    val finalIndex: Array[Int] = minElements ++ selectedMajElements.take((minElements.length * ratio).toInt)
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.info("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.info("NEW DATA SIZE: %d".format(finalIndex.length))
      logger.info("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      logger.info("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.info("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
