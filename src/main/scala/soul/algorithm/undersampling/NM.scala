package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** NearMiss. Original paper: "kNN Approach to Unbalanced Data Distribution: A Case Study involving Information
  * Extraction" by Jianping Zhang and Inderjeet Mani.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param dist        object of Distance enumeration representing the distance to be used
  * @param version     version of the core to execute
  * @param nNeighbours number of neighbours to take for each minority example (only used if version is set to 3)
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @param verbose     choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class NM(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN, version: Int = 1,
         nNeighbours: Int = 3, ratio: Double = 1.0, normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the NM algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)
    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val classesToWorkWith: Array[Any] = if (randomData) {
      val randomIndex: List[Int] = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val majElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    val minElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    classesToWorkWith.zipWithIndex.foreach(i => if (i._1 == untouchableClass) minElements += i._2 else majElements += i._2)
    val minNeighbours: Array[Array[Double]] = minElements.toArray map dataToWorkWith
    val majNeighbours: Array[Array[Double]] = majElements.toArray map dataToWorkWith
    val minClasses: Array[Any] = minElements.toArray map classesToWorkWith
    val majClasses: Array[Any] = majElements.toArray map classesToWorkWith
    val selectedMajElements: Array[Int] = if (version == 1) {
      majElements.map { i: Int =>
        val result: (Any, Array[Int], Array[Double]) = if (dist == Distance.EUCLIDEAN) {
          nnRule(minNeighbours, dataToWorkWith(i), i, minClasses, 3, "nearest")
        } else {
          nnRuleHVDM(minNeighbours, dataToWorkWith(i), i, minClasses, 3, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter, "nearest")
        }
        (i, (result._2 map result._3).sum / result._2.length)
      }.sortBy(_._2).map(_._1)
    } else if (version == 2) {
      majElements.map { i: Int =>
        val result: (Any, Array[Int], Array[Double]) = if (dist == Distance.EUCLIDEAN) {
          nnRule(minNeighbours, dataToWorkWith(i), i, minClasses, 3, "farthest")
        } else {
          nnRuleHVDM(minNeighbours, dataToWorkWith(i), i, minClasses, 3, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter, "farthest")
        }
        (i, (result._2 map result._3).sum / result._2.length)
      }.sortBy(_._2).map(_._1)
    } else if (version == 3) {
      // We shuffle the data because, at last, we are going to take, at least, minElements.length * ratio elements and if
      // we don't shuffle, we only take majority elements examples that are near to the first minority class examples
      new Random(seed).shuffle(minElements.flatMap { i: Int =>
        if (dist == Distance.EUCLIDEAN) {
          nnRule(majNeighbours, dataToWorkWith(i), i, majClasses, nNeighbours, "nearest")._2
        } else {
          nnRuleHVDM(majNeighbours, dataToWorkWith(i), i, majClasses, nNeighbours, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter, "nearest")._2
        }
      }.distinct.toList).toArray
    } else {
      throw new Exception("Invalid argument: version should be: 1, 2 or 3")
    }

    val finalIndex: Array[Int] = minElements ++ selectedMajElements.take((minElements.length * ratio).toInt)
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}
