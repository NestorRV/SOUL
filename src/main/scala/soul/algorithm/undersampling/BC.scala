package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Balance Cascade algorithm. Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu,
  * Jianxin Wu and Zhi-Hua Zhou.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param dist        object of DistanceType representing the distance to be used
  * @param k           number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param nMaxSubsets maximum number of subsets to generate
  * @param nFolds      number of subsets to create when applying cross-validation
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @param verbose     choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class BC(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         dist: DistanceType = Distance(euclideanDistance), k: Int = 3, nMaxSubsets: Int = 5, nFolds: Int = 5,
         ratio: Double = 1.0, val normalize: Boolean = false, val randomData: Boolean = false, val verbose: Boolean = false) {

  /** Compute the BC algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
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

    var search: Boolean = true
    var subsetsCounter: Int = 0
    val mask: Array[Boolean] = Array.fill(classesToWorkWith.length)(true)
    val subsets: ArrayBuffer[Array[Int]] = new ArrayBuffer[Array[Int]](0)
    val minorityElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    val majorityElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

    while (search) {
      val indexToUnderSample: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val minorityIndex: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val classesCounter: Map[Any, Int] = (boolToIndex(mask) map classesToWorkWith).groupBy(identity).mapValues(_.length)

      classesCounter.foreach { target: (Any, Int) =>
        val indexClass: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (c, i) if c == target._1 => i }
        if (target._1 != untouchableClass) {
          val sameClassBool: Array[Boolean] = mask.zipWithIndex.collect { case (c, i) if classesToWorkWith(i) == target._1 => c }
          val indexClassInterest: Array[Int] = boolToIndex(sameClassBool) map indexClass
          val indexTargetClass: List[Int] = random.shuffle((indexClassInterest map classesToWorkWith).indices.toList).take(counter(untouchableClass))
          indexToUnderSample ++= (indexTargetClass map indexClassInterest)
          majorityElements ++= (indexTargetClass map indexClassInterest)
        } else {
          minorityIndex ++= indexClass
          minorityElements ++= indexClass
        }
      }

      subsetsCounter += 1
      val subset: Array[Int] = (indexToUnderSample ++ minorityIndex).toArray
      subsets += subset

      val classesToWorkWithSubset: Array[Any] = subset map classesToWorkWith
      val dataToWorkWithSubset: Array[Array[Double]] = subset map dataToWorkWith
      val prediction: Array[Any] = (dist match {
        case distance: Distance =>
          kFoldPrediction(dataToWorkWithSubset, classesToWorkWithSubset, k, nFolds, distance, "nearest")
        case _ =>
          kFoldPredictionHVDM(dataToWorkWithSubset, classesToWorkWithSubset, k, nFolds, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter, "nearest")
      }).take(indexToUnderSample.length)

      val classifiedInstances: Array[Boolean] = ((indexToUnderSample.indices map classesToWorkWithSubset)
        zip prediction).map((e: (Any, Any)) => e._1 == e._2).toArray
      (boolToIndex(classifiedInstances) map indexToUnderSample).foreach((i: Int) => mask(i) = false)

      if (subsetsCounter == nMaxSubsets) search = false

      val finalTargetStats: Map[Any, Int] = (boolToIndex(mask) map classesToWorkWith).groupBy(identity).mapValues(_.length)
      classesToWorkWith.distinct.filter((c: Any) => c != untouchableClass).foreach { c: Any =>
        if (finalTargetStats(c) < counter(untouchableClass)) search = false
      }
    }

    val majorityIndexHistogram: Array[(Int, Int)] = majorityElements.groupBy(identity).mapValues(_.length).toArray.sortBy(_._2).reverse
    val majorityIndex: Array[Int] = majorityIndexHistogram.take((minorityElements.distinct.length * ratio).toInt).map(_._1)
    val finalIndex: Array[Int] = minorityElements.distinct.toArray ++ majorityIndex
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}