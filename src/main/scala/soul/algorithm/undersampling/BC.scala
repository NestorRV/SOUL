package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Balance Cascade algorithm. Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu,
  * Jianxin Wu and Zhi-Hua Zhou.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param dist        distance to be used. It should be "HVDM" or a function of the type: (Array[Double], Array[Double]) => Double.
  * @param k           number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param nMaxSubsets maximum number of subsets to generate
  * @param nFolds      number of subsets to create when applying cross-validation
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class BC(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         dist: Any = Utilities.euclideanDistance _, k: Int = 3, nMaxSubsets: Int = 5, nFolds: Int = 5,
         ratio: Double = 1.0, val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {

  private[soul] val distance: Distances.Distance = getDistance(dist)

  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the BC algorithm.
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

    var search: Boolean = true
    var subsetsCounter: Int = 0
    val mask: Array[Boolean] = Array.fill(classesToWorkWith.length)(true)
    val subsets: ArrayBuffer[Array[Int]] = new ArrayBuffer[Array[Int]](0)
    val minorityElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    val majorityElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

    while (search) {
      val indexToUnderSample: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val minorityIndex: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val classesCounter: Map[Any, Int] = (boolToIndex(mask) map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)

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
      val prediction: Array[Any] = (if (distance == Distances.USER) {
        kFoldPrediction(dataToWorkWithSubset, classesToWorkWithSubset, k, nFolds, dist, "nearest")
      } else {
        kFoldPredictionHVDM(dataToWorkWithSubset, classesToWorkWithSubset, k, nFolds, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
      }).take(indexToUnderSample.length)

      val classifiedInstances: Array[Boolean] = ((indexToUnderSample.indices map classesToWorkWithSubset)
        zip prediction).map((e: (Any, Any)) => e._1 == e._2).toArray
      (boolToIndex(classifiedInstances) map indexToUnderSample).foreach((i: Int) => mask(i) = false)

      if (subsetsCounter == nMaxSubsets) search = false

      val finalTargetStats: Map[Any, Int] = (boolToIndex(mask) map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      classesToWorkWith.distinct.filter((c: Any) => c != untouchableClass).foreach { c: Any =>
        if (finalTargetStats(c) < counter(untouchableClass)) search = false
      }
    }

    val majorityIndexHistogram: Array[(Int, Int)] = majorityElements.groupBy(identity).mapValues((_: ArrayBuffer[Int]).length).toArray.sortBy((_: (Int, Int))._2).reverse
    val majorityIndex: Array[Int] = majorityIndexHistogram.take((minorityElements.distinct.length * ratio).toInt).map((_: (Int, Int))._1)
    val finalIndex: Array[Int] = minorityElements.distinct.toArray ++ majorityIndex
    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (finalIndex map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

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