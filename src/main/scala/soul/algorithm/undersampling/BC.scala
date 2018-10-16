package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Balance Cascade algorithm. Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu,
  * Jianxin Wu and Zhi-Hua Zhou.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param file        file to store the log. If its set to None, log process would not be done
  * @param distance    distance to use when calling the NNRule core
  * @param k           number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param nMaxSubsets maximum number of subsets to generate
  * @param nFolds      number of subsets to create when applying cross-validation
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @author Néstor Rodríguez Vico
  */
class BC(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3, nMaxSubsets: Int = 5, nFolds: Int = 5, ratio: Double = 1.0) {

  private[soul] val minorityClass: Any = -1
  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = this.data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if minorityClass is set to -1.
  // Otherwise, minorityClass will be used as the minority one
  private[soul] val untouchableClass: Any = this.counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.y.indices.toList)
  // Use randomized data
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (this.index map zeroOneNormalization(this.data)).toArray else (this.index map this.data.processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.data.y).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, this.data.fileInfo.nominal, this.data.y)

  /** Compute the Balance Cascade core.
    *
    * @return array of data structures with all the important information and index of elements kept for each subset
    */

  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val random: Random = new Random(this.seed)
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
        if (target._1 != this.untouchableClass) {
          val sameClassBool: Array[Boolean] = mask.zipWithIndex.collect { case (c, i) if classesToWorkWith(i) == target._1 => c }
          val indexClassInterest: Array[Int] = boolToIndex(sameClassBool) map indexClass
          val indexTargetClass: List[Int] = random.shuffle((indexClassInterest map classesToWorkWith).indices.toList).take(this.counter(this.untouchableClass))
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
      val prediction: Array[Any] = kFoldPrediction(classesToWorkWithSubset, distances = distances, k = k,
        nFolds = nFolds).take(indexToUnderSample.length)

      val classifiedInstances: Array[Boolean] = ((indexToUnderSample.indices map classesToWorkWithSubset)
        zip prediction).map((e: (Any, Any)) => e._1 == e._2).toArray
      (boolToIndex(classifiedInstances) map indexToUnderSample).foreach((i: Int) => mask(i) = false)

      if (subsetsCounter == nMaxSubsets) search = false

      val finalTargetStats: Map[Any, Int] = (boolToIndex(mask) map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      classesToWorkWith.distinct.filter((c: Any) => c != this.untouchableClass).foreach { c: Any =>
        if (finalTargetStats(c) < this.counter(this.untouchableClass)) search = false
      }
    }

    val majorityIndexHistogram: Array[(Int, Int)] = majorityElements.groupBy(identity).mapValues((_: ArrayBuffer[Int]).length).toArray.sortBy((_: (Int, Int))._2).reverse
    val majorityIndex: Array[Int] = majorityIndexHistogram.take((minorityElements.distinct.length * ratio).toInt).map((_: (Int, Int))._1)
    val finalIndex: Array[Int] = minorityElements.distinct.toArray ++ majorityIndex
    val finishTime: Long = System.nanoTime()

    this.data.index = (finalIndex map this.index).sorted
    this.data.resultData = this.data.index map this.data.x
    this.data.resultClasses = this.data.index map this.data.y

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      this.logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(this.counter, this.untouchableClass)))
      this.logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, this.untouchableClass)))
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      this.logger.storeFile(file.get)
    }

    this.data
  }
}