package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** Easy Ensemble algorithm. Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu,
  * Jianxin Wu and Zhi-Hua Zhou.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param file        file to store the log. If its set to None, log process would not be done
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param replacement whether or not to sample randomly with replacement or not. false by default
  * @param nTimes      times to perform the random algorithm
  * @author Néstor Rodríguez Vico
  */
class EE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         ratio: Double = 1.0, replacement: Boolean = false, nTimes: Int = 5) {

  private[soul] val minorityClass: Any = -1
  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = this.data.originalClasses.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if minorityClass is set to -1.
  // Otherwise, minorityClass will be used as the minority one
  private[soul] val untouchableClass: Any = this.counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.originalClasses.indices.toList)
  // Use randomized data
  val dataToWorkWith: Array[Array[Double]] = (this.index map this.data.processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.data.originalClasses).toArray

  /** Compute the Easy Ensemble core.
    *
    * @return data structure with all the important information and index of elements kept
    */
  def compute(): Data = {
    // Start the time
    val initTime: Long = System.nanoTime()

    // The elements in the minority class (untouchableClass) are maintained
    val minorityIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == this.untouchableClass => i }
    // Get the index of the elements in the majority class
    val random: Random = new Random(this.seed)

    val majIndex: List[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != this.untouchableClass => i }.toList
    val majElements: Array[Int] = (0 until nTimes).flatMap { _: Int =>
      val majorityIndex: Array[Int] = random.shuffle(majIndex).toArray
      if (!replacement) majorityIndex.take((minorityIndex.length * ratio).toInt) else majorityIndex.indices.map((_: Int) =>
        random.nextInt(majorityIndex.length)).toArray map majorityIndex
    }.toArray

    // Make an histogram and select the majority class examples that have been selected more times
    val majorityIndexHistogram: Array[(Int, Int)] = majElements.groupBy(identity).mapValues((_: Array[Int]).length).toArray.sortBy((_: (Int, Int))._2).reverse

    val majorityIndex: Array[Int] = majorityIndexHistogram.take((minorityIndex.length * ratio).toInt).map((_: (Int, Int))._1)

    // Get the index of the reduced data
    val finalIndex: Array[Int] = minorityIndex ++ majorityIndex

    // Stop the time
    val finishTime: Long = System.nanoTime()

    this.data.index = (finalIndex map this.index).sorted
    this.data.resultData = this.data.index map this.data.originalData
    this.data.resultClasses = this.data.index map this.data.originalClasses

    if (file.isDefined) {
      // Recount of classes
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)

      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))

      this.logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(this.counter, this.untouchableClass)))
      // Recompute the Imbalanced Ratio
      this.logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, this.untouchableClass)))

      // Save the time
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))

      // Save the log
      this.logger.storeFile(file.get)
    }

    this.data
  }
}