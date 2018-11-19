package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities._

/** Easy Ensemble algorithm. Original paper: "Exploratory Undersampling for Class-Imbalance Learning" by Xu-Ying Liu,
  * Jianxin Wu and Zhi-Hua Zhou.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param replacement whether or not to sample randomly with replacement or not. false by default
  * @param nTimes      times to perform the random algorithm
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @param verbose     choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class EE(data: Data, seed: Long = System.currentTimeMillis(), ratio: Double = 1.0, replacement: Boolean = false, nTimes: Int = 5,
         normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the EE algorithm.
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

    val minorityIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val majIndex: List[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }.toList
    val majElements: Array[Int] = (0 until nTimes).flatMap { _: Int =>
      val majorityIndex: Array[Int] = random.shuffle(majIndex).toArray
      if (!replacement) majorityIndex.take((minorityIndex.length * ratio).toInt) else majorityIndex.indices.map(_ =>
        random.nextInt(majorityIndex.length)).toArray map majorityIndex
    }.toArray

    // Make an histogram and select the majority class examples that have been selected more times
    val majorityIndexHistogram: Array[(Int, Int)] = majElements.groupBy(identity).mapValues(_.length).toArray.sortBy(_._2).reverse
    val majorityIndex: Array[Int] = majorityIndexHistogram.take((minorityIndex.length * ratio).toInt).map(_._1)
    val finalIndex: Array[Int] = minorityIndex ++ majorityIndex
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