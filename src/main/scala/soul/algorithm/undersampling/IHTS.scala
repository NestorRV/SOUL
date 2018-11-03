package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._
import weka.classifiers.trees.J48
import weka.core.Instances


/** Instance Hardness Threshold. Original paper: "An Empirical Study of Instance Hardness" by Michael R. Smith,
  * Tony Martinez and Christophe Giraud-Carrier.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param nFolds     number of subsets to create when applying cross-validation
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class IHTS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), nFolds: Int = 5,
           val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {

  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the IHTS algorithm.
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

    // Each element is the index of test elements
    val indices: Array[Array[Int]] = random.shuffle(classesToWorkWith.indices.toList).toArray.grouped((classesToWorkWith.length.toFloat / nFolds).ceil.toInt).toArray
    val probabilities: Array[Double] = new Array[Double](classesToWorkWith.length)

    indices.foreach { testIndex: Array[Int] =>
      val trainIndex: Array[Int] = classesToWorkWith.indices.diff(testIndex).toArray

      val j48: J48 = new J48
      j48.setOptions(Array("-U", "-M", "1"))

      val trainInstances: Instances = buildInstances(data = trainIndex map dataToWorkWith,
        classes = trainIndex map classesToWorkWith, fileInfo = data.fileInfo)
      val testInstances: Instances = buildInstances(data = testIndex map dataToWorkWith,
        classes = testIndex map classesToWorkWith, fileInfo = data.fileInfo)

      j48.buildClassifier(trainInstances)

      val probs: Array[Array[Double]] = testIndex.indices.map((i: Int) => j48.distributionForInstance(testInstances.instance(i))).toArray
      val classes: Array[Any] = (testIndex map classesToWorkWith).distinct
      val values: Array[Double] = (testIndex map classesToWorkWith).zipWithIndex.map((e: (Any, Int)) => probs(e._2)(classes.indexOf(e._1)))

      (testIndex zip values).foreach((i: (Int, Double)) => probabilities(i._1) = i._2)
    }

    val finalIndex: Array[Int] = classesToWorkWith.distinct.flatMap { targetClass: Any =>
      val indexTargetClass: Array[Int] = if (targetClass != untouchableClass) {
        val nSamples: Int = counter(untouchableClass)
        val targetIndex: Array[Int] = boolToIndex(classesToWorkWith.map((c: Any) => c == targetClass))
        val targetProbabilities: Array[Double] = targetIndex map probabilities
        val percentile: Double = (1.0 - (nSamples / counter(targetClass))) * 100.0
        val threshold: Double = targetProbabilities.sorted.apply(math.ceil((targetProbabilities.length - 1) * (percentile / 100.0)).toInt)
        boolToIndex((targetIndex map probabilities).map((e: Double) => e >= threshold))
      }
      else {
        classesToWorkWith.filter((c: Any) => c == targetClass).indices.toArray
      }

      indexTargetClass map boolToIndex(classesToWorkWith.map((c: Any) => c == targetClass))
    }

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
