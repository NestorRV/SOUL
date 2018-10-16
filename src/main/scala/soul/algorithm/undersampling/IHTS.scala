package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._
import weka.classifiers.trees.J48
import weka.core.Instances

import scala.util.Random


/** Instance Hardness Threshold. Original paper: "An Empirical Study of Instance Hardness" by Michael R. Smith,
  * Tony Martinez and Christophe Giraud-Carrier.
  *
  * @param data   data to work with
  * @param seed   seed to use. If it is not provided, it will use the system time
  * @param file   file to store the log. If its set to None, log process would not be done
  * @param nFolds number of subsets to create when applying cross-validation
  * @author Néstor Rodríguez Vico
  */
class IHTS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None, nFolds: Int = 5) {

  private[soul] val minorityClass: Any = -1
  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if minorityClass is set to -1.
  // Otherwise, minorityClass will be used as the minority one
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val randomIndex: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, _) = processData(data)
  // Use randomized data
  val dataToWorkWith: Array[Array[Double]] = (randomIndex map processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (randomIndex map data.y).toArray

  /** Compute InstanceHardnessThreshold algorithm
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val random: Random = new Random(seed)
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

    val index: Array[Int] = (finalIndex map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}
