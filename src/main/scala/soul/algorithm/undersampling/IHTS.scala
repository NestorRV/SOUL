package soul.algorithm.undersampling

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
  * @param verbose    choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class IHTS(data: Data, seed: Long = System.currentTimeMillis(), nFolds: Int = 5,
           normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the IHTS algorithm.
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
        classesToWorkWith.zipWithIndex.collect { case (c, i) if c == targetClass => i }
      }

      indexTargetClass
    }

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
