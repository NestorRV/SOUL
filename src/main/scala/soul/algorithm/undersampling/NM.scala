package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** NearMiss. Original paper: "kNN Approach to Unbalanced Data Distribution: A Case Study involving Information
  * Extraction" by Jianping Zhang and Inderjeet Mani.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param file        file to store the log. If its set to None, log process would not be done
  * @param distance    distance to use when calling the NNRule core
  * @param version     version of the core to execute
  * @param nNeighbours number of neighbours to take for each minority example (only used if version is set to 3)
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @author Néstor Rodríguez Vico
  */
class NM(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         distance: Distances.Distance = Distances.EUCLIDEAN, version: Int = 1, nNeighbours: Int = 3, ratio: Double = 1.0) {


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
  // Use normalized data for EUCLIDEAN distance and randomized data
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (randomIndex map zeroOneNormalization(data, processedData)).toArray else (randomIndex map processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (randomIndex map data.y).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, data.fileInfo.nominal, data.y)

  /** Compute NearMiss algorithm
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val majElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }
    val minElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val selectedMajElements: Array[Int] = if (version == 1) {
      majElements.map { instance: Int =>
        val result: (Any, Array[Int]) = nnRule(distances = distances(instance), selectedElements = minElements,
          labels = classesToWorkWith, k = 3)
        (instance, (result._2 map distances(instance)).sum / result._2.length)
      }.sortBy((_: (Int, Double))._2).map((_: (Int, Double))._1)
    } else if (version == 2) {
      majElements.map { instance: Int =>
        val result: (Any, Array[Int]) = nnRule(distances = distances(instance), selectedElements = minElements,
          labels = classesToWorkWith, k = 3, which = "farthest")
        (instance, (result._2 map distances(instance)).sum / result._2.length)
      }.sortBy((_: (Int, Double))._2).map((_: (Int, Double))._1)
    } else if (version == 3) {
      // We shuffle the data because, at last, we are going to take, at least, minElements.length * ratio elements and if
      // we don't shuffle, we only take majority elements examples that are near to the first minority class examples
      new Random(seed).shuffle(minElements.flatMap { instance: Int =>
        nnRule(distances = distances(instance), selectedElements = majElements,
          labels = classesToWorkWith, k = nNeighbours)._2
      }.distinct.toList).toArray
    } else {
      throw new Exception("Invalid argument: version should be: 1, 2 or 3")
    }

    val finalIndex: Array[Int] = minElements ++ selectedMajElements.take((minElements.length * ratio).toInt)
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
