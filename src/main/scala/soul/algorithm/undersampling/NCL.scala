package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** Neighbourhood Cleaning Rule. Original paper: "Improving Identification of Difficult Small Classes by Balancing Class
  * Distribution" by J. Laurikkala.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param file       file to store the log. If its set to None, log process would not be done
  * @param distance   distance to use when calling the NNRule
  * @param k          number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param threshold  consider a class to be undersampled if the number of instances of this class is
  *                   greater than data.size * threshold
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class NCL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3, threshold: Double = 0.5,
          val normalize: Boolean = false, val randomData: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the NCL algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    // Note: the notation used to refers the subsets of data is the used in the original paper.
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

    val enn = new ENN(data, file = None, distance = distance, k = k)
    val resultENN: Data = enn.compute()
    val indexA1: Array[Int] = classesToWorkWith.indices.diff(resultENN.index.toList).toArray
    val minorityClassIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (c, i) if c == untouchableClass => i }
    val targetNeighbours: Array[Array[Double]] = minorityClassIndex map dataToWorkWith
    val classes: Array[Any] = minorityClassIndex map classesToWorkWith
    val (predictedLabels, selectedNeighbours): (Array[Any], Array[Array[Int]]) = minorityClassIndex.map { i: Int =>
      nnRule(neighbours = targetNeighbours, instance = dataToWorkWith(i), id = i, labels = classes, k = k, distance = distance,
        nominal = data.fileInfo.nominal, sds = sds, attrCounter = attrCounter, attrClassesCounter = attrClassesCounter)
    }.unzip
    val neighbourhoodBooleanIndex: Array[Boolean] = (predictedLabels zip (minorityClassIndex map classesToWorkWith)).map((c: (Any, Any)) => c._1 != c._2)
    val finalNeighbours: Array[Int] = (boolToIndex(neighbourhoodBooleanIndex) map selectedNeighbours).flatten.distinct
    val classesToUnderSample: Array[Any] = counter.collect { case (c, num_values) if c != untouchableClass &&
      num_values > dataToWorkWith.length * threshold => c
    }.toArray
    val indexA2: Array[AnyVal] = finalNeighbours.map((i: Int) => if (classesToUnderSample.indexOf(classesToWorkWith(i)) != -1) i)
    val finalIndex: Array[Int] = classesToWorkWith.indices.diff((indexA1 ++ indexA2).toList).toArray
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
