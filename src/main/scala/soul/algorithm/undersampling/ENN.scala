package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Edited Nearest Neighbour rule. Original paper: "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"
  * by Dennis L. Wilson.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param file       file to store the log. If its set to None, log process would not be done
  * @param distance   distance to use when calling the NNRule
  * @param k          number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class ENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3, val normalize: Boolean = false, val randomData: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the ENN algorithm.
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

    val finalIndex = new ArrayBuffer[Int]()
    val uniqueClasses = classesToWorkWith.distinct

    var j = 0
    val majorityClassIndex = new ArrayBuffer[Int]()
    while (j < classesToWorkWith.length) {
      if (classesToWorkWith(j) == untouchableClass) finalIndex += j else majorityClassIndex += j
      j += 1
    }

    var i = 0
    val neighbours: Array[Array[Double]] = (majorityClassIndex map dataToWorkWith).toArray
    val classes: Array[Any] = (majorityClassIndex map classesToWorkWith).toArray
    while (i < uniqueClasses.length) {
      val targetClass = uniqueClasses(i)
      if (targetClass != untouchableClass) {
        var j = 0
        while (j < majorityClassIndex.length) {
          val predictedLabel = nnRule(neighbours = neighbours, instance = dataToWorkWith(j), id = j, labels = classes, k = k,
            distance = distance, nominal = data.fileInfo.nominal, sds = sds, attrCounter = attrCounter, attrClassesCounter = attrClassesCounter)._1
          if (predictedLabel == targetClass)
            finalIndex += majorityClassIndex(j)
          j += 1
        }
      }

      i += 1
    }

    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (finalIndex.toArray map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex.toArray map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
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
