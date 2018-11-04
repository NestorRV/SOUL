package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Neighbourhood Cleaning Rule. Original paper: "Improving Identification of Difficult Small Classes by Balancing Class
  * Distribution" by J. Laurikkala.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of DistanceType representing the distance to be used
  * @param k          number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param threshold  consider a class to be undersampled if the number of instances of this class is
  *                   greater than data.size * threshold
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class NCL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), dist: DistanceType = Distance(euclideanDistance),
          k: Int = 3, threshold: Double = 0.5, val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {
  /** Compute the NCL algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    // Note: the notation used to refers the subsets of data is the used in the original paper.
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
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

    val (attrCounter, attrClassesCounter, sds) = if (dist.isInstanceOf[HVDM]) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val minorityIndex: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    val majorityIndex: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

    var i = 0
    while (i < classesToWorkWith.length) {
      if (classesToWorkWith(i) == untouchableClass) minorityIndex += i else majorityIndex += i
      i += 1
    }

    val ennData = new Data(toXData((majorityIndex map dataToWorkWith).toArray), (majorityIndex map classesToWorkWith).toArray, None, data.fileInfo)
    ennData.processedData = (majorityIndex map dataToWorkWith).toArray
    val enn = new ENN(ennData, dist = dist, k = k)
    val resultENN: Data = enn.compute()
    val indexA1: Array[Int] = resultENN.index.get map majorityIndex

    val uniqueMajClasses = (majorityIndex map classesToWorkWith).distinct
    val ratio: Double = dataToWorkWith.length * threshold

    def selectNeighbours(l: Int, targetClass: Any): ArrayBuffer[Int] = {
      val selected = new ArrayBuffer[Int]()
      val (label, nNeighbours, _) = dist match {
        case distance: Distance =>
          nnRule(dataToWorkWith, dataToWorkWith(l), l, classesToWorkWith, k, distance, "nearest")
        case _ =>
          nnRuleHVDM(dataToWorkWith, dataToWorkWith(l), l, classesToWorkWith, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
      }

      if (label != targetClass) {
        nNeighbours.foreach { n =>
          val nNeighbourClass: Any = classesToWorkWith(n)
          if (nNeighbourClass != untouchableClass && counter(nNeighbourClass) > ratio) {
            selected += n
          }
        }
      }
      selected
    }

    var j = 0
    val indexA2 = new Array[ArrayBuffer[Int]](minorityIndex.length)
    while (j < uniqueMajClasses.length) {
      val targetClass: Any = classesToWorkWith(j)
      minorityIndex.zipWithIndex.par.foreach { l =>
        indexA2(l._2) = selectNeighbours(l._1, targetClass)
      }
      j += 1
    }

    val finalIndex: Array[Int] = classesToWorkWith.indices.diff((indexA1 ++ indexA2.flatten.toList).toList).toArray
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
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