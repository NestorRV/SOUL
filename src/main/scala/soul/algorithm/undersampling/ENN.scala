package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Edited Nearest Neighbour rule. Original paper: "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"
  * by Dennis L. Wilson.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of DistanceType representing the distance to be used
  * @param k          number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class ENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          dist: DistanceType = Distance(euclideanDistance), k: Int = 3, val normalize: Boolean = false,
          val randomData: Boolean = false) extends LazyLogging {

  /** Compute the ENN algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
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

    val finalIndex = new ArrayBuffer[Int]()
    val uniqueClasses = classesToWorkWith.distinct

    var j = 0
    val majorityClassIndex = new ArrayBuffer[Int]()
    while (j < classesToWorkWith.length) {
      if (classesToWorkWith(j) == untouchableClass) finalIndex += j else majorityClassIndex += j
      j += 1
    }

    val kdTree: KDTree = new KDTree(dataToWorkWith, classesToWorkWith, dataToWorkWith(0).length)

    var i = 0
    while (i < uniqueClasses.length) {
      val targetClass = uniqueClasses(i)
      val selected: Array[(Int, Boolean)] = if (targetClass != untouchableClass) {
        majorityClassIndex.par.map { j =>
          val label = dist match {
            case distance: Distance =>
              mode(kdTree.nNeighbours(dataToWorkWith(j), k, leaveOneOut = true)._2.toArray)
            case _ =>
              nnRuleHVDM(dataToWorkWith, dataToWorkWith(j), j, classesToWorkWith, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")._1
          }

          (j, label == targetClass)
        }.toArray
      } else {
        new Array[(Int, Boolean)](0)
      }

      selected.foreach(e => if (e._2) finalIndex += e._1)

      i += 1
    }

    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex.toArray map data.x, finalIndex.toArray map data.y, Some(finalIndex.toArray), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (finalIndex.toArray map classesToWorkWith).groupBy(identity).mapValues(_.length)
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
