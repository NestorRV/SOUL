package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.math.min

/** Class Purity Maximization core. Original paper: "An Unsupervised Learning Approach to Resolving the
  * Data Imbalanced Issue in Supervised Learning Problems in Functional Genomics" by Kihoon Yoon and Stephen Kwek.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       distance to be used. It should be "HVDM" or a function of the type: (Array[Double], Array[Double]) => Double.
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class CPM(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
          dist: Any, val normalize: Boolean = false, val randomData: Boolean = false) extends LazyLogging {

  private[soul] val distance: Distances.Distance = getDistance(dist)
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
  private[soul] val centers: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

  /** Compute the CPM algorithm.
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

    val posElements: Int = counter.head._2
    val negElements: Int = counter.tail.values.sum
    val impurity: Double = posElements.asInstanceOf[Double] / negElements.asInstanceOf[Double]
    val cluster: Array[Int] = new Array[Int](dataToWorkWith.length).indices.toArray

    val (attrCounter, attrClassesCounter, sds) = if (distance == Distances.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues((_: Array[Double]).length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    def purityMaximization(parentImpurity: Double, parentCluster: Array[Int], center: Int): Unit = {
      val classes: Array[Any] = (randomIndex map data.y).toArray

      val cluster1: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val cluster2: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val posElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val negElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

      var center1: Int = 0
      var center2: Int = 0
      var pointer: Int = 0
      var impurity: Double = Double.PositiveInfinity
      var impurity1: Double = Double.PositiveInfinity
      var impurity2: Double = Double.PositiveInfinity

      parentCluster.foreach((f: Int) => if (classes(f) == untouchableClass) posElements += f else negElements += f)

      val pairs: ArrayBuffer[(Int, Int)] = for {x <- negElements; y <- posElements} yield (x, y)

      while (parentImpurity <= impurity) {
        if (pointer >= pairs.length) {
          centers += center
          return
        }

        center1 = pairs(pointer)._1
        center2 = pairs(pointer)._2

        parentCluster.foreach { element: Int =>
          val d1: Double = if (distance == Distances.USER) {
            dist.asInstanceOf[(Array[Double], Array[Double]) => Double](dataToWorkWith(element), dataToWorkWith(center1))
          } else {
            HVDM(dataToWorkWith(element), dataToWorkWith(center1), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }

          val d2: Double = if (distance == Distances.USER) {
            dist.asInstanceOf[(Array[Double], Array[Double]) => Double](dataToWorkWith(element), dataToWorkWith(center2))
          } else {
            HVDM(dataToWorkWith(element), dataToWorkWith(center2), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }

          if (d1 < d2)
            cluster1 += element else cluster2 += element
        }

        if (cluster1.nonEmpty)
          impurity1 = cluster1.count((element: Int) => classes(element) == untouchableClass).toDouble / cluster1.length
        else {
          centers += center2
          return
        }

        if (cluster2.nonEmpty)
          impurity2 = cluster2.count((element: Int) => classes(element) == untouchableClass).toDouble / cluster2.length
        else {
          centers += center1
          return
        }

        impurity = min(impurity1, impurity2)
        pointer += 1
      }

      purityMaximization(impurity1, cluster1.toArray, center1)
      purityMaximization(impurity2, cluster2.toArray, center2)
    }

    purityMaximization(impurity, cluster, 0)

    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (centers.toArray map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (centers.toArray map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.info("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.info("NEW DATA SIZE: %d".format(centers.toArray.length))
      logger.info("REDUCTION PERCENTAGE: %s".format(100 - (centers.toArray.length.toFloat / dataToWorkWith.length) * 100))
      logger.info("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.info("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
