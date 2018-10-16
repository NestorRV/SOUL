package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.math.min

/** Class Purity Maximization core. Original paper: "An Unsupervised Learning Approach to Resolving the
  * Data Imbalanced Issue in Supervised Learning Problems in Functional Genomics" by Kihoon Yoon and Stephen Kwek.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param distance distance to use when calling the NNRule core
  * @author Néstor Rodríguez Vico
  */
class CPM(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          distance: Distances.Distance = Distances.EUCLIDEAN) {

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
  private[soul] val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, data.fileInfo.nominal, data.y)
  private[soul] val centers: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

  /** Compute the CPM algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val posElements: Int = counter.head._2
    val negElements: Int = counter.tail.values.sum
    val impurity: Double = posElements.asInstanceOf[Double] / negElements.asInstanceOf[Double]
    val cluster: Array[Int] = new Array[Int](dataToWorkWith.length).indices.toArray

    purityMaximization(impurity, cluster, 0)

    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (centers.toArray map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (centers.toArray map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.addMsg("NEW DATA SIZE: %d".format(centers.toArray.length))
      logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (centers.toArray.length.toFloat / dataToWorkWith.length) * 100))
      logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }

  /** Purity maximization method
    *
    * @param parentImpurity impurity of the parent cluster
    * @param parentCluster  elements in the parent cluster
    * @param center         center of the cluster
    */
  private[soul] def purityMaximization(parentImpurity: Double, parentCluster: Array[Int], center: Int): Unit = {
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

      parentCluster.foreach((element: Int) => if (distances(element)(center1) < distances(element)(center2))
        cluster1 += element else cluster2 += element)

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
}
