package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** ClusterOSS. Original paper: "ClusterOSS: a new undersampling method for imbalanced learning."
  * by Victor H Barella, Eduardo P Costa and André C P L F Carvalho.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param file          file to store the log. If its set to None, log process would not be done
  * @param distance      distance to use when calling the NNRule core
  * @param k             number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param numClusters   number of clusters to be created by KMeans core
  * @param restarts      number of times to relaunch KMeans core
  * @param minDispersion stop KMeans core if dispersion is lower than this value
  * @param maxIterations number of iterations to be done in KMeans core
  * @author Néstor Rodríguez Vico
  */
class ClusterOSS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
                 distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3, numClusters: Int = 15, restarts: Int = 5, minDispersion: Double = 0.0001, maxIterations: Int = 100) {

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
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (randomIndex map zeroOneNormalization(data, processedData)).toArray else (randomIndex map processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (randomIndex map data.y).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, data.fileInfo.nominal, data.y)

  /** Undersampling method based in ClusterOSS
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val majElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }
    val (_, centroids, assignment) = kMeans(data = majElements map dataToWorkWith, nominal = data.fileInfo.nominal,
      numClusters = numClusters, restarts = restarts, minDispersion = minDispersion, maxIterations = maxIterations, seed = seed)

    val result: (Array[Int], Array[Array[Int]]) = assignment.par.map { cluster: (Int, Array[Int]) =>
      val distances: Array[(Int, Double)] = cluster._2.map { instance: Int =>
        if (data.fileInfo.nominal.length == 0)
          (instance, euclideanDistance(dataToWorkWith(instance), centroids(cluster._1)))
        else
          (instance, euclideanNominalDistance(dataToWorkWith(instance), centroids(cluster._1), data.fileInfo.nominal))
      }

      val closestInstance: Int = if (distances.isEmpty) -1 else distances.minBy((_: (Int, Double))._2)._1
      (closestInstance, cluster._2.diff(List(closestInstance)))
    }.toArray.unzip

    // Remove foo values
    val train: Array[Int] = result._1.diff(List(-1))
    // Flatten all the clusters
    val test: Array[Int] = result._2.flatten

    val calculatedLabels: Array[(Int, Any)] = test.map { testInstance: Int =>
      (testInstance, nnRule(distances = distances(testInstance), selectedElements = train, labels = classesToWorkWith, k = 1)._1)
    }

    // if the label matches (it is well classified) the element is useful
    val misclassified: Array[Int] = calculatedLabels.collect { case (i, label) if label != classesToWorkWith(i) => i }
    val newDataIndex: Array[Int] = misclassified ++ train

    // Construct a data object to be passed to Tomek Link
    val auxData: Data = new Data(x = toXData(newDataIndex map dataToWorkWith),
      y = newDataIndex map classesToWorkWith, fileInfo = data.fileInfo)
    val tl = new TL(auxData, file = None, distance = distance, dists = Some((newDataIndex map distances).map(newDataIndex map _)))
    tl.untouchableClass_=(untouchableClass)
    val resultTL: Data = tl.compute()
    // The final randomIndex is the result of applying Tomek Link to the content of newDataIndex
    val finalIndex: Array[Int] = (resultTL.index.get.toList map newDataIndex).toArray
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