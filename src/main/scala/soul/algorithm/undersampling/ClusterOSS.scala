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
  * @param distance      distance to use when calling the NNRule
  * @param k             number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param numClusters   number of clusters to be created by KMeans core
  * @param restarts      number of times to relaunch KMeans core
  * @param minDispersion stop KMeans core if dispersion is lower than this value
  * @param maxIterations number of iterations to be done in KMeans core
  * @param normalize     normalize the data or not
  * @param randomData    iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class ClusterOSS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
                 distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3, numClusters: Int = 15, restarts: Int = 5,
                 minDispersion: Double = 0.0001, maxIterations: Int = 100, val normalize: Boolean = false, val randomData: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** Compute the ClusterOSS algorithm
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

    val majElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }
    val (_, centroids, assignment) = kMeans(data = majElements map dataToWorkWith, nominal = data.fileInfo.nominal,
      numClusters = numClusters, restarts = restarts, minDispersion = minDispersion, maxIterations = maxIterations, seed = seed)

    val result: (Array[Int], Array[Array[Int]]) = assignment.par.map { cluster: (Int, Array[Int]) =>
      val distances: Array[(Int, Double)] = cluster._2.map { instance: Int =>
        (instance, euclideanDistance(dataToWorkWith(instance), centroids(cluster._1)))
      }

      val closestInstance: Int = if (distances.isEmpty) -1 else distances.minBy((_: (Int, Double))._2)._1
      (closestInstance, cluster._2.diff(List(closestInstance)))
    }.toArray.unzip

    // Remove foo values
    val train: Array[Int] = result._1.diff(List(-1))
    // Flatten all the clusters
    val test: Array[Int] = result._2.flatten
    val neighbours: Array[Array[Double]] = test map dataToWorkWith
    val classes: Array[Any] = test map classesToWorkWith
    val calculatedLabels: Array[(Int, Any)] = test.map { i: Int =>
      (i, nnRule(neighbours = neighbours, instance = dataToWorkWith(i), id = i, labels = classes, k = 1, distance = distance, nominal = data.fileInfo.nominal,
        sds = sds, attrCounter = attrCounter, attrClassesCounter = attrClassesCounter)._1)
    }

    // if the label matches (it is well classified) the element is useful
    val misclassified: Array[Int] = calculatedLabels.collect { case (i, label) if label != classesToWorkWith(i) => i }
    val newDataIndex: Array[Int] = misclassified ++ train

    // Construct a data object to be passed to Tomek Link
    val auxData: Data = new Data(x = toXData(newDataIndex map dataToWorkWith),
      y = newDataIndex map classesToWorkWith, fileInfo = data.fileInfo)
    auxData.processedData = newDataIndex map dataToWorkWith
    val tl = new TL(auxData, file = None, distance = distance)
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