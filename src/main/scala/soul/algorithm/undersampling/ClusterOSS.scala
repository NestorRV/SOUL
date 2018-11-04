package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities._

/** ClusterOSS. Original paper: "ClusterOSS: a new undersampling method for imbalanced learning."
  * by Victor H Barella, Eduardo P Costa and André C P L F Carvalho.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param dist          object of DistanceType representing the distance to be used
  * @param k             number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param numClusters   number of clusters to be created by KMeans core
  * @param restarts      number of times to relaunch KMeans core
  * @param minDispersion stop KMeans core if dispersion is lower than this value
  * @param maxIterations number of iterations to be done in KMeans core
  * @param normalize     normalize the data or not
  * @param randomData    iterate through the data randomly or not
  * @param verbose       choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class ClusterOSS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
                 dist: DistanceType = Distance(euclideanDistance), k: Int = 3, numClusters: Int = 15, restarts: Int = 5, minDispersion: Double = 0.0001,
                 maxIterations: Int = 100, val normalize: Boolean = false, val randomData: Boolean = false, val verbose: Boolean = false) {

  /** Compute the ClusterOSS algorithm
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

    val majElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label != untouchableClass => i }
    val (_, centroids, assignment) = kMeans(data = majElements map dataToWorkWith, nominal = data.fileInfo.nominal,
      numClusters = numClusters, restarts = restarts, minDispersion = minDispersion, maxIterations = maxIterations, seed = seed)

    val result: (Array[Int], Array[Array[Int]]) = assignment.par.map { cluster: (Int, Array[Int]) =>
      val distances: Array[(Int, Double)] = cluster._2.map { instance: Int =>
        (instance, euclideanDistance(dataToWorkWith(instance), centroids(cluster._1)))
      }

      val closestInstance: Int = if (distances.isEmpty) -1 else distances.minBy(_._2)._1
      (closestInstance, cluster._2.diff(List(closestInstance)))
    }.toArray.unzip

    // Remove foo values
    val train: Array[Int] = result._1.diff(List(-1))
    // Flatten all the clusters
    val test: Array[Int] = result._2.flatten
    val neighbours: Array[Array[Double]] = test map dataToWorkWith
    val classes: Array[Any] = test map classesToWorkWith
    val calculatedLabels: Array[(Int, Any)] = test.map { i: Int =>
      val label: Any = dist match {
        case distance: Distance =>
          nnRule(neighbours, dataToWorkWith(i), i, classes, 1, distance, "nearet")._1
        case _ =>
          nnRuleHVDM(neighbours, dataToWorkWith(i), i, classes, 1, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter, "nearest")._1
      }
      (i, label)
    }

    // if the label matches (it is well classified) the element is useful
    val misclassified: Array[Int] = calculatedLabels.collect { case (i, label) if label != classesToWorkWith(i) => i }
    val newDataIndex: Array[Int] = misclassified ++ train

    // Construct a data object to be passed to Tomek Link
    val auxData: Data = new Data(x = toXData(newDataIndex map dataToWorkWith),
      y = newDataIndex map classesToWorkWith, fileInfo = data.fileInfo)
    auxData.processedData = newDataIndex map dataToWorkWith
    val tl = new TL(auxData, dist = dist, minorityClass = Some(untouchableClass))
    val resultTL: Data = tl.compute()
    // The final instances is the result of applying Tomek Link to the content of newDataIndex
    val finalIndex: Array[Int] = (resultTL.index.get.toList map newDataIndex).toArray
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}