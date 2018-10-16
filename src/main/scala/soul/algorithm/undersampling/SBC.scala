package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.math.{max, min}
import scala.util.Random

/** Undersampling Based on Clustering core. Original paper: "Under-Sampling Approaches for Improving Prediction of the
  * Minority Class in an Imbalanced Dataset" by Show-Jane Yen and Yue-Shi Lee.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param file          file to store the log. If its set to None, log process would not be done
  * @param method        selection method to apply. Possible options: random, NearMiss1, NearMiss2, NearMiss3, MostDistant and MostFar
  * @param m             ratio used in the SSize calculation
  * @param k             number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param numClusters   number of clusters to be created by KMeans core
  * @param restarts      number of times to relaunch KMeans core
  * @param minDispersion stop KMeans core if dispersion is lower than this value
  * @param maxIterations number of iterations to be done in KMeans core
  * @author Néstor Rodríguez Vico
  */
class SBC(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          method: String = "random", m: Double = 1.0, k: Int = 3, numClusters: Int = 50, restarts: Int = 1,
          minDispersion: Double = 0.0001, maxIterations: Int = 200) {

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
  val dataToWorkWith: Array[Array[Double]] = (randomIndex map processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (randomIndex map data.y).toArray

  /** Undersampling method based in SBC
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val (_, centroids, assignment) = kMeans(data = dataToWorkWith, nominal = data.fileInfo.nominal, numClusters = numClusters, restarts = restarts,
      minDispersion = minDispersion, maxIterations = maxIterations, seed = seed)
    val minMajElements: List[(Int, Int)] = (0 until numClusters).toList.map { cluster: Int =>
      val elements = assignment(cluster)
      val minElements: Int = (elements map classesToWorkWith).count((c: Any) => c == untouchableClass)
      (minElements, elements.length - minElements)
    }
    val nPos: Double = minMajElements.unzip._2.sum.toDouble
    val sizeK: Double = minMajElements.map((pair: (Int, Int)) => pair._2.toDouble / max(pair._1, 1)).sum
    val sSizes: Array[(Int, Int)] = assignment.map { element: (Int, Array[Int]) =>
      val ratio: (Int, Int) = minMajElements(element._1)
      // The min is to prevent infinity values if no minority elements are added to the cluster
      (element._1, min(m * nPos * ((ratio._2.toDouble / (ratio._1 + 1)) / sizeK), ratio._2).toInt)
    }.toArray
    val minorityElements: Array[Int] = assignment.flatMap((element: (Int, Array[Int])) => element._2.filter((index: Int) =>
      classesToWorkWith(index) == untouchableClass)).toArray

    val random: Random = new util.Random(seed)
    val majorityElements: Array[Int] = if (method.equals("random")) {
      sSizes.filter((_: (Int, Int))._2 != 0).flatMap { clusteridSize: (Int, Int) =>
        random.shuffle(assignment(clusteridSize._1).toList).filter((e: Int) =>
          classesToWorkWith(e) != untouchableClass).take(clusteridSize._2)
      }
    } else {
      sSizes.filter((_: (Int, Int))._2 != 0).flatMap { clusteridSize: (Int, Int) =>
        val majorityElementsIndex: Array[(Int, Int)] = assignment(clusteridSize._1).zipWithIndex.filter((e: (Int, Int)) =>
          classesToWorkWith(e._1) != untouchableClass)

        // If no minority class elements are assigned to the cluster
        if (majorityElementsIndex.length == assignment(clusteridSize._1).length) {
          // Use the centroid as "minority class" element
          val distances: Array[Double] = assignment(clusteridSize._1).map { instance: Int =>
            if (data.fileInfo.nominal.length == 0)
              euclideanDistance(dataToWorkWith(instance), centroids(clusteridSize._1))
            else
              euclideanNominalDistance(dataToWorkWith(instance), centroids(clusteridSize._1), data.fileInfo.nominal)
          }

          distances.zipWithIndex.sortBy((_: (Double, Int))._2).take(clusteridSize._2).map((_: (Double, Int))._2) map assignment(clusteridSize._1)
        } else {
          val minorityElementsIndex: Array[(Int, Int)] = assignment(clusteridSize._1).zipWithIndex.filter((e: (Int, Int)) =>
            classesToWorkWith(e._1) == untouchableClass)

          val distances: Array[Array[Double]] = computeDistances(data = assignment(clusteridSize._1) map dataToWorkWith,
            distance = Distances.EUCLIDEAN, nominal = data.fileInfo.nominal, classes = assignment(clusteridSize._1) map classesToWorkWith)

          if (method.equals("NearMiss1")) {
            // selects the majority class samples whose average distances to k nearest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { majElement: (Int, Int) =>
              val result: (Any, Array[Int]) = nnRule(distances = distances(majElement._2),
                selectedElements = minorityElementsIndex.unzip._2, labels = classesToWorkWith, k = k)
              (majElement._1, (result._2 map distances(majElement._2)).sum / result._2.length)
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map((_: (Int, Double))._1)
          } else if (method.equals("NearMiss2")) {
            // selects the majority class samples whose average distances to k farthest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { majElement: (Int, Int) =>
              val result: (Any, Array[Int]) = nnRule(distances = distances(majElement._2),
                selectedElements = minorityElementsIndex.unzip._2,
                labels = classesToWorkWith, k = k, which = "farthest")
              (majElement._1, (result._2 map distances(majElement._2)).sum / result._2.length)
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map((_: (Int, Double))._1)
          } else if (method.equals("NearMiss3")) {
            // selects the majority class samples whose average distances to the closest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { majElement: (Int, Int) =>
              val result: (Any, Array[Int]) = nnRule(distances = distances(majElement._2),
                selectedElements = minorityElementsIndex.unzip._2, labels = classesToWorkWith, k = 1)
              (majElement._1, (result._2 map distances(majElement._2)).sum / result._2.length)
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map((_: (Int, Double))._1)
          } else if (method.equals("MostDistant")) {
            // selects the majority class samples whose average distances to M closest minority class samples in the ith cluster are the farthest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { majElement: (Int, Int) =>
              val result: (Any, Array[Int]) = nnRule(distances = distances(majElement._2),
                selectedElements = minorityElementsIndex.unzip._2, labels = classesToWorkWith, k = k)
              (majElement._1, (result._2 map distances(majElement._2)).sum / result._2.length)
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).reverse.take(clusteridSize._2).map((_: (Int, Double))._1)
          } else if (method.equals("MostFar")) {
            // selects the majority class samples whose average distances to all minority class samples in the cluster are the farthest
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { majElement: (Int, Int) =>
              val result: (Any, Array[Int]) = nnRule(distances = distances(majElement._2),
                selectedElements = minorityElementsIndex.unzip._2, labels = classesToWorkWith,
                k = minorityElementsIndex.unzip._2.length)
              (majElement._1, (result._2 map distances(majElement._2)).sum / result._2.length)
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map((_: (Int, Double))._1)
          } else {
            throw new Exception("Invalid argument: method should be: random, NearMiss1, NearMiss2, NearMiss3, MostDistant or MostFar")
          }
        }
      }
    }

    val finalIndex: Array[Int] = minorityElements.distinct ++ majorityElements.distinct
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