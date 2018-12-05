package soul.algorithm.undersampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.math.{max, min}

/** Undersampling Based on Clustering. Original paper: "Under-Sampling Approaches for Improving Prediction of the
  * Minority Class in an Imbalanced Dataset" by Show-Jane Yen and Yue-Shi Lee.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param method        selection method to apply. Possible options: random, NearMiss1, NearMiss2, NearMiss3, MostDistant and MostFar
  * @param m             ratio used in the SSize calculation
  * @param k             number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param numClusters   number of clusters to be created by KMeans core
  * @param restarts      number of times to relaunch KMeans core
  * @param minDispersion stop KMeans core if dispersion is lower than this value
  * @param maxIterations number of iterations to be done in KMeans core
  * @param dist          object of Distance enumeration representing the distance to be used
  * @param normalize     normalize the data or not
  * @param randomData    iterate through the data randomly or not
  * @param verbose       choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class SBC(data: Data, seed: Long = System.currentTimeMillis(), method: String = "random", m: Double = 1.0, k: Int = 3, numClusters: Int = 50,
          restarts: Int = 1, minDispersion: Double = 0.0001, maxIterations: Int = 200, val dist: Distance = Distance.EUCLIDEAN,
          normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the SBC algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)
    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val classesToWorkWith: Array[Any] = if (randomData) {
      val randomIndex: List[Int] = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val (_, centroids, assignment) = kMeans(dataToWorkWith, data.fileInfo.nominal, numClusters, restarts, minDispersion, maxIterations, seed)
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

    val majorityElements: Array[Int] = if (method.equals("random")) {
      sSizes.filter(_._2 != 0).flatMap { clusterIdSize: (Int, Int) =>
        random.shuffle(assignment(clusterIdSize._1).toList).filter((e: Int) =>
          classesToWorkWith(e) != untouchableClass).take(clusterIdSize._2)
      }
    } else {
      sSizes.filter(_._2 != 0).flatMap { clusteridSize: (Int, Int) =>
        val majorityElementsIndex: Array[(Int, Int)] = assignment(clusteridSize._1).zipWithIndex.filter((e: (Int, Int)) =>
          classesToWorkWith(e._1) != untouchableClass)

        // If no minority class elements are assigned to the cluster
        if (majorityElementsIndex.length == assignment(clusteridSize._1).length) {
          // Use the centroid as "minority class" element
          val distances: Array[Double] = assignment(clusteridSize._1).map { instance: Int =>
            euclidean(dataToWorkWith(instance), centroids(clusteridSize._1))
          }

          distances.zipWithIndex.sortBy(_._2).take(clusteridSize._2).map(_._2) map assignment(clusteridSize._1)
        } else {
          val minorityElementsIndex: Array[(Int, Int)] = assignment(clusteridSize._1).zipWithIndex.filter((e: (Int, Int)) =>
            classesToWorkWith(e._1) == untouchableClass)
          val majorityElementsIndex: Array[(Int, Int)] = assignment(clusteridSize._1).zipWithIndex.filter((e: (Int, Int)) =>
            classesToWorkWith(e._1) != untouchableClass)

          val minNeighbours: Array[Array[Double]] = minorityElementsIndex.unzip._2 map dataToWorkWith
          val majNeighbours: Array[Array[Double]] = majorityElementsIndex.unzip._2 map dataToWorkWith
          val minClasses: Array[Any] = minorityElementsIndex.unzip._2 map classesToWorkWith
          val majClasses: Array[Any] = majorityElementsIndex.unzip._2 map classesToWorkWith

          val KDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
            Some(new KDTree(minNeighbours, minClasses, dataToWorkWith(0).length))
          } else {
            None
          }

          val majorityKDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
            Some(new KDTree(majNeighbours, majClasses, dataToWorkWith(0).length))
          } else {
            None
          }

          val reverseKDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
            Some(new KDTree(minNeighbours, minClasses, dataToWorkWith(0).length, which = "farthest"))
          } else {
            None
          }

          if (method.equals("NearMiss1")) {
            // selects the majority class samples whose average distances to k nearest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { i: (Int, Int) =>
              if (dist == Distance.EUCLIDEAN) {
                val index = KDTree.get.nNeighbours(dataToWorkWith(i._1), k)._3
                (i._1, index.map(j => euclidean(dataToWorkWith(i._1), dataToWorkWith(j))).sum / index.length)
              } else {
                val result: (Any, Array[Int], Array[Double]) = nnRuleHVDM(minNeighbours, dataToWorkWith(i._1), -1, minClasses, k,
                  data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
                (i._1, (result._2 map result._3).sum / result._2.length)
              }
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map(_._1)
          } else if (method.equals("NearMiss2")) {
            // selects the majority class samples whose average distances to k farthest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { i: (Int, Int) =>
              if (dist == Distance.EUCLIDEAN) {
                val index = reverseKDTree.get.nNeighbours(dataToWorkWith(i._1), k)._3
                (i._1, index.map(j => euclidean(dataToWorkWith(i._1), dataToWorkWith(j))).sum / index.length)
              } else {
                val result: (Any, Array[Int], Array[Double]) = nnRuleHVDM(minNeighbours, dataToWorkWith(i._1), -1, minClasses, k,
                  data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "farthest")
                (i._1, (result._2 map result._3).sum / result._2.length)
              }
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map(_._1)
          } else if (method.equals("NearMiss3")) {
            // selects the majority class samples whose average distances to the closest minority class samples in the ith cluster are the smallest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { i: (Int, Int) =>
              if (dist == Distance.EUCLIDEAN) {
                val index = majorityKDTree.get.nNeighbours(dataToWorkWith(i._1), k)._3
                (i._1, index.map(j => euclidean(dataToWorkWith(i._1), dataToWorkWith(j))).sum / index.length)
              } else {
                val result: (Any, Array[Int], Array[Double]) = nnRuleHVDM(majNeighbours, dataToWorkWith(i._1), -1, majClasses, k,
                  data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
                (i._1, (result._2 map result._3).sum / result._2.length)
              }
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map(_._1)
          } else if (method.equals("MostDistant")) {
            // selects the majority class samples whose average distances to M closest minority class samples in the ith cluster are the farthest.
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { i: (Int, Int) =>
              if (dist == Distance.EUCLIDEAN) {
                val index = KDTree.get.nNeighbours(dataToWorkWith(i._1), k)._3
                (i._1, index.map(j => euclidean(dataToWorkWith(i._1), dataToWorkWith(j))).sum / index.length)
              } else {
                val result: (Any, Array[Int], Array[Double]) = nnRuleHVDM(minNeighbours, dataToWorkWith(i._1), -1, minClasses, k,
                  data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
                (i._1, (result._2 map result._3).sum / result._2.length)
              }
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).reverse.take(clusteridSize._2).map(_._1)
          } else if (method.equals("MostFar")) {
            // selects the majority class samples whose average distances to all minority class samples in the cluster are the farthest
            val meanDistances: Array[(Int, Double)] = majorityElementsIndex.map { i: (Int, Int) =>
              if (dist == Distance.EUCLIDEAN) {
                val index = KDTree.get.nNeighbours(dataToWorkWith(i._1), minorityElementsIndex.length)._3
                (i._1, index.map(j => euclidean(dataToWorkWith(i._1), dataToWorkWith(j))).sum / index.length)
              } else {
                val result: (Any, Array[Int], Array[Double]) = nnRuleHVDM(minNeighbours, dataToWorkWith(i._1), -1, minClasses,
                  minorityElementsIndex.length, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")
                (i._1, (result._2 map result._3).sum / result._2.length)
              }
            }
            meanDistances.sortBy((pair: (Int, Double)) => pair._2).take(clusteridSize._2).map(_._1)
          } else {
            throw new Exception("Invalid argument: method should be: random, NearMiss1, NearMiss2, NearMiss3, MostDistant or MostFar")
          }
        }
      }
    }

    val finalIndex: Array[Int] = minorityElements.distinct ++ majorityElements.distinct
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}