package soul.algorithm.oversampling

import soul.algorithm.Algorithm
import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** MWMOTE algorithm. Original paper: "MWMOTE—Majority Weighted Minority Oversampling Technique for Imbalanced Data Set
  * Learning" by Sukarna Barua, Md. Monirul Islam, Xin Yao, Fellow, IEEE, and Kazuyuki Muras.
  *
  * @param data data to work with
  * @param seed seed to use. If it is not provided, it will use the system time
  * @author David López Pretel
  */
class MWMOTE(private[soul] val data: Data,
             override private[soul] val seed: Long = System.currentTimeMillis()) extends Algorithm {

  //data with the samples
  private var samples: Array[Array[Double]] = data._processedData
  private var distanceType: Distances.Distance = Distances.EUCLIDEAN

  /** cut-off function
    *
    * @param value value to be checked
    * @param cut   cut-off value
    * @return the value cutted-off
    */
  private def f(value: Double, cut: Double): Double = {
    if (value < cut) value else cut
  }

  /**
    * Compute the closeness factor
    *
    * @param x    index of one node to work with
    * @param y    index of another node to work with
    * @param Nmin neighbors of 'y' necessaries to calculate the closeness factor
    * @return the closeness factor
    */

  private def Cf(y: (Int, Int), x: Int, Nmin: Array[Array[Int]]): Double = {
    val cut: Double = 5 // values used in the paper
    val CMAX: Double = 2

    if (!Nmin(y._2).contains(x))
      f(samples(0).length / computeDistanceOversampling(samples(y._1), samples(x), distanceType, data._nominal.length == 0,
        (samples, data._originalClasses)), cut) * CMAX
    else
      0.0
  }

  /** Compute the information weight
    *
    * @param x     index of one node to work with
    * @param y     index of another node to work with
    * @param Nmin  neighbors of 'y' necessaries to calculate the closeness factor
    * @param Simin informative minority set necessary to calculate density factor
    * @return the information weight
    */
  private def Iw(y: (Int, Int), x: Int, Nmin: Array[Array[Int]], Simin: Array[Int]): Double = {
    val cf = Cf(y, x, Nmin)
    val df = cf / Simin.map(Cf(y, _, Nmin)).sum
    cf + df
  }

  /**
    * compute distance between two clusters based in the distance between their centroids
    *
    * @param cluster1 index of the elements in the cluster
    * @param cluster2 index of the elements in the cluster
    * @return distance between the two clusters
    */
  private def clusterDistance(cluster1: Array[Int], cluster2: Array[Int]): Double = {
    val centroid1: Array[Double] = (cluster1 map samples).transpose.map(_.sum / cluster1.length)
    val centroid2: Array[Double] = (cluster2 map samples).transpose.map(_.sum / cluster2.length)
    computeDistanceOversampling(centroid1, centroid2, distanceType, data._nominal.length == 0,
      (Array.concat(cluster1, cluster2) map samples, Array.concat(cluster1, cluster2) map data._originalClasses))
  }

  /**
    * search the minimum distance between each pair of clusters
    *
    * @param cluster clusters to work with
    * @return index of the two clusters and the distance
    */
  private def minDistance(cluster: ArrayBuffer[ArrayBuffer[Int]]): (Int, Int, Double) = {
    var minDist: (Int, Int, Double) = (0, 0, 99999999)
    cluster.indices.foreach(i => cluster.indices.foreach(j => {
      if (i != j) {
        val dist = clusterDistance(cluster(i).toArray, cluster(j).toArray)
        if (dist < minDist._3) minDist = (i, j, dist)
      }
    }))
    minDist
  }

  /**
    * average-linkage agglomerative clustering
    *
    * @param Sminf necessary to compute the cut-off Th
    * @return the data divided in clusters
    */
  private def cluster(Sminf: Array[Int]): Array[Array[Int]] = {
    val dist: Array[Array[Double]] = Array.fill(Sminf.length, Sminf.length)(9999999.0)
    Sminf.indices.foreach(i => Sminf.indices.foreach(j => if (i != j) dist(i)(j) = computeDistanceOversampling(samples(Sminf(i)),
      samples(Sminf(j)), distanceType, data._nominal.length == 0, (Sminf map samples, Sminf map data._originalClasses))))

    val Cp: Double = 3 // used in paper
    val Th: Double = dist.map(_.min).sum / Sminf.length * Cp
    var minDist: (Int, Int, Double) = (0, 0, 0.0)
    val clusters: ArrayBuffer[ArrayBuffer[Int]] = Sminf.map(ArrayBuffer(_)).to[ArrayBuffer]
    while (minDist._3 < Th) {
      //compute the min distance between each cluster
      minDist = minDistance(clusters)
      //merge the two more proximal clusters
      clusters(minDist._1) ++= clusters(minDist._2)
      clusters -= clusters(minDist._2)
    }

    clusters.map(_.toArray).toArray
  }

  /** Compute the MWMOTE algorithm
    *
    * @param file  file to store the log. If its set to None, log process would not be done
    * @param N     number of synthetic samples to be generated
    * @param k1    number of neighbors used for predicting noisy minority class samples
    * @param k2    number of majority neighbors used for constructing informative minority set
    * @param k3    number of minority neighbors used for constructing informative minority set
    * @param dType the type of distance to use, hvdm or euclidean
    * @return synthetic samples generated
    */
  def compute(file: Option[String] = None, N: Int = 500, k1: Int = 5, k2: Int = 5, k3: Int = 5,
              dType: Distances.Distance = Distances.EUCLIDEAN): Unit = {
    if (dType != Distances.EUCLIDEAN && dType != Distances.HVDM) {
      throw new Exception("The distance must be euclidean or hvdm")
    }

    // Start the time
    val initTime: Long = System.nanoTime()

    distanceType = dType
    if (dType == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data._originalClasses)
    data._minorityClass = data._originalClasses(minorityClassIndex(0))
    // compute majority class
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // construct the filtered minority set
    val Sminf: Array[Int] = minorityClassIndex.map(index => {
      val neighbors = kNeighbors(samples, index, k1, dType, data._nominal.length == 0, (samples, data._originalClasses))
      if (neighbors map data._originalClasses contains data._originalClasses(minorityClassIndex(0))) {
        Some(index)
      } else {
        None
      }
    }).filterNot(_.forall(_ == None)).map(_.get)

    //for each sample in Sminf compute the nearest majority set
    val Sbmaj: Array[Int] = Sminf.flatMap(x => kNeighbors(majorityClassIndex map samples, samples(x), k2, dType,
      data._nominal.length == 0, (majorityClassIndex map samples, majorityClassIndex map data._originalClasses))).distinct.map(majorityClassIndex(_))
    // for each majority example in Sbmaj , compute the nearest minority set
    val Nmin: Array[Array[Int]] = Sbmaj.map(x => kNeighbors(minorityClassIndex map samples, samples(x), k3, dType,
      data._nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data._originalClasses)).map(minorityClassIndex(_)))

    // find the informative minority set (union of all Nmin)
    val Simin: Array[Int] = Nmin.flatten.distinct
    // for each sample in Simin compute the selection weight
    val Sw: Array[Double] = Simin.map(x => Sbmaj.zipWithIndex.map(y => Iw(y, x, Nmin, Simin)).sum)
    val sumSw: Double = Sw.sum
    // convert each Sw into probability
    val Sp: Array[(Double, Int)] = Sw.map(_ / sumSw).zip(Simin).sortWith(_._1 > _._1)

    // compute the clusters
    val clusters: Array[Array[Int]] = cluster(minorityClassIndex) // cluster => index to processedData
    val clustersIndex: Map[Int, Int] = clusters.zipWithIndex.flatMap(c => {
      clusters(c._2).map(index => (index, c._2))
    }).toMap // index to processedData => cluster

    //output data
    val output: Array[Array[Double]] = Array.fill(N, samples(0).length)(0.0)

    val probsSum: Double = Sp.map(_._1).sum
    val r: Random = new Random(this.seed)

    (0 until N).foreach(i => {
      // select a sample, then select another randomly from the cluster that have this sample
      val x = chooseByProb(Sp, probsSum, r)
      val y = clusters(clustersIndex(x))(r.nextInt(clusters(clustersIndex(x)).length))
      // compute atributtes of the sample
      samples(0).indices.foreach(atrib => {
        val diff: Double = samples(y)(atrib) - samples(x)(atrib)
        val gap: Float = r.nextFloat
        output(i)(atrib) = samples(x)(atrib) + gap * diff
      })
    })

    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + output.length).indices.toList).toArray
    // check if the data is nominal or numerical
    if (data._nominal.length == 0) {
      data._resultData = dataShuffled map to2Decimals(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output))
    } else {
      data._resultData = dataShuffled map toNominal(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output), data._nomToNum)
    }
    data._resultClasses = dataShuffled map Array.concat(data._originalClasses, Array.fill(output.length)(data._minorityClass))

    // Stop the time
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      this.logger.addMsg("ORIGINAL SIZE: %d".format(data._originalData.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(data._resultData.length))
      this.logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) this.logger.addMsg("%d".format(index._2)))
      // Save the time
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))

      // Save the log
      this.logger.storeFile(file.get + "_Mwmote")
    }
  }
}
