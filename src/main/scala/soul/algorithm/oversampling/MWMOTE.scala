package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** MWMOTE algorithm. Original paper: "MWMOTE—Majority Weighted Minority Oversampling Technique for Imbalanced Data Set
  * Learning" by Sukarna Barua, Md. Monirul Islam, Xin Yao, Fellow, IEEE, and Kazuyuki Muras.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param N        number of synthetic samples to be generated
  * @param k1       number of neighbors used for predicting noisy minority class samples
  * @param k2       number of majority neighbors used for constructing informative minority set
  * @param k3       number of minority neighbors used for constructing informative minority set
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class MWMOTE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
             N: Int = 500, k1: Int = 5, k2: Int = 5, k3: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, nomToNum) = processData(data)
  //data with the samples
  private var samples: Array[Array[Double]] = processedData

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
      f(samples(0).length / computeDistanceOversampling(samples(y._1), samples(x), distance, data.fileInfo.nominal.length == 0,
        (samples, data.y)), cut) * CMAX
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
    computeDistanceOversampling(centroid1, centroid2, distance, data.fileInfo.nominal.length == 0,
      (Array.concat(cluster1, cluster2) map samples, Array.concat(cluster1, cluster2) map data.y))
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
      samples(Sminf(j)), distance, data.fileInfo.nominal.length == 0, (Sminf map samples, Sminf map data.y))))

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
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data, processedData)
    }
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))
    // compute majority class
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // construct the filtered minority set
    val Sminf: Array[Int] = minorityClassIndex.map(index => {
      val neighbors = kNeighbors(samples, index, k1, distance, data.fileInfo.nominal.length == 0, (samples, data.y))
      if (neighbors map data.y contains data.y(minorityClassIndex(0))) {
        Some(index)
      } else {
        None
      }
    }).filterNot(_.forall(_ == None)).map(_.get)

    //for each sample in Sminf compute the nearest majority set
    val Sbmaj: Array[Int] = Sminf.flatMap(x => kNeighbors(majorityClassIndex map samples, samples(x), k2, distance,
      data.fileInfo.nominal.length == 0, (majorityClassIndex map samples, majorityClassIndex map data.y))).distinct.map(majorityClassIndex(_))
    // for each majority example in Sbmaj , compute the nearest minority set
    val Nmin: Array[Array[Int]] = Sbmaj.map(x => kNeighbors(minorityClassIndex map samples, samples(x), k3, distance,
      data.fileInfo.nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data.y)).map(minorityClassIndex(_)))

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
    val r: Random = new Random(seed)

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
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      dataShuffled map to2Decimals(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      dataShuffled map toNominal(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), nomToNum)
    }, dataShuffled map Array.concat(data.y, Array.fill(output.length)(minorityClass)),
      Some(dataShuffled.zipWithIndex.collect { case (c, i) if c >= samples.length => i }), data.fileInfo)
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      logger.addMsg("NEW DATA SIZE: %d".format(newData.x.length))
      logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) logger.addMsg("%d".format(index._2)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}
