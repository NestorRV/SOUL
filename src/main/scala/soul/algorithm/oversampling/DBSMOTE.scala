package soul.algorithm.oversampling

import soul.algorithm.Algorithm
import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** DBSMOTE algorithm. Original paper: "DBSMOTE: Density-Based Synthetic Minority Over-sampling Technique" by
  * Chumphol Bunkhumpornpat, Krung Sinapiromsaran and Chidchanok Lursinsap.
  *
  * @param data data to work with
  * @author David López Pretel
  */
class DBSMOTE(private[soul] val data: Data) extends Algorithm {
  // the data of the samples
  private var samples: Array[Array[Double]] = data._processedData
  private var distanceType: Distances.Distance = Distances.EUCLIDEAN
  // compute minority class
  private val minorityClassIndex: Array[Int] = minority(data._originalClasses)

  /** Compute the neighborhood (used in DBScan algorithm)
    *
    * @param point the point to get their neighbors
    * @param eps   epsilon to indicate the distance that must be between two points
    * @return cluster generated
    */
  private def regionQuery(point: Int, eps: Double): Array[Int] = {
    (minorityClassIndex map samples).indices.map(sample => {
      if (computeDistanceOversampling(samples(minorityClassIndex(point)), samples(minorityClassIndex(sample)), distanceType, data._nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data._originalClasses)) <= eps) {
        Some(sample)
      } else {
        None
      }
    }).filterNot(_.forall(_ == None)).map(_.get).toArray
  }

  /** expand the cluster for a point (used in DBScan algorithm)
    *
    * @param point      the point to get the cluster
    * @param clusterId  id to represent each cluster
    * @param clusterIds array of Ids which represent where must be each point
    * @param eps        epsilon to indicate the distance that must be between two points
    * @return cluster generated
    */
  private def expandCluster(point: Int, clusterId: Int, clusterIds: Array[Int], eps: Double, minPts: Int): Boolean = {
    val neighbors: ArrayBuffer[Int] = ArrayBuffer(regionQuery(point, eps): _*)
    if (neighbors.length < minPts) {
      clusterIds(point) = -2 //noise
      return false
    } else {
      neighbors.foreach(clusterIds(_) = clusterId)
      clusterIds(point) = clusterId

      var numNeighbors: Int = neighbors.length
      for (current <- 0 until numNeighbors) {
        val neighborsOfCurrent: Array[Int] = regionQuery(current, eps)
        if (neighborsOfCurrent.length >= minPts) {
          neighborsOfCurrent.foreach(neighbor => {
            if (clusterIds(neighbor) == -1 || clusterIds(neighbor) == -2) { //Noise or Unclassified
              if (clusterIds(neighbor) == -1) { //Unclassified
                neighbors += neighbor
                numNeighbors += 1
              }
              clusterIds(neighbor) = clusterId
            }
          })
        }
      }
    }

    true
  }

  /** Compute the DBSCAN algorithm
    *
    * @param eps    epsilon to indicate the distance that must be between two points
    * @param minPts number of neighbors
    * @return cluster generated, index from the minority class
    */
  private def dbscan(eps: Double, minPts: Int): Array[Array[Int]] = {
    var clusterId: Int = 0
    val clusterIds: Array[Int] = Array.fill(minorityClassIndex.length)(-1)
    minorityClassIndex.indices.foreach(point => {
      if (clusterIds(point) == -1) {
        if (expandCluster(point, clusterId, clusterIds, eps, minPts)) {
          clusterId += 1
        }
      }
    })

    if (clusterId != 0) {
      val clusters: Array[Array[Int]] = Array.fill(clusterId)(Array())
      (0 until clusterId).foreach(i => {
        clusters(i) = clusterIds.zipWithIndex.filter(_._1 == i).map(_._2)
      })
      clusters
    } else { // the cluster is all the data
      Array(Array.range(0, minorityClassIndex.length))
    }
  }

  /** Build a graph
    *
    * @param cluster array with the index that is associated to the cluster
    * @param eps     epsilon to indicate the distance that must be between two points
    * @param minPts  number of neighbor
    * @return graph, it is a matrix of boolean, true indicates that the nodes are connected
    */
  private def buildGraph(cluster: Array[Int], eps: Double, minPts: Int): Array[Array[Boolean]] = {
    val graph: Array[Array[Boolean]] = Array.fill(cluster.length, cluster.length)(false)
    //distance between each par of nodes
    val distances: Array[Array[Double]] = cluster.map(i => cluster.map(j => computeDistanceOversampling(samples(minorityClassIndex(i)), samples(minorityClassIndex(j)), distanceType, data._nominal.length == 0, (cluster map samples, cluster map data._originalClasses))))

    // number of nodes connected to another which satisfied distance(a,b) <= eps
    val NNq: Array[Int] = distances.map(row => row.map(dist => if (dist <= eps) 1 else 0)).map(_.sum)

    //build the graph
    cluster.indices.foreach(i => {
      if (cluster.length >= minPts + 1) {
        distances(i).zipWithIndex.foreach(dist => {
          if (dist._1 <= eps && dist._1 > 0 && NNq(dist._2) >= minPts) {
            graph(i)(dist._2) = true
          }
        })
      } else {
        distances(i).zipWithIndex.foreach(dist => {
          if (dist._1 <= eps && dist._1 > 0) {
            graph(i)(dist._2) = true
          }
        })
      }
    })
    graph
  }

  /** Compute the dijsktra algorithm
    *
    * @param graph   each row has 1 if the node is connected to another and 0 if not
    * @param source  node to start the search
    * @param target  node to reach
    * @param cluster index to get access to the original data to compute the distance between two neighbor
    * @return shortest path
    */
  private def dijsktra(graph: Array[Array[Boolean]], source: Int, target: Int, cluster: Array[Int]): Array[Int] = {
    // distance from source to node, prev node, node visited or not
    val nodeInfo: Array[(Double, Int, Boolean)] = Array.fill(graph.length)((9999999, -1, false))
    nodeInfo(source) = (0.0, source, false)

    val findMin = (x: ((Double, Int, Boolean), Int), y: ((Double, Int, Boolean), Int)) => if ((x._1._1 < y._1._1 && !x._1._3) || (!x._1._3 && y._1._3)) x else y

    nodeInfo.indices.foreach(_ => {
      val u: Int = nodeInfo.zipWithIndex.reduceLeft(findMin)._2 //vertex with min distance
      nodeInfo(u) = (nodeInfo(u)._1, nodeInfo(u)._2, true)
      if (u == target) { // return shortest path
        val shortestPath: ArrayBuffer[Int] = ArrayBuffer()
        var current = target
        while (current != source) {
          shortestPath += current
          current = nodeInfo(current)._2
        }
        shortestPath += current
        return shortestPath.toArray
      }
      graph(u).indices.foreach(v => {
        if (graph(u)(v) && !nodeInfo(v)._3) {
          val alt = nodeInfo(u)._1 + computeDistanceOversampling(samples(minorityClassIndex(cluster(u))), samples(minorityClassIndex(cluster(v))), distanceType, data._nominal.length == 0, (cluster map samples, cluster map data._originalClasses))
          if (alt < nodeInfo(v)._1) nodeInfo(v) = (alt, u, nodeInfo(v)._3)
        }
      })
    })

    throw new Exception("Path not found")
  }

  /** Compute the DensityBasedSmote algorithm
    *
    * @param file  file to store the log. If its set to None, log process would not be done
    * @param eps   epsilon to indicate the distance that must be between two points
    * @param k     number of neighbors
    * @param dType the type of distance to use, hvdm or euclidean
    * @param seed  seed for the random
    * @return synthetic samples generated
    */
  def compute(file: Option[String] = None, eps: Double = -1, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {
    if (dType != Distances.EUCLIDEAN && dType != Distances.HVDM) {
      throw new Exception("The distance must be euclidean or hvdm")
    }

    // Start the time
    val initTime: Long = System.nanoTime()

    data._minorityClass = data._originalClasses(minorityClassIndex(0))
    distanceType = dType
    if (dType == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    //check if the user pass the epsilon parameter
    var eps2 = eps
    if (eps == -1) {
      eps2 = samples.map(i => samples.map(j => computeDistanceOversampling(i, j, dType, data._nominal.length == 0, (samples, data._originalClasses))).sum).sum / (samples.length * samples.length)
    }

    //compute the clusters using dbscan
    val clusters: Array[Array[Int]] = dbscan(eps2, k)

    //the output of the algorithm
    val output: Array[Array[Double]] = Array.fill(clusters.map(_.length).sum, samples(0).length)(0)

    //for each cluster
    clusters.foreach(c => {
      //build a graph with the data of each cluster
      val graph: Array[Array[Boolean]] = buildGraph(c, eps2, k)
      val r: Random.type = scala.util.Random
      r.setSeed(seed)
      var newIndex: Int = 0
      //compute pseudo-centroid, centroid is the mean of the cluster
      val centroid = (c map samples).transpose.map(_.sum / c.length)
      var pseudoCentroid: (Int, Double) = (0, 99999999.0)
      //the pseudo-centroid is the sample that is closest to the centroid
      (c map samples).zipWithIndex.foreach(sample => {
        val distance = computeDistanceOversampling(sample._1, centroid, dType, data._nominal.length == 0, (c map samples, c map data._originalClasses))
        if (distance < pseudoCentroid._2) pseudoCentroid = (sample._2, distance)
      })

      c.indices.foreach(p => {
        //compute the shortest path between the pseudo centroid and the samples in each cluster
        val shortestPath: Array[Int] = dijsktra(graph, p, pseudoCentroid._1, c)
        //a random sample in the path
        val e = r.nextInt(shortestPath.length)
        //get the nodes connected by e, then only the two first will be used
        val v1_v2: Array[(Boolean, Int)] = graph(shortestPath(e)).zipWithIndex.filter(_._1 == true)
        samples(0).indices.foreach(attrib => {
          // v1(attrib) - v2(attrib)
          val dif: Double = samples(minorityClassIndex(c(v1_v2(1)._2)))(attrib) - samples(minorityClassIndex(c(v1_v2(0)._2)))(attrib)
          val gap: Double = r.nextFloat()
          // v1(attrib) + gap * dif
          output(newIndex)(attrib) = samples(minorityClassIndex(c(v1_v2(0)._2)))(attrib) + gap * dif
        })
        newIndex += 1
      })
    })

    val r: Random.type = scala.util.Random
    r.setSeed(seed)
    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + output.length).indices.toList).toArray
    // check if the data is nominal or numerical
    if (data._nominal.length == 0) {
      data._resultData = dataShuffled map to2Decimals(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output))
    } else {
      data._resultData = dataShuffled map toNominal(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output), data._nomToNum)
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
      this.logger.storeFile(file.get + "_DBSmote")
    }
  }
}