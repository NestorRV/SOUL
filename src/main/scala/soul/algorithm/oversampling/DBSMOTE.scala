package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** DBSMOTE algorithm. Original paper: "DBSMOTE: Density-Based Synthetic Minority Over-sampling Technique" by
  * Chumphol Bunkhumpornpat, Krung Sinapiromsaran and Chidchanok Lursinsap.
  *
  * @author David López Pretel
  */
class DBSMOTE() {

  /** Compute the DBSMOTE algorithm
    *
    * @param data      data to work with
    * @param eps       epsilon to indicate the distance that must be between two points
    * @param k         number of neighbors
    * @param dist      object of Distance enumeration representing the distance to be used
    * @param seed      seed to use. If it is not provided, it will use the system time
    * @param normalize normalize the data or not
    * @param verbose   choose to display information about the execution or not
    * @return synthetic samples generated
    */
  def compute(data: Data, eps: Double = -1, k: Int = 5, dist: Distance = Distance.EUCLIDEAN,
              seed: Long = 5, normalize: Boolean = false, verbose: Boolean = false): Data = {
    val initTime: Long = System.nanoTime()
    val minorityClassIndex: Array[Int] = minority(data.y)
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    def regionQuery(point: Int, eps: Double): Array[Int] = {
      (minorityClassIndex map samples).indices.map(sample => {
        val D: Double = if (dist == Distance.EUCLIDEAN) {
          euclidean(samples(minorityClassIndex(point)), samples(minorityClassIndex(sample)))
        } else {
          HVDM(samples(minorityClassIndex(point)), samples(minorityClassIndex(sample)), data.fileInfo.nominal, sds,
            attrCounter, attrClassesCounter)
        }
        if (D <= eps) {
          Some(sample)
        } else {
          None
        }
      }).filterNot(_.forall(_ == None)).map(_.get).toArray
    }

    def expandCluster(point: Int, clusterId: Int, clusterIds: Array[Int], eps: Double, minPts: Int): Boolean = {
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

    def dbscan(eps: Double, minPts: Int): Array[Array[Int]] = {
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

    def buildGraph(cluster: Array[Int], eps: Double, minPts: Int): Array[Array[Boolean]] = {
      val graph: Array[Array[Boolean]] = Array.fill(cluster.length, cluster.length)(false)
      //distance between each pair of nodes
      val distances: Array[Array[Double]] = cluster.map { i =>
        cluster.map { j =>
          if (dist == Distance.EUCLIDEAN) {
            euclidean(samples(minorityClassIndex(i)), samples(minorityClassIndex(j)))
          } else {
            HVDM(samples(minorityClassIndex(i)), samples(minorityClassIndex(j)), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }

        }
      }

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

    def dijsktra(graph: Array[Array[Boolean]], source: Int, target: Int, cluster: Array[Int]): Array[Int] = {
      // distance from source to node, prev node, node visited or not
      val nodeInfo: Array[(Double, Int, Boolean)] = Array.fill(graph.length)((9999999, -1, false))
      nodeInfo(source) = (0.0, source, false)

      val findMin = (x: ((Double, Int, Boolean), Int), y: ((Double, Int, Boolean), Int)) =>
        if ((x._1._1 < y._1._1 && !x._1._3) || (!x._1._3 && y._1._3)) x else y

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
            val d: Double = if (dist == Distance.EUCLIDEAN) {
              euclidean(samples(minorityClassIndex(cluster(u))),
                samples(minorityClassIndex(cluster(v))))
            } else {
              HVDM(samples(minorityClassIndex(cluster(u))), samples(minorityClassIndex(cluster(v))), data.fileInfo.nominal,
                sds, attrCounter, attrClassesCounter)
            }
            val alt = nodeInfo(u)._1 + d
            if (alt < nodeInfo(v)._1) nodeInfo(v) = (alt, u, nodeInfo(v)._3)
          }
        })
      })

      throw new Exception("Path not found")
    }

    val minorityClass: Any = data.y(minorityClassIndex(0))
    //check if the user pass the epsilon parameter
    var eps2 = eps
    if (eps == -1) {
      eps2 = samples.map { i =>
        samples.map { j =>
          if (dist == Distance.EUCLIDEAN) {
            euclidean(i, j)
          } else {
            HVDM(i, j, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }
        }.sum
      }.sum / (samples.length * samples.length)
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
        val d: Double = if (dist == Distance.EUCLIDEAN) {
          euclidean(sample._1, centroid)
        } else {
          HVDM(sample._1, centroid, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
        }
        if (d < pseudoCentroid._2) pseudoCentroid = (sample._2, d)
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

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(data.x.length + output.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
  }
}
