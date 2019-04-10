/*
SOUL: Scala Oversampling and Undersampling Library.
Copyright (C) 2019 Néstor Rodríguez, David López

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation in version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** MWMOTE algorithm. Original paper: "MWMOTE—Majority Weighted Minority Oversampling Technique for Imbalanced Data Set
  * Learning" by Sukarna Barua, Md. Monirul Islam, Xin Yao, Fellow, IEEE, and Kazuyuki Muras.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param N         number of synthetic samples to be generated
  * @param k1        number of neighbors used for predicting noisy minority class samples
  * @param k2        number of majority neighbors used for constructing informative minority set
  * @param k3        number of minority neighbors used for constructing informative minority set
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class MWMOTE(data: Data, seed: Long = System.currentTimeMillis(), N: Int = 500, k1: Int = 5, k2: Int = 5, k3: Int = 5,
             dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the MWMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    def f(value: Double, cut: Double): Double = {
      if (value < cut) value else cut
    }

    def Cf(y: (Int, Int), x: Int, Nmin: Array[Array[Int]]): Double = {
      val cut: Double = 5 // values used in the paper
      val CMAX: Double = 2

      if (!Nmin(y._2).contains(x)) {
        val D: Double = if (dist == Distance.EUCLIDEAN) {
          euclidean(samples(y._1), samples(x))
        } else {
          HVDM(samples(y._1), samples(x), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
        }
        f(samples(0).length / D, cut) * CMAX
      } else
        0.0
    }

    def Iw(y: (Int, Int), x: Int, Nmin: Array[Array[Int]], Simin: Array[Int]): Double = {
      val cf = Cf(y, x, Nmin)
      val df = cf / Simin.map(Cf(y, _, Nmin)).sum
      cf + df
    }

    def clusterDistance(cluster1: Array[Int], cluster2: Array[Int]): Double = {
      val centroid1: Array[Double] = (cluster1 map samples).transpose.map(_.sum / cluster1.length)
      val centroid2: Array[Double] = (cluster2 map samples).transpose.map(_.sum / cluster2.length)

      if (dist == Distance.EUCLIDEAN) {
        euclidean(centroid1, centroid2)
      } else {
        HVDM(centroid1, centroid2, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
    }

    def minDistance(cluster: ArrayBuffer[ArrayBuffer[Int]]): (Int, Int, Double) = {
      var minDist: (Int, Int, Double) = (0, 0, 99999999)
      var i, j: Int = 0
      while (i < cluster.length) {
        j = 0
        while (j < cluster.length) {
          if (i != j) {
            val dist = clusterDistance(cluster(i).toArray, cluster(j).toArray)
            if (dist < minDist._3) minDist = (i, j, dist)
          }
          j += 1
        }
        i += 1
      }
      minDist
    }

    def cluster(Sminf: Array[Int]): Array[Array[Int]] = {
      val distances: Array[Array[Double]] = Array.fill(Sminf.length, Sminf.length)(9999999.0)
      var i, j: Int = 0
      while (i < Sminf.length) {
        j = 0
        while (j < Sminf.length) {
          if (i != j) {
            distances(i)(j) = if (dist == Distance.EUCLIDEAN) {
              euclidean(samples(Sminf(i)), samples(Sminf(j)))
            } else {
              HVDM(samples(Sminf(i)), samples(Sminf(j)), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
            }
          }
          j += 1
        }
        i += 1
      }

      val Cp: Double = 3 // used in paper
      val Th: Double = distances.map(_.min).sum / Sminf.length * Cp
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

    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))
    // compute majority class
    val majorityClassIndex: Array[Int] = samples.indices.par.diff(minorityClassIndex.toList).toArray

    // construct the filtered minority set
    val Sminf: Array[Int] = minorityClassIndex.par.map(index => {
      val neighbors = if (dist == Distance.EUCLIDEAN) {
        kNeighbors(samples, index, k1)
      } else {
        kNeighborsHVDM(samples, index, k1, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
      if (neighbors map data.y contains data.y(minorityClassIndex(0))) {
        Some(index)
      } else {
        None
      }
    }).filterNot(_.forall(_ == None)).map(_.get).toArray

    //for each sample in Sminf compute the nearest majority set
    val Sbmaj: Array[Int] = Sminf.par.flatMap { x =>
      if (dist == Distance.EUCLIDEAN) {
        kNeighbors(majorityClassIndex map samples, samples(x), k2)
      } else {
        kNeighborsHVDM(majorityClassIndex map samples, samples(x), k2, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
    }.distinct.par.map(majorityClassIndex(_)).toArray

    // for each majority example in Sbmaj , compute the nearest minority set
    val Nmin: Array[Array[Int]] = Sbmaj.par.map { x =>
      (if (dist == Distance.EUCLIDEAN) {
        kNeighbors(minorityClassIndex map samples, samples(x), k3)
      } else {
        kNeighborsHVDM(minorityClassIndex map samples, samples(x), k3, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }).par.map(minorityClassIndex(_)).toArray
    }.toArray

    // find the informative minority set (union of all Nmin)
    val Simin: Array[Int] = Nmin.par.flatten.distinct.toArray
    // for each sample in Simin compute the selection weight
    val Sw: Array[Double] = Simin.par.map(x => Sbmaj.zipWithIndex.par.map(y => Iw(y, x, Nmin, Simin)).sum).toArray
    val sumSw: Double = Sw.sum
    // convert each Sw into probability
    val Sp: Array[(Double, Int)] = Sw.par.map(_ / sumSw).toArray.zip(Simin).sortWith(_._1 > _._1)

    // compute the clusters
    val clusters: Array[Array[Int]] = cluster(minorityClassIndex) // cluster => index to processedData
    val clustersIndex: Map[Int, Int] = clusters.zipWithIndex.flatMap(c => {
      clusters(c._2).map(index => (index, c._2))
    }).toMap // index to processedData => cluster

    //output data
    val output: Array[Array[Double]] = Array.ofDim(N, samples(0).length)

    val probsSum: Double = Sp.map(_._1).sum
    val r: Random = new Random(seed)

    (0 until N).par.foreach(i => {
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

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(data.x.length + output.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
  }
}
