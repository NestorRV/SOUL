package soul.algorithm.oversampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.util.Random

/** ADASYN algorithm. Original paper: "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning" by Haibo He,
  * Yang Bai, Edwardo A. Garcia, and Shutao Li.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param d         preset threshold for the maximum tolerated degree of class imbalance radio
  * @param B         balance level after generation of synthetic data
  * @param k         number of neighbors
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class ADASYN(data: Data, seed: Long = System.currentTimeMillis(), d: Double = 1, B: Double = 1, k: Int = 5,
             dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the ADASYN algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    if (B > 1 || B < 0) {
      throw new Exception("B must be between 0 and 1, both included")
    }

    if (d > 1 || d <= 0) {
      throw new Exception("d must be between 0 and 1, zero not included")
    }

    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val KDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
      Some(new KDTree(samples, data.y, samples(0).length))
    } else {
      None
    }

    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    // calculate size of the output
    val ms: Int = minorityClassIndex.length
    val ml: Int = data.y.length - ms
    val G: Int = ((ml - ms) * B).asInstanceOf[Int]

    // k neighbors of each minority sample
    val neighbors: Array[Array[Int]] = new Array[Array[Int]](minorityClassIndex.length)
    minorityClassIndex.indices.par.foreach { i =>
      if (dist == Distance.EUCLIDEAN) {
        neighbors(i) = kNeighbors(samples, minorityClassIndex(i), k)
        //neighbors(i) = KDTree.get.nNeighbours(samples(minorityClassIndex(i)), k)._3.toArray
      } else {
        neighbors(i) = kNeighborsHVDM(samples, minorityClassIndex(i), k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
    }

    // ratio of each minority sample
    var ratio: Array[Double] = new Array[Double](neighbors.length)
    neighbors.zipWithIndex.par.foreach(neighborsOfX => {
      ratio(neighborsOfX._2) = neighborsOfX._1.map(neighbor => {
        if (data.y(neighbor) != minorityClass) 1 else 0
      }).sum.asInstanceOf[Double] / k
    })

    // normalize ratios
    val sumRatios: Double = ratio.sum
    ratio.indices.par.foreach(i => ratio(i) = ratio(i) / sumRatios)

    // number of synthetic samples for each sample
    val g: Array[Int] = new Array[Int](ratio.length)
    ratio.zipWithIndex.par.foreach(ri => g(ri._2) = (ri._1 * G).asInstanceOf[Int])

    // output with a size of sum(Gi) samples
    val output: Array[Array[Double]] = Array.ofDim(g.sum, samples(0).length)

    val r: Random = new Random(seed)
    // must compute the random information before the loops due to parallelism

    var counter: Int = 0
    val increment: Array[Int] = new Array[Int](g.length)
    var i = 0
    while(i < g.length) {
      increment(i) = counter
      counter += g(i)
      i += 1
    }

    // for each minority class sample, create gi synthetic samples
    minorityClassIndex.indices.zip(increment).foreach(xi => {
      (0 until g(xi._1)).foreach(n => {
        // compute synthetic sample si = (xzi - xi) * lambda + xi
        samples(0).indices.foreach(atrib => {
          val nn: Int = neighbors(xi._1)(r.nextInt(neighbors(xi._1).length))
          val diff: Double = samples(nn)(atrib) - samples(minorityClassIndex(xi._1))(atrib)
          val gap: Float = r.nextFloat
          output(xi._2 + n)(atrib) = samples(minorityClassIndex(xi._1))(atrib) + gap * diff
        })
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
