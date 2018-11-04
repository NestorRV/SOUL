package soul.algorithm.oversampling

import soul.data.Data
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
  * @param dist      object of DistanceType representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class ADASYN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
             d: Double = 1, B: Double = 1, k: Int = 5, dist: DistanceType = Distance(euclideanDistance),
             val normalize: Boolean = false, val verbose: Boolean = false) {

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

    val (attrCounter, attrClassesCounter, sds) = if (dist.isInstanceOf[HVDM]) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    // calculate size of the output
    val ms: Int = minorityClassIndex.length
    val ml: Int = data.y.length - ms
    val G: Int = ((ml - ms) * B).asInstanceOf[Int]
    // k neighbors of each minority sample
    val neighbors: Array[Array[Int]] = minorityClassIndex.indices.map { sample =>
      dist match {
        case distance: Distance =>
          kNeighbors(samples, minorityClassIndex(sample), k, distance)
        case _ =>
          kNeighborsHVDM(samples, minorityClassIndex(sample), k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
    }.toArray

    // ratio of each minority sample
    var ratio: Array[Double] = neighbors.map(neighborsOfX => {
      neighborsOfX.map(neighbor => {
        if (data.y(neighbor) != minorityClass) 1 else 0
      }).sum.asInstanceOf[Double] / k
    })

    // normalize ratios
    ratio = ratio.map(_ / ratio.sum)
    // number of synthetic samples for each sample
    val g: Array[Int] = ratio.map(ri => (ri * G).asInstanceOf[Int])
    // output with a size of sum(Gi) samples
    val output: Array[Array[Double]] = Array.fill(g.sum, samples(0).length)(0.0)
    var newIndex: Int = 0
    val r: Random = new Random(seed)
    // for each minority class sample, create gi synthetic samples
    minorityClassIndex.zipWithIndex.foreach(xi => {
      (0 until g(xi._2)).foreach(_ => {
        // compute synthetic sample si = (xzi - xi) * lambda + xi
        val xzi = r.nextInt(neighbors(xi._2).length)

        samples(0).indices.foreach(atrib => {
          val diff: Double = samples(neighbors(xi._2)(xzi))(atrib) - samples(xi._1)(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(xi._1)(atrib) + gap * diff
        })
        newIndex += 1
      })
    })

    // check if the data is nominal or numerical
    val newData = new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(newData.x.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
