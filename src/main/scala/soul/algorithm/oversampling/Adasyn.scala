package soul.algorithm.oversampling

import soul.algorithm.Algorithm
import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** Adasyn algorithm
  *
  * @author David López Pretel
  */
class Adasyn(private[soul] val data: Data) extends Algorithm {
  /** Compute the Smote algorithm
    *
    * @param file  file to store the log. If its set to None, log process would not be done
    * @param d     preset threshold for the maximum tolerated degree of class imbalance radio
    * @param B     balance level after generation of synthetic data
    * @param k     number of neighbors
    * @param dType the type of distance to use, hvdm or euclidean
    * @param seed  seed for the random
    * @return synthetic samples generated
    */
  def compute(file: Option[String] = None, d: Double = 1, B: Double = 1, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {
    if (B > 1 || B < 0) {
      throw new Exception("B must be between 0 and 1, both included")
    } else if (d > 1 || d <= 0) {
      throw new Exception("d must be between 0 and 1, zero not included")
    }

    if (dType != Distances.EUCLIDEAN && dType != Distances.HVDM) {
      throw new Exception("The distance must be euclidean or hvdm")
    }

    // Start the time
    val initTime: Long = System.nanoTime()

    var samples: Array[Array[Double]] = data._processedData
    if (dType == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data._originalClasses)
    data._minorityClass = data._originalClasses(minorityClassIndex(0))

    // calculate size of the output
    val ms: Int = minorityClassIndex.length
    val ml: Int = data._originalClasses.length - ms
    val G: Int = ((ml - ms) * B).asInstanceOf[Int]
    // k neighbors of each minority sample
    val neighbors: Array[Array[Int]] = minorityClassIndex.indices.map(sample => {
      kNeighbors(samples, minorityClassIndex(sample), k, dType, data._nominal.length == 0, (samples, data._originalClasses))
    }).toArray

    // ratio of each minority sample
    var ratio: Array[Double] = neighbors.map(neighborsOfX => {
      neighborsOfX.map(neighbor => {
        if (data._originalClasses(neighbor) != data._minorityClass) {
          1
        } else {
          0
        }
      }).sum.asInstanceOf[Double] / k
    })

    // normalize ratios
    ratio = ratio.map(_ / ratio.sum)
    // number of synthetic samples for each sample
    val g: Array[Int] = ratio.map(ri => (ri * G).asInstanceOf[Int])
    // output with a size of sum(Gi) samples
    val output: Array[Array[Double]] = Array.fill(g.sum, samples(0).length)(0.0)
    var newIndex: Int = 0
    val r: Random.type = scala.util.Random
    r.setSeed(seed)
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
      this.logger.storeFile(file.get + "_Adasyn")
    }
  }
}
