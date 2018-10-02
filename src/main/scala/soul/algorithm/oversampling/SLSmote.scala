package soul.algorithm.oversampling

import soul.algorithm.Algorithm
import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** Safe-Level-SMOTE algorithm
  *
  * @author David López Pretel
  */
class SLSmote(private[soul] val data: Data) extends Algorithm {
  /** Compute the Smote algorithm
    *
    * @param file  file to store the log. If its set to None, log process would not be done
    * @param k     Number of nearest neighbors
    * @param dType the type of distance to use, hvdm or euclidean
    * @param seed  seed for the random
    * @return synthetic samples generated
    */
  def compute(file: Option[String] = None, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {
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

    // output with a size of |D|-t samples
    val output: Array[Array[Double]] = Array.fill(minorityClassIndex.length, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    var sl_ratio: Double = 0.0
    val r: Random.type = scala.util.Random
    r.setSeed(seed)
    // for each minority class sample
    minorityClassIndex.foreach(i => {
      // compute k neighbors from p and save number of positive instances
      neighbors = kNeighbors(samples, i, k, dType, data._nominal.length == 0, (samples, data._originalClasses))
      val n: Int = neighbors(r.nextInt(neighbors.length))
      val slp: Int = neighbors.map(neighbor => {
        if (data._originalClasses(neighbor) == data._minorityClass) {
          1
        } else {
          0
        }
      }).sum
      // compute k neighbors from n and save number of positive instances
      val sln: Int = kNeighbors(samples, n, k, dType, data._nominal.length == 0, (samples, data._originalClasses)).map(neighbor => {
        if (data._originalClasses(neighbor) == data._minorityClass) {
          1
        } else {
          0
        }
      }).sum
      if (sln != 0) { //sl is safe level
        sl_ratio = slp / sln
      } else {
        sl_ratio = 99999999
      }
      if (!(sl_ratio == 99999999 && slp == 0)) {
        // calculate synthetic sample
        var gap: Double = 0.0 // 2 case
        if (sl_ratio == 1) { // 3 case
          gap = r.nextFloat
        } else if (sl_ratio > 1 && sl_ratio != 99999999) { // 4 case
          gap = r.nextFloat * (1 / sl_ratio)
        } else if (sl_ratio < 1) { // 5 case
          gap = r.nextFloat()
          if (gap < 1 - sl_ratio) {
            gap = gap + 1 - sl_ratio
          }
        }
        samples(i).indices.foreach(atrib => {
          val diff: Double = samples(n)(atrib) - samples(i)(atrib)
          output(newIndex)(atrib) = samples(i)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      }
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

    this.logger.addMsg("ORIGINAL SIZE: %d".format(data._originalData.length))
    this.logger.addMsg("NEW DATA SIZE: %d".format(data._resultData.length))
    this.logger.addMsg("NEW SAMPLES ARE:")
    dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) this.logger.addMsg("%d".format(index._2)))
    // Save the time
    this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))

    // Save the log
    this.logger.storeFile(file.get + "_SLSmote")
  }
}
