package soul.algorithm.oversampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities
import soul.util.Utilities._

import scala.util.Random

/** SafeLevel-SMOTE algorithm. Original paper: "Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling Technique
  * for Handling the Class Imbalanced Problem" by Chumphol Bunkhumpornpat, Krung Sinapiromsaran, and Chidchanok Lursinsap.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param k         Number of nearest neighbors
  * @param dist      distance to be used. It should be "HVDM" or a function of the type: (Array[Double], Array[Double]) => Double.
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class SafeLevelSMOTE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
                     k: Int = 5, dist: Any = Utilities.euclideanDistance _, val normalize: Boolean = false) extends LazyLogging {

  private[soul] val distance: Distances.Distance = getDistance(dist)

  /** Compute the SafeLevelSMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    val (attrCounter, attrClassesCounter, sds) = if (distance == Distances.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues((_: Array[Double]).length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

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
      neighbors = if (distance == Distances.USER) {
        kNeighbors(samples, i, k, dist)
      } else {
        kNeighborsHVDM(samples, i, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
      val n: Int = neighbors(r.nextInt(neighbors.length))
      val slp: Int = neighbors.map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
          1
        } else {
          0
        }
      }).sum
      // compute k neighbors from n and save number of positive instances
      val selectedNeighbors: Array[Int] = if (distance == Distances.USER) {
        kNeighbors(samples, n, k, dist)
      } else {
        kNeighborsHVDM(samples, n, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }

      val sln: Int = selectedNeighbors.map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
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

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
    val finishTime: Long = System.nanoTime()

    logger.whenInfoEnabled {
      logger.info("ORIGINAL SIZE: %d".format(data.x.length))
      logger.info("NEW DATA SIZE: %d".format(newData.x.length))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
