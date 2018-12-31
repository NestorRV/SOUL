package soul.algorithm.oversampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.Array._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** SMOTERSB algorithm. Original paper: "kNN Approach to Unbalanced Data Distribution: SMOTE-RSB: a hybrid preprocessing
  * approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory"
  * by Enislay Ramentol, Yailé Caballero, Rafael Bello and Francisco Herrera.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   amount of Smote N%
  * @param k         number of minority class nearest neighbors
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class SMOTERSB(data: Data, seed: Long = System.currentTimeMillis(), percent: Int = 500, k: Int = 5,
               dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the SMOTERSB algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

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

    // check if the percent is correct
    var T: Int = minorityClassIndex.length
    var N: Int = percent

    if (N < 100) {
      T = N / 100 * T
      N = 100
    }
    N = N / 100

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.ofDim[Double](N * T, samples(0).length)

    val r: Random = new Random(seed)

    // for each minority class sample
    minorityClassIndex.indices.par.foreach((i: Int) => {
      val neighbors: Array[Int] = if (dist == Distance.EUCLIDEAN) {
        KDTree.get.nNeighbours(samples(minorityClassIndex(i)), k)._3.toArray
      } else {
        kNeighborsHVDM(samples, minorityClassIndex(i), k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }

      // compute populate for the sample
      (0 until N).par.foreach((n: Int) => {
        val nn: Int = neighbors(r.nextInt(neighbors.length))
        // compute attributes of the sample
        samples(0).indices.foreach((atrib: Int) => {
          val diff: Double = samples(nn)(atrib) - samples(minorityClassIndex(i))(atrib)
          val gap: Double = r.nextFloat()
          output(i * N + n)(atrib) = samples(minorityClassIndex(i))(atrib) + gap * diff
        })
      })
    })

    //compute the majority class
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // minimum and maximum value for each attrib
    val maxMinValues: Array[(Double, Double)] = Array.concat(majorityClassIndex map samples, output).transpose.map(column => (column.max, column.min))

    //compute the similarity matrix
    val similarityMatrix: Array[Array[Double]] = Array.ofDim(output.length, majorityClassIndex.length)
    output.indices.par.foreach(i => {
      (majorityClassIndex map samples).zipWithIndex.par.foreach(j => {
        similarityMatrix(i)(j._2) = output(i).indices.map(k => {
          if (data.nomToNum(0).isEmpty) {
            1 - (Math.abs(output(i)(k) - j._1(k)) / (maxMinValues(k)._1 - maxMinValues(k)._2)) // this expression must be multiplied by wk
          } else { // but all the features are included, so wk is 1
            if (output(i)(k) == j._1(k)) 1 else 0
          }
        }).sum / output(i).length
      })
    })

    var result: ArrayBuffer[Int] = ArrayBuffer()
    var similarityValue: Double = 0.4
    var lowerApproximation: Boolean = true
    while (similarityValue < 0.9) {
      output.indices.foreach(i => {
        lowerApproximation = true
        majorityClassIndex.indices.foreach(j => {
          if (similarityMatrix(i)(j) > similarityValue)
            lowerApproximation = false
        })
        if (lowerApproximation) result += i
      })
      similarityValue += 0.05
    }

    //if there are not synthetic samples with lower approximation, return all synthetic samples
    if (result.isEmpty) {
      result = ArrayBuffer.range(0, output.length)
    } else {
      result = result.distinct
    }

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(data.x.length + output.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(result.toArray map output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else result.toArray map output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(result.toArray map output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else result.toArray map output), data.nomToNum)
    }, Array.concat(data.y, Array.fill((result.toArray map output).length)(minorityClass)), None, data.fileInfo)
  }
}
