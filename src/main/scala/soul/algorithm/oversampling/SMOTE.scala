package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SMOTE algorithm. Original paper: "SMOTE: Synthetic Minority Over-sampling Technique" by Nitesh V. Chawla, Kevin W.
  * Bowyer, Lawrence O. Hall and W. Philip Kegelmeyer.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param file      file to store the log. If its set to None, log process would not be done
  * @param percent   amount of SMOTE N%
  * @param k         number of minority class nearest neighbors
  * @param distance  distance to use when calling the NNRule
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class SMOTE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
            percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN, val normalize: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger

  /** Compute the SMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    val (attrCounter, attrClassesCounter, sds) = if (distance == Distances.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues((_: Array[Double]).length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
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
    val output: Array[Array[Double]] = Array.fill(N * T, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    val r: Random = new Random(seed)
    // for each minority class sample
    var i: Int = 0
    while (i < minorityClassIndex.length) {
      neighbors = kNeighbors(minorityClassIndex map samples, i, k, distance, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter).map(minorityClassIndex(_))
      // compute populate for the sample
      var j: Int = 0
      while (j < N) {
        val nn: Int = r.nextInt(neighbors.length)
        // compute attributes of the sample
        var atrib: Int = 0
        while (atrib < samples(0).length) {
          val diff: Double = samples(neighbors(nn))(atrib) - samples(minorityClassIndex(i))(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(minorityClassIndex(i))(atrib) + gap * diff
          atrib = atrib + 1
        }
        newIndex = newIndex + 1
        j = j + 1
      }
      i = i + 1
    }

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      logger.addMsg("NEW DATA SIZE: %d".format(newData.x.length))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}
