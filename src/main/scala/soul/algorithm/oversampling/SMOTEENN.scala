package soul.algorithm.oversampling

import soul.algorithm.undersampling.ENN
import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SMOTEENN algorithm. Original paper: "A Study of the Behavior of Several Methods for Balancing Machine Learning
  * Training Data" by Gustavo E. A. P. A. Batista, Ronaldo C. Prati and Maria Carolina Monard.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param file      file to store the log. If its set to None, log process would not be done
  * @param percent   amount of Smote N%
  * @param k         number of minority class nearest neighbors
  * @param distance  distance to use when calling the NNRule
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class SMOTEENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
               percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN, val normalize: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger

  /** Compute the SMOTEENN algorithm
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
    minorityClassIndex.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter).map(minorityClassIndex(_))
      // compute populate for the sample
      (0 until N).foreach(_ => {
        val nn: Int = r.nextInt(neighbors.length)
        // compute attributes of the sample
        samples(i._1).indices.foreach(atrib => {
          val diff: Double = samples(neighbors(nn))(atrib) - samples(i._1)(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(i._1)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      })
    })

    val result: Array[Array[Double]] = Array.concat(samples, output)
    val resultClasses: Array[Any] = Array.concat(data.y, Array.fill(output.length)(minorityClass))

    val ennData: Data = new Data(x = toXData(result), y = resultClasses, fileInfo = data.fileInfo)
    ennData.processedData = result
    val enn = new ENN(ennData, file = None, distance = distance)
    val resultTL: Data = enn.compute()
    val finalIndex: Array[Int] = result.indices.diff(resultTL.index.get).toArray

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.nomToNum(0).isEmpty) {
      to2Decimals(zeroOneDenormalization(finalIndex map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs))
    } else {
      toNominal(zeroOneDenormalization(finalIndex map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs), data.nomToNum)
    }, finalIndex map resultClasses, None, data.fileInfo)
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
