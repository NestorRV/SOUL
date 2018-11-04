package soul.algorithm.oversampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

import scala.Array._
import scala.util.Random

/** SMOTERSB algorithm. Original paper: "kNN Approach to Unbalanced Data Distribution: SMOTE-RSB: a hybrid preprocessing
  * approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory"
  * by Enislay Ramentol, Yailé Caballero, Rafael Bello and Francisco Herrera.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   amount of Smote N%
  * @param k         number of minority class nearest neighbors
  * @param dist      object of DistanceType representing the distance to be used
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class SMOTERSB(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
               percent: Int = 500, k: Int = 5, dist: DistanceType = Distance(euclideanDistance),
               val normalize: Boolean = false) extends LazyLogging {

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

    val (attrCounter, attrClassesCounter, sds) = if (dist.isInstanceOf[HVDM]) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
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
      neighbors = (dist match {
        case distance: Distance =>
          kNeighbors(minorityClassIndex map samples, i._2, k, distance)
        case _ =>
          kNeighborsHVDM(minorityClassIndex map samples, i._2, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }).map(minorityClassIndex(_))
      // calculate populate for the sample
      (0 until N).foreach(_ => {
        val nn: Int = r.nextInt(neighbors.length)
        // calculate attributes of the sample
        samples(i._1).indices.foreach(atrib => {
          val diff: Double = samples(neighbors(nn))(atrib) - samples(i._1)(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(i._1)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      })
    })

    //compute the majority class
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // minimum and maximum value for each attrib
    val Amax: Array[Double] = Array.concat(majorityClassIndex map samples, output).transpose.map(column => column.max)
    val Amin: Array[Double] = Array.concat(majorityClassIndex map samples, output).transpose.map(column => column.min)

    //compute the similarity matrix
    val similarityMatrix: Array[Array[Double]] = output.map(i => {
      (majorityClassIndex map samples).map(j => {
        i.indices.map(k => {
          if (data.nomToNum(0).isEmpty) {
            1 - (Math.abs(i(k) - j(k)) / (Amax(k) - Amin(k))) // this expression must be multiplied by wk
          } else { // but all the features are included, so wk is 1
            if (i(k) == j(k)) 1 else 0
          }
        }).sum / i.length
      })
    })

    var result: Array[Int] = Array()
    var similarityValue: Double = 0.4
    var lowerApproximation: Boolean = false

    while (similarityValue < 0.9) {
      output.indices.foreach(i => {
        lowerApproximation = false
        majorityClassIndex.indices.foreach(j => {
          if (similarityMatrix(i)(j) > similarityValue)
            lowerApproximation = true
        })
        if (!lowerApproximation) result = result :+ i
      })
      similarityValue += 0.05
    }

    //if there are not synthetic samples with lower approximation, return all synthetic samples
    if (result.length == 0) {
      result = Array.range(0, output.length)
    }

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(result map output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else result map output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(result map output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else result map output), data.nomToNum)
    }, Array.concat(data.y, Array.fill((result map output).length)(minorityClass)), None, data.fileInfo)
    val finishTime: Long = System.nanoTime()

    logger.whenInfoEnabled {
      logger.info("ORIGINAL SIZE: %d".format(data.x.length))
      logger.info("NEW DATA SIZE: %d".format(newData.x.length))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    data
  }
}
