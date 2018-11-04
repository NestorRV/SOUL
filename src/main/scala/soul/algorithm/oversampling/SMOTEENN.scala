package soul.algorithm.oversampling

import soul.algorithm.undersampling.ENN
import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** SMOTEENN algorithm. Original paper: "A Study of the Behavior of Several Methods for Balancing Machine Learning
  * Training Data" by Gustavo E. A. P. A. Batista, Ronaldo C. Prati and Maria Carolina Monard.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   amount of Smote N%
  * @param k         number of minority class nearest neighbors
  * @param dist      object of DistanceType representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class SMOTEENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
               percent: Int = 500, k: Int = 5, dist: DistanceType = Distance(euclideanDistance),
               val normalize: Boolean = false, val verbose: Boolean = false) {

  /** Compute the SMOTEENN algorithm
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
      neighbors = dist match {
        case distance: Distance =>
          kNeighbors(minorityClassIndex map samples, i._2, k, distance)
        case _ =>
          kNeighborsHVDM(minorityClassIndex map samples, i._2, k, data.fileInfo.nominal, sds, attrCounter,
            attrClassesCounter).map(minorityClassIndex(_))
      }
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
    val enn = new ENN(ennData, dist = dist)
    val resultTL: Data = enn.compute()
    val finalIndex: Array[Int] = result.indices.diff(resultTL.index.get).toArray

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.nomToNum(0).isEmpty) {
      to2Decimals(zeroOneDenormalization(finalIndex map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs))
    } else {
      toNominal(zeroOneDenormalization(finalIndex map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs), data.nomToNum)
    }, finalIndex map resultClasses, None, data.fileInfo)
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(newData.x.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
