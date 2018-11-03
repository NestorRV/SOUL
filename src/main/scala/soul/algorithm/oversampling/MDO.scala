package soul.algorithm.oversampling

import breeze.linalg.{DenseMatrix, DenseVector, eigSym, inv, sum}
import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** MDO algorithm. Original paper: "To combat multi-class imbalanced problems by means of over-sampling and boosting
  * techniques" by Lida Adbi and Sattar Hashemi.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class MDO(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), val normalize: Boolean = false) extends LazyLogging {

  /** create the new samples for MDO algorithm
    *
    * @param Ti    the samples changed of basis
    * @param mean  the mean of every characteristic
    * @param V     the vector of coefficients
    * @param Orate majoritySamples - minoritySamples
    * @param seed  seed to use. If it is not provided, it will use the system time
    * @return return the new samples generated
    */
  def MDO_oversampling(Ti: DenseMatrix[Double], mean: Array[Double], V: DenseVector[Double], Orate: Int, seed: Long): Array[Array[Double]] = {
    // check the number of new samples to be created
    var I: Int = Ti.rows
    var N: Int = Orate / I
    if (I > Orate) {
      N = 1
      I = Orate
    }

    val output: Array[Array[Double]] = Array.fill(Orate, Ti.cols)(0.0)
    var newIndex: Int = 0
    val rand: Random.type = scala.util.Random
    rand.setSeed(seed)

    (0 until I).foreach(i => {
      // square of each sample
      val x: DenseVector[Double] = Ti(i, ::).t :* Ti(i, ::).t
      // vector results from α × V , which forms the denominators of ellipse equation
      val alpha: Double = sum(x :/ V)
      val alphaV: DenseVector[Double] = V :* alpha
      (0 until N).foreach(_ => {
        var s: Double = 0.0
        (0 until Ti.cols - 1).foreach(p => {
          //random number between -sqrt(alphaV(p)) and sqrt(alphaV(p))
          val r: Double = -alphaV(p) / (Ti.cols - 1) + rand.nextFloat() * (alphaV(p) / (Ti.cols - 1) + alphaV(p) / (Ti.cols - 1))
          //this number is the value for the attrib p
          output(newIndex)(p) = r
          // sum necessary to compute the last attrib later
          s = s + (r * r / alphaV(p))
        })
        //compute the last attrib
        val lastFeaVal: Double = (1 - s) * alphaV(alphaV.size - 1) / (Ti.cols - 1)
        output(newIndex)(alphaV.size - 1) = if (rand.nextInt() % 2 == 0) -lastFeaVal else lastFeaVal
        newIndex += 1
      })
    })
    output
  }

  /** Compute the MDO algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    // compute majority class
    val minorityClass: Any = data.y(minorityClassIndex(0))
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // compute the mean for the values of each attribute
    val mean: Array[Double] = (minorityClassIndex map samples).transpose.map(_.sum / minorityClassIndex.length)

    // subtract the mean to every attrib and then compute the covariance matrix
    val Zi: DenseMatrix[Double] = DenseMatrix(minorityClassIndex map samples: _*) :- DenseMatrix(minorityClassIndex.map(_ => mean): _*)
    val oneDividedByN: Array[Array[Double]] = Array.fill(Zi.cols, Zi.cols)(Zi.rows)
    val S: DenseMatrix[Double] = (Zi.t * Zi) :/ DenseMatrix(oneDividedByN: _*)
    //compute the eigenvectors and eigenvalues of S
    val eigen = eigSym(S)

    // the eigenvectors form the columns of the matrix that performs the change os basis
    val Ti: DenseMatrix[Double] = (eigen.eigenvectors * Zi.t).t
    // the diag are the eigenvalues
    val V: DenseVector[Double] = eigen.eigenvalues

    //compute the new samples
    val newSamples: Array[Array[Double]] = MDO_oversampling(Ti, mean, V, majorityClassIndex.length - minorityClassIndex.length, seed)

    //transform the samples to the original basis
    val newSamplesToOriginalSpace: DenseMatrix[Double] = (inv(eigen.eigenvectors) * DenseMatrix(newSamples: _*).t).t

    //sum the mean again
    val samplesWithMean: DenseMatrix[Double] = newSamplesToOriginalSpace :+ DenseMatrix((0 until newSamplesToOriginalSpace.rows).map(_ => mean): _*)

    // the output
    val output: Array[Array[Double]] = Array.range(0, samplesWithMean.rows).map(i => samplesWithMean(i, ::).t.toArray)

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
