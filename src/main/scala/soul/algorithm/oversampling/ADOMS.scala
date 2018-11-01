package soul.algorithm.oversampling

import breeze.linalg.{DenseMatrix, eigSym}
import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** ADOMS algorithm. Original paper: "The Generation Mechanism of Synthetic Minority Class Examples" by Sheng TANG
  * and Si-ping CHEN.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   amount of samples N%
  * @param k         number of neighbors
  * @param distance  distance to use when calling the NNRule
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class ADOMS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
            percent: Int = 300, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN,
            val normalize: Boolean = false) extends LazyLogging {
  /** Compute the first principal component axis
    *
    * @param A the data
    * @return the first principal component axis
    */
  private def PCA(A: Array[Array[Double]]): Array[Double] = {
    val mean: Array[Double] = A.transpose.map(_.sum / A.length)
    // subtract the mean to the data
    val dataNoMean: DenseMatrix[Double] = DenseMatrix(A: _*) :- DenseMatrix(A.map(_ => mean): _*)
    // get the covariance matrix
    val oneDividedByN: Array[Array[Double]] = Array.fill(dataNoMean.cols, dataNoMean.cols)(dataNoMean.rows)
    val S: DenseMatrix[Double] = (dataNoMean.t * dataNoMean) :/ DenseMatrix(oneDividedByN: _*)
    //compute the eigenvectors and eigenvalues of S
    val eigen = eigSym(S)

    //return the first eigenvector because it represent the first principal component axis
    eigen.eigenvectors(0, ::).t.toArray
  }

  /** Compute the ADOMS algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))
    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(minorityClassIndex.length * percent / 100, samples(0).length)(0.0)
    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)
    var newIndex: Int = 0
    val r: Random = new Random(seed)

    val (attrCounter, attrClassesCounter, sds) = if (distance == Distances.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues((_: Array[Double]).length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    (0 until percent / 100).foreach(_ => {
      // for each minority class sample
      minorityClassIndex.zipWithIndex.foreach(i => {
        neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
        // calculate first principal component axis of local data distribution
        val l2: Array[Double] = PCA((neighbors map minorityClassIndex) map samples)
        val n: Int = r.nextInt(neighbors.length)
        val D: Double = computeDistance(samples(i._1), samples(minorityClassIndex(neighbors(n))), distance, data.fileInfo.nominal,
          sds, attrCounter, attrClassesCounter)
        // compute projection of n in l2, M is on l2
        val dotMN: Double = l2.indices.map(j => {
          samples(i._1)(j) - samples(minorityClassIndex(neighbors(n)))(j)
        }).toArray.zipWithIndex.map(j => {
          j._1 * l2(j._2)
        }).sum
        val dotMM: Double = l2.map(x => x * x).sum
        // create synthetic sample
        output(newIndex) = l2.indices.map(j => samples(i._1)(j) + dotMN / dotMM * l2(j) * D * r.nextFloat()).toArray
        newIndex = newIndex + 1
      })
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
