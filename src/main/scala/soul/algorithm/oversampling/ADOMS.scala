package soul.algorithm.oversampling

import breeze.linalg.{DenseMatrix, eigSym}
import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.util.Random

/** ADOMS algorithm. Original paper: "The Generation Mechanism of Synthetic Minority Class Examples" by Sheng TANG
  * and Si-ping CHEN.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   amount of samples N%
  * @param k         number of neighbors
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class ADOMS(data: Data, seed: Long = System.currentTimeMillis(), percent: Int = 300, k: Int = 5,
            dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the first principal component axis
    *
    * @param A the data
    * @return the first principal component axis
    */
  private def PCA(A: Array[Array[Double]]): Array[Double] = {
    val mean: Array[Double] = A.transpose.map(_.sum / A.length)
    // subtract the mean to the data
    val dataNoMean: DenseMatrix[Double] = DenseMatrix(A: _*) -:- DenseMatrix(A.map(_ => mean): _*)
    // get the covariance matrix
    val oneDividedByN: Array[Array[Double]] = Array.fill(dataNoMean.cols, dataNoMean.cols)(dataNoMean.rows)
    val S: DenseMatrix[Double] = (dataNoMean.t * dataNoMean) /:/ DenseMatrix(oneDividedByN: _*)
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
    val output: Array[Array[Double]] = Array.ofDim(minorityClassIndex.length * percent / 100, samples(0).length)
    // index array to save the neighbors of each sample
    val r: Random = new Random(seed)

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

    val N: Int = percent / 100

    (0 until N).par.foreach(nn => {
      // for each minority class sample
      minorityClassIndex.zipWithIndex.par.foreach(i => {
        val neighbors: Array[Int] = if (dist == Distance.EUCLIDEAN) {
          KDTree.get.nNeighbours(samples(i._1), k)._3.toArray
        } else {
          kNeighborsHVDM(samples, i._2, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
        }

        val n: Int = r.nextInt(neighbors.length)

        val D: Double = if (dist == Distance.EUCLIDEAN) {
          euclidean(samples(i._1), samples(neighbors(n)))
        } else {
          HVDM(samples(i._1), samples(neighbors(n)), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
        }

        // calculate first principal component axis of local data distribution
        val l2: Array[Double] = PCA(neighbors map samples)
        // compute projection of n in l2, M is on l2
        val dotMN: Double = l2.indices.map(j => {
          samples(i._1)(j) - samples(neighbors(n))(j)
        }).toArray.zipWithIndex.map(j => {
          j._1 * l2(j._2)
        }).sum
        val dotMM: Double = l2.map(x => x * x).sum
        // create synthetic sample
        output(nn * minorityClassIndex.length + i._2) = l2.indices.map(j => samples(i._1)(j) + dotMN / dotMM * l2(j)).toArray
        output(nn * minorityClassIndex.length + i._2) = output(nn * minorityClassIndex.length + i._2).indices.map(j => output(nn * minorityClassIndex.length + i._2)(j) + (samples(i._1)(j) - output(nn * minorityClassIndex.length + i._2)(j)) * r.nextFloat()).toArray
      })
    })

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(data.x.length + output.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output))
    } else {
      toNominal(Array.concat(data.processedData, if (normalize) zeroOneDenormalization(output, data.fileInfo.maxAttribs,
        data.fileInfo.minAttribs) else output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
  }
}
