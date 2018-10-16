package soul.algorithm.oversampling

import breeze.linalg.{DenseMatrix, eigSym}
import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** ADOMS algorithm. Original paper: "The Generation Mechanism of Synthetic Minority Class Examples" by Sheng TANG
  * and Si-ping CHEN.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param percent  amount of samples N%
  * @param k        number of neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class ADOMS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
            percent: Int = 300, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, nomToNum) = processData(data)

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
    var samples: Array[Array[Double]] = processedData
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data, processedData)
    }

    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))
    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(minorityClassIndex.length * percent / 100, samples(0).length)(0.0)
    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)
    var newIndex: Int = 0
    val r: Random = new Random(seed)

    (0 until percent / 100).foreach(_ => {
      // for each minority class sample
      minorityClassIndex.zipWithIndex.foreach(i => {
        neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data.fileInfo.nominal.length == 0,
          (minorityClassIndex map samples, minorityClassIndex map data.y))
        // calculate first principal component axis of local data distribution
        val l2: Array[Double] = PCA((neighbors map minorityClassIndex) map samples)
        val n: Int = r.nextInt(neighbors.length)
        val D: Double = computeDistanceOversampling(samples(i._1), samples(minorityClassIndex(neighbors(n))), distance,
          data.fileInfo.nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data.y))
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

    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + output.length).indices.toList).toArray
    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      dataShuffled map to2Decimals(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      dataShuffled map toNominal(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), nomToNum)
    }, dataShuffled map Array.concat(data.y, Array.fill(output.length)(minorityClass)),
      Some(dataShuffled.zipWithIndex.collect { case (c, i) if c >= samples.length => i }), data.fileInfo)
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      logger.addMsg("NEW DATA SIZE: %d".format(newData.x.length))
      logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) logger.addMsg("%d".format(index._2)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}
