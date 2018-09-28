package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** Adoms algorithm
  *
  * @author David López Pretel
  */
class Adoms(private val data: Data) {

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


  /** Compute the Smote algorithm
    *
    * @param percent Amount of samples N%
    * @param k       number of neighbors
    * @param dType   the type of distance to use, hvdm or euclidean
    * @param seed    seed for the random
    * @return synthetic samples generated
    */
  def compute(percent: Int = 300, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {

    if (dType != Distances.EUCLIDEAN && dType != Distances.HVDM) {
      throw new Exception("The distance must be euclidean or hvdm")
    }

    var samples: Array[Array[Double]] = data._processedData
    if (dType == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data._originalClasses)
    data._minorityClass = data._originalClasses(minorityClassIndex(0))

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(minorityClassIndex.length * percent / 100, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    val r: Random.type = scala.util.Random
    r.setSeed(seed)

    (0 until percent / 100).foreach(_ => {
      // for each minority class sample
      minorityClassIndex.zipWithIndex.foreach(i => {
        neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, dType, data._nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data._originalClasses))
        // calculate first principal component axis of local data distribution
        val l2: Array[Double] = PCA((neighbors map minorityClassIndex) map samples)
        val n: Int = r.nextInt(neighbors.length)
        val D: Double = computeDistanceOversampling(samples(i._1), samples(minorityClassIndex(neighbors(n))), dType, data._nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data._originalClasses))
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
    if (data._nominal.length == 0) {
      data._resultData = dataShuffled map to2Decimals(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output))
    } else {
      data._resultData = dataShuffled map toNominal(Array.concat(data._processedData, if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output, data._maxAttribs, data._minAttribs) else output), data._nomToNum)
    }
    data._resultClasses = dataShuffled map Array.concat(data._originalClasses, Array.fill(output.length)(data._minorityClass))
  }
}
