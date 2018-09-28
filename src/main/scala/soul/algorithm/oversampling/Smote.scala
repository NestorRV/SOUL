package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** SMOTE algorithm
  *
  * @author David López Pretel
  */
class Smote(private val data: Data) {
  /** Compute the Smote algorithm
    *
    * @param percent Amount of Smote N%
    * @param k       Number of minority class nearest neighbors
    * @param dType   the type of distance to use, hvdm or euclidean
    * @param seed    seed for the random
    * @return synthetic samples generated
    */
  def compute(percent: Int = 500, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

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
    val r: Random.type = scala.util.Random
    r.setSeed(seed)
    // for each minority class sample
    minorityClassIndex.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, dType, data._nominal.length == 0, (samples, data._originalClasses)).map(minorityClassIndex(_))
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
