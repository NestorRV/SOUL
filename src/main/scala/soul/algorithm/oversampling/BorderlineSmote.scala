package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities._

import scala.util.Random

/** Borderline-SMOTE algorithm
  *
  * @author David López Pretel
  */
class BorderlineSmote(private val data: Data) {
  /** Compute the Smote algorithm
    *
    * @param m     Number of nearest neighbors
    * @param k     Number of minority class nearest neighbors
    * @param dType the type of distance to use, hvdm or euclidean
    * @param seed  seed for the random
    * @return synthetic samples generated
    */
  def compute(m: Int = 10, k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN, seed: Long = 5): Unit = {
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

    // compute minority class neighbors
    val minorityClassNeighbors: Array[Array[Int]] = minorityClassIndex.map(node => kNeighbors(samples, node, m, dType, data._nominal.length == 0, (samples, data._originalClasses)))

    //compute nodes in borderline
    val DangerNodes: Array[Int] = minorityClassNeighbors.map(neighbors => {
      var counter = 0
      neighbors.foreach(neighbor => {
        if (data._originalClasses(neighbor) != data._minorityClass) {
          counter += 1
        } else {
          counter
        }
      })
      counter
    }).zipWithIndex.map(nNonMinorityClass => {
      if (m / 2 <= nNonMinorityClass._1 && nNonMinorityClass._1 < m) {
        Some(nNonMinorityClass._2)
      } else {
        None
      }
    }).filterNot(_.forall(_ == None)).map(x => minorityClassIndex(x.get))

    val r: Random.type = scala.util.Random
    r.setSeed(seed)
    val s: Int = r.nextInt(k) + 1

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(s * DangerNodes.length, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    // for each minority class sample
    DangerNodes.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, dType, data._nominal.length == 0, (minorityClassIndex map samples, minorityClassIndex map data._originalClasses)).map(minorityClassIndex(_))
      val sNeighbors: Array[Int] = (0 until s).map(_ => r.nextInt(neighbors.length)).toArray.distinct
      neighbors = sNeighbors map neighbors
      // calculate populate for the sample
      neighbors.foreach(j => {
        // calculate attributes of the sample
        samples(i._1).indices.foreach(atrib => {
          val diff: Double = samples(j)(atrib) - samples(i._1)(atrib)
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
