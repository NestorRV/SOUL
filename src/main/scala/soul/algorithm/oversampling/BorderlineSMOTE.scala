package soul.algorithm.oversampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.util.Random

/** Borderline-SMOTE algorithm. Original paper: "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets
  * Learning." by Hui Han, Wen-Yuan Wang, and Bing-Huan Mao.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param m         number of nearest neighbors
  * @param k         number of minority class nearest neighbors
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class BorderlineSMOTE(data: Data, seed: Long = System.currentTimeMillis(), m: Int = 10, k: Int = 5,
                      dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the BorderlineSMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    // compute minority class neighbors
    val minorityClassNeighbors: Array[Array[Int]] = if (dist == Distance.EUCLIDEAN) {
      minorityClassIndex.map(node => kNeighbors(samples, node, m))
    } else {
      minorityClassIndex.map(node => kNeighborsHVDM(samples, node, m, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter))
    }

    //compute nodes in borderline
    val DangerNodes: Array[Int] = minorityClassNeighbors.map(neighbors => {
      var counter = 0
      neighbors.foreach(neighbor => {
        if (data.y(neighbor) != minorityClass) {
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

    val r: Random = new Random(seed)
    val s: Int = r.nextInt(k) + 1

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(s * DangerNodes.length, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    // for each minority class sample
    DangerNodes.zipWithIndex.foreach(i => {
      neighbors = if (dist == Distance.EUCLIDEAN) {
        kNeighbors(minorityClassIndex map samples, i._2, k)
      } else {
        kNeighborsHVDM(minorityClassIndex map samples, i._2, k, data.fileInfo.nominal, sds, attrCounter,
          attrClassesCounter).map(minorityClassIndex(_))
      }
      val sNeighbors: Array[Int] = (0 until s).map(_ => r.nextInt(neighbors.length)).toArray.distinct
      neighbors = sNeighbors map neighbors
      // calculate populate for the sample
      neighbors.foreach(j => {
        // calculate attributes of the sample
        samples(i._1).indices.foreach(attrib => {
          val diff: Double = samples(j)(attrib) - samples(i._1)(attrib)
          val gap: Float = r.nextFloat
          output(newIndex)(attrib) = samples(i._1)(attrib) + gap * diff
        })
        newIndex = newIndex + 1
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
