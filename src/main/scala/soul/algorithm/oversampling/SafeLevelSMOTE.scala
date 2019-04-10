/*
SOUL: Scala Oversampling and Undersampling Library.
Copyright (C) 2019 Néstor Rodríguez, David López

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation in version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package soul.algorithm.oversampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.util.Random

/** SafeLevel-SMOTE algorithm. Original paper: "Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling Technique
  * for Handling the Class Imbalanced Problem" by Chumphol Bunkhumpornpat, Krung Sinapiromsaran, and Chidchanok Lursinsap.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param k         Number of nearest neighbors
  * @param dist      object of Distance enumeration representing the distance to be used
  * @param normalize normalize the data or not
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class SafeLevelSMOTE(data: Data, seed: Long = System.currentTimeMillis(), k: Int = 5,
                     dist: Distance = Distance.EUCLIDEAN, normalize: Boolean = false, verbose: Boolean = false) {

  /** Compute the SafeLevelSMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

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

    var sl_ratio: Double = 0.0
    val r: Random = new Random(seed)

    val output: Array[Array[Double]] = minorityClassIndex.indices.par.map(i => {
      // compute k neighbors from p and save number of positive instances
      val neighbors: Array[Int] = if (dist == Distance.EUCLIDEAN) {
        KDTree.get.nNeighbours(samples(minorityClassIndex(i)), k)._3.toArray
      } else {
        kNeighborsHVDM(samples, i, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
      val n: Int = neighbors(r.nextInt(neighbors.length))
      val slp: Int = neighbors.map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
          1
        } else {
          0
        }
      }).sum
      // compute k neighbors from n and save number of positive instances
      val selectedNeighbors: Array[Int] = if (dist == Distance.EUCLIDEAN) {
        KDTree.get.nNeighbours(samples(n), k)._3.toArray
      } else {
        kNeighborsHVDM(samples, n, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }

      val sln: Int = selectedNeighbors.map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
          1
        } else {
          0
        }
      }).sum
      if (sln != 0) { //sl is safe level
        sl_ratio = slp / sln
      } else {
        sl_ratio = 99999999
      }
      if (sl_ratio == 99999999 && slp == 0) {
        // dont create a synthetic instance
        None
      }
      else {
        // calculate synthetic sample
        Some(samples(i).indices.map(atrib => {
          var gap: Double = 0.0 // 2 case
          if (sl_ratio == 1) { // 3 case
            gap = r.nextFloat
          } else if (sl_ratio > 1 && sl_ratio != 99999999) { // 4 case
            gap = r.nextFloat * (1 / sl_ratio)
          } else if (sl_ratio < 1) { // 5 case
            gap = r.nextFloat()
            if (gap < 1 - sl_ratio) {
              gap = gap + 1 - sl_ratio
            }
          }
          val diff: Double = samples(n)(atrib) - samples(minorityClassIndex(i))(atrib)
          samples(minorityClassIndex(i))(atrib) + gap * diff
        }).toArray)
      }
    }).filterNot(_.forall(_ == None)).map(_.get).toArray

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
