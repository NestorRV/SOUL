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
package soul.algorithm.undersampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Edited Nearest Neighbour rule. Original paper: "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"
  * by Dennis L. Wilson.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of Distance enumeration representing the distance to be used
  * @param k          number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @param verbose    choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class ENN(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN,
          k: Int = 3, normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the ENN algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)
    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val classesToWorkWith: Array[Any] = if (randomData) {
      val randomIndex: List[Int] = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val finalIndex = new ArrayBuffer[Int]()
    val uniqueClasses = classesToWorkWith.distinct

    var j = 0
    val majorityClassIndex = new ArrayBuffer[Int]()
    while (j < classesToWorkWith.length) {
      if (classesToWorkWith(j) == untouchableClass) finalIndex += j else majorityClassIndex += j
      j += 1
    }

    val KDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
      Some(new KDTree(dataToWorkWith, classesToWorkWith, dataToWorkWith(0).length))
    } else {
      None
    }

    var i = 0
    while (i < uniqueClasses.length) {
      val targetClass = uniqueClasses(i)
      val selected: Array[(Int, Boolean)] = if (targetClass != untouchableClass) {
        majorityClassIndex.par.map { j =>
          val label = if (dist == Distance.EUCLIDEAN) {
            mode(KDTree.get.nNeighbours(dataToWorkWith(j), k)._2.toArray)
          } else {
            nnRuleHVDM(dataToWorkWith, dataToWorkWith(j), j, classesToWorkWith, k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")._1
          }

          (j, label == targetClass)
        }.toArray
      } else {
        new Array[(Int, Boolean)](0)
      }

      selected.foreach(e => if (e._2) finalIndex += e._1)

      i += 1
    }

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex.toArray map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex.toArray map data.x, finalIndex.toArray map data.y, Some(finalIndex.toArray), data.fileInfo)
  }
}
