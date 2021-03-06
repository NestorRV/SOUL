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

/** One-Side Selection. Original paper: "Addressing the Curse of Imbalanced
  * Training Sets: One-Side Selection" by Miroslav Kubat and Stan Matwin.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of Distance enumeration representing the distance to be used
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @param verbose    choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class OSS(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN,
          normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the OSS algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    // Note: the notation used to refers the subsets of data is the used in the original paper.
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

    val positives: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val randomElement: Int = classesToWorkWith.indices.diff(positives)(new util.Random(seed).nextInt(classesToWorkWith.length - positives.length))
    val c: Array[Int] = positives ++ Array(randomElement)

    val KDTree: Option[KDTree] = if (dist == Distance.EUCLIDEAN) {
      Some(new KDTree(c map dataToWorkWith, c map classesToWorkWith, dataToWorkWith(0).length))
    } else {
      None
    }

    val labels: Seq[(Int, Any)] = if (dist == Distance.EUCLIDEAN) {
      dataToWorkWith.indices.map(i => (i, mode(KDTree.get.nNeighbours(dataToWorkWith(i), 1)._2.toArray)))
    } else {
      val neighbours = c map dataToWorkWith
      val classes = c map classesToWorkWith

      dataToWorkWith.indices.map(i => (i, nnRuleHVDM(neighbours, dataToWorkWith(i), c.indexOf(i), classes, 1, data.fileInfo.nominal,
        sds, attrCounter, attrClassesCounter, "nearest")._1))
    }
    val misclassified: Array[Int] = labels.collect { case (i, label) if label != classesToWorkWith(i) => i }.toArray
    val finalC: Array[Int] = (misclassified ++ c).distinct

    val auxData: Data = new Data(x = toXData(finalC map dataToWorkWith), y = finalC map classesToWorkWith, fileInfo = data.fileInfo)
    auxData.processedData = finalC map dataToWorkWith
    val tl = new TL(auxData, dist = dist, minorityClass = Some(untouchableClass))
    val resultTL: Data = tl.compute()
    val finalIndex: Array[Int] = (resultTL.index.get.toList map finalC).toArray
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}
