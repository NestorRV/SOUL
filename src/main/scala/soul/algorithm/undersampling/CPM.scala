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
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.math.min

/** Class Purity Maximization. Original paper: "An Unsupervised Learning Approach to Resolving the
  * Data Imbalanced Issue in Supervised Learning Problems in Functional Genomics" by Kihoon Yoon and Stephen Kwek.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of Distance enumeration representing the distance to be used
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @param verbose    choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class CPM(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN,
          normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the CPM algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)
    val centers: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val classesToWorkWith: Array[Any] = if (randomData) {
      val randomIndex: List[Int] = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val posElements: Int = counter.head._2
    val negElements: Int = counter.tail.values.sum
    val impurity: Double = posElements.asInstanceOf[Double] / negElements.asInstanceOf[Double]
    val cluster: Array[Int] = new Array[Int](dataToWorkWith.length).indices.toArray

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    def purityMaximization(parentImpurity: Double, parentCluster: Array[Int], center: Int): Unit = {
      val cluster1: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val cluster2: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val posElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)
      val negElements: ArrayBuffer[Int] = new ArrayBuffer[Int](0)

      var center1: Int = 0
      var center2: Int = 0
      var pointer: Int = 0
      var impurity: Double = Double.PositiveInfinity
      var impurity1: Double = Double.PositiveInfinity
      var impurity2: Double = Double.PositiveInfinity

      parentCluster.foreach((f: Int) => if (data.y(f) == untouchableClass) posElements += f else negElements += f)

      val pairs: ArrayBuffer[(Int, Int)] = for {x <- negElements; y <- posElements} yield (x, y)

      while (parentImpurity <= impurity) {
        if (pointer >= pairs.length) {
          centers += center
          return
        }

        center1 = pairs(pointer)._1
        center2 = pairs(pointer)._2

        parentCluster.foreach { element: Int =>
          val d1: Double = if (dist == Distance.EUCLIDEAN) {
            euclidean(dataToWorkWith(element), dataToWorkWith(center1))
          } else {
            HVDM(dataToWorkWith(element), dataToWorkWith(center1), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }

          val d2: Double = if (dist == Distance.EUCLIDEAN) {
            euclidean(dataToWorkWith(element), dataToWorkWith(center2))
          } else {
            HVDM(dataToWorkWith(element), dataToWorkWith(center2), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          }

          if (d1 < d2)
            cluster1 += element else cluster2 += element
        }

        if (cluster1.nonEmpty)
          impurity1 = cluster1.count((element: Int) => data.y(element) == untouchableClass).toDouble / cluster1.length
        else {
          centers += center2
          return
        }

        if (cluster2.nonEmpty)
          impurity2 = cluster2.count((element: Int) => data.y(element) == untouchableClass).toDouble / cluster2.length
        else {
          centers += center1
          return
        }

        impurity = min(impurity1, impurity2)
        pointer += 1
      }

      purityMaximization(impurity1, cluster1.toArray, center1)
      purityMaximization(impurity2, cluster2.toArray, center2)
    }

    purityMaximization(impurity, cluster, 0)

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (centers.toArray map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(centers.toArray.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (centers.toArray.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(centers.toArray map data.x, centers.toArray map data.y, Some(centers.toArray), data.fileInfo)
  }
}
