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
package soul.util

import com.thesamet.spatial.{DimensionalOrdering, KDTreeMap, Metric}

import scala.language.implicitConversions
import scala.math.sqrt

/** Wrapper of a com.thesamet.spatial.KDTreeMap adapted for Arrays of Doubles
  *
  * @param x          data
  * @param y          labels
  * @param dimensions number of dimensions
  * @param which      if it's set to "nearest", return the nearest neighbours, if it sets "farthest", return the farthest ones
  * @author Néstor Rodríguez Vico
  */
class KDTree(x: Array[Array[Double]], y: Array[Any], dimensions: Int, which: String = "nearest") {

  private[soul] var kDTreeMap: KDTreeMap[Array[Double], (Any, Int)] = if (which == "nearest") {
    KDTreeMap.fromSeq((x zip y.zipWithIndex).map(f => f._1 -> (f._2._1, f._2._2)))(dimensionalOrderingForArray[Array[Double], Double](dimensions))
  } else {
    KDTreeMap.fromSeq((x zip y.zipWithIndex).map(f => f._1 -> (f._2._1, f._2._2)))(dimensionalReverseOrderingForArray[Array[Double], Double](dimensions))
  }

  def nNeighbours(instance: Array[Double], k: Int, leaveOneOut: Boolean = true): (Seq[Array[Double]], Seq[Any], Seq[Int]) = {
    val realK: Int = if (leaveOneOut) k + 1 else k
    val drop: Int = if (leaveOneOut) 1 else 0
    val instances: (Seq[Array[Double]], Seq[(Any, Int)]) = kDTreeMap.findNearest(instance, realK).drop(drop).unzip
    val (labels, index) = instances._2.unzip
    (instances._1, labels, index)
  }

  def apply(x: Array[Double]): (Any, Int) = kDTreeMap(x)

  def addElement(x: Array[Double], y: Any): Unit = {
    kDTreeMap = kDTreeMap + (x -> (y, kDTreeMap.size + 1))
  }

  def dimensionalOrderingForArray[T <: Array[A], A](dim: Int)(implicit ord: Ordering[A]): DimensionalOrdering[T] =
    new DimensionalOrdering[T] {
      val dimensions: Int = dim

      def compareProjection(d: Int)(x: T, y: T): Int = ord.compare(x(d), y(d))
    }

  def dimensionalReverseOrderingForArray[T <: Array[A], A](dim: Int)(implicit ord: Ordering[A]): DimensionalOrdering[T] =
    new DimensionalOrdering[T] {
      val dimensions: Int = dim

      def compareProjection(d: Int)(x: T, y: T): Int = ord.compare(y(d), x(d))
    }

  implicit def metricFromArray(implicit n: Numeric[Double]): Metric[Array[Double], Double] = new Metric[Array[Double], Double] {
    override def distance(x: Array[Double], y: Array[Double]): Double = sqrt(x.zip(y).map { z =>
      val d = z._1 - z._2
      d * d
    }.sum)

    override def planarDistance(dimension: Int)(x: Array[Double], y: Array[Double]): Double = {
      val dd = x(dimension) - y(dimension)
      dd * dd
    }
  }
}