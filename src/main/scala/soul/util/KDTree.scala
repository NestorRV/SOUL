package soul.util

import com.thesamet.spatial.{DimensionalOrdering, KDTreeMap, Metric}

import scala.language.implicitConversions
import scala.math.Numeric.Implicits._

/** Wrapper of a com.thesamet.spatial.KDTreeMap adapted for Arrays of Doubles
  *
  * @param x          data
  * @param y          labels
  * @param dimensions number of dimensions
  * @author Néstor Rodríguez Vico
  */
class KDTree(x: Array[Array[Double]], y: Array[Any], dimensions: Int) {

  private val kDTreeMap: KDTreeMap[Array[Double], (Any, Int)] =
    KDTreeMap.fromSeq((x zip y.zipWithIndex).map(f => f._1 -> (f._2._1, f._2._2)))(dimensionalOrderingForArray[Array[Double], Double](dimensions))

  def nNeighbours(instance: Array[Double], k: Int, leaveOneOut: Boolean): (Seq[Array[Double]], Seq[Any], Seq[Int]) = {
    val realK: Int = if (leaveOneOut) k + 1 else k
    val drop: Int = if (leaveOneOut) 1 else 0
    val instances: (Seq[Array[Double]], Seq[(Any, Int)]) = kDTreeMap.findNearest(instance, realK).drop(drop).unzip
    val (labels, index) = instances._2.unzip
    (instances._1, labels, index)
  }

  def apply(x: Array[Double]) = kDTreeMap(x)

  def dimensionalOrderingForArray[T <: Array[A], A](dim: Int)(implicit ord: Ordering[A]): DimensionalOrdering[T] =
    new DimensionalOrdering[T] {
      val dimensions: Int = dim

      def compareProjection(d: Int)(x: T, y: T): Int = ord.compare(x(d), y(d))
    }

  implicit def metricFromArray[A](implicit n: Numeric[A]): Metric[Array[A], A] = new Metric[Array[A], A] {
    override def distance(x: Array[A], y: Array[A]): A = x.zip(y).map { z =>
      val d = z._1 - z._2
      d * d
    }.sum

    override def planarDistance(dimension: Int)(x: Array[A], y: Array[A]): A = {
      val dd = x(dimension) - y(dimension)
      dd * dd
    }
  }
}