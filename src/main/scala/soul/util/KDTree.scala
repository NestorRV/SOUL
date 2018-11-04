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
class KDTree(val x: Array[Array[Double]], val y: Array[Any], val dimensions: Int) {

  private val kDTreeMap: KDTreeMap[Array[Double], Any] =
    KDTreeMap.fromSeq((x zip y).map(f => f._1 -> f._2))(dimensionalOrderingForArray[Array[Double], Double](dimensions))

  def nNeighbours(instance: Array[Double], k: Int, leaveOneOut: Boolean): (Seq[Array[Double]], Seq[Any]) = {
    if (leaveOneOut) kDTreeMap.findNearest(instance, k + 1).drop(1).unzip else kDTreeMap.findNearest(instance, k).unzip
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