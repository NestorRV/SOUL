package soul

import soul.algorithm.undersampling._
import soul.data.Data
import soul.io.Reader
import soul.util.KDTree

object main {
  def main(args: Array[String]): Unit = {
    val reader = new Reader
    val csvData: Data = reader.readDelimitedText(file = "109416.csv")

    def readCSV() : Array[Array[Double]] = {
      scala.io.Source.fromFile("/testData.csv").getLines().map(_.split(",").map(_.trim.toDouble)).toArray
    }

    /*val kDTree = new KDTree(Array(Array(1.0), Array(2.0)), Array("A", "B"), 2)
    val kDTreeMap = kDTree.getTree

    val result = kDTreeMap.findNearest(Array(1.0), 1)*/

    println()

    val times: Seq[Long] = (0 until 1).map { _ =>
      val alg = new NCL(csvData, seed = 0L)
      val start = System.currentTimeMillis
      alg.compute()
      val end = System.currentTimeMillis
      println((end - start) + " msec")
      end - start
    }

    /*
      25930 msec
      25096 msec
      24319 msec
      24669 msec
      24607 msec
      24924
     */
    println(times.sum / times.length)
  }
}
