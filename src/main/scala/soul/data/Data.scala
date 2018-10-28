package soul.data

import scala.collection.mutable

/** Data structure used by the algorithms
  *
  * @param x        data associated to the file (x)
  * @param y        classes associated to the file (y)
  * @param index    randomIndex representing the kept elements
  * @param fileInfo object with the information needed to save the data into a file
  * @author Néstor Rodríguez Vico
  */
class Data private[soul](private[soul] val x: Array[Array[Any]], private[soul] val y: Array[Any],
                         private[soul] val index: Option[Array[Int]] = None, private[soul] val fileInfo: FileInfo) {

  private[soul] var processedData: Array[Array[Double]] = new Array[Array[Double]](0)
  private[soul] var nomToNum: Array[mutable.Map[Double, Any]] = new Array[mutable.Map[Double, Any]](0)
}
