package soul.data

/** Data structure used by the algorithms
  *
  * @param x        data associated to the file (x)
  * @param y        classes associated to the file (y)
  * @param fileInfo object with the information needed to save the data into a file
  * @author Néstor Rodríguez Vico
  */
class Data private[soul](private[soul] val x: Array[Array[Any]], private[soul] val y: Array[Any],
                         private[soul] val fileInfo: FileInfo) {

  // data obtained after applying an core
  private[soul] var resultData: Array[Array[Any]] = _
  // classes obtained after applying an core
  private[soul] var resultClasses: Array[Any] = _
  // index representing the kept elements
  private[soul] var index: Array[Int] = _
}
