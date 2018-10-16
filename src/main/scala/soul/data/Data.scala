package soul.data

import soul.util.Utilities.processData

/** Data structure used by the algorithms
  *
  * @param originalData    data associated to the file (x)
  * @param originalClasses classes associated to the file (y)
  * @param fileInfo        object with the information needed to save the data into a file
  * @author Néstor Rodríguez Vico
  */
class Data private[soul](private[soul] val originalData: Array[Array[Any]],
                         private[soul] val originalClasses: Array[Any], private[soul] val fileInfo: FileInfo) {

  // data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, nomToNum) = processData(this)
  // data obtained after applying an core
  private[soul] var resultData: Array[Array[Any]] = _
  // classes obtained after applying an core
  private[soul] var resultClasses: Array[Any] = _
  // index representing the kept elements
  private[soul] var index: Array[Int] = _
}
