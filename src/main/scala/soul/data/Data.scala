package soul.data

import soul.util.Utilities.processData

/** Data structure used by the algorithms
  *
  * @param _nominal         array to know which attributes are nominal
  * @param _originalData    data associated to the file (x)
  * @param _originalClasses classes associated to the file (y)
  * @param _fileInfo        object with the information needed to save the data into a file
  * @author Néstor Rodríguez Vico
  */
class Data private[soul](private[soul] val _nominal: Array[Int], private[soul] val _originalData: Array[Array[Any]],
                         private[soul] val _originalClasses: Array[Any], private[soul] val _fileInfo: FileInfo) {

  // data without NA values and with nominal values transformed to numeric values
  private[soul] val (_processedData, _nomToNum) = processData(this)
  // data obtained after applying an core
  private[soul] var _resultData: Array[Array[Any]] = _
  // classes obtained after applying an core
  private[soul] var _resultClasses: Array[Any] = _
  // index representing the kept elements
  private[soul] var _index: Array[Int] = _
  // class obtained after applying an algorithm
  private[soul] var _minorityClass: Any = _
  // data necessary to denormalize the data
  private[soul] var _maxAttribs: Array[Double] = _
  private[soul] var _minAttribs: Array[Double] = _


  /** originalData getter
    *
    * @return read data
    */
  def originalData: Array[Array[Any]] = _originalData

  /** originalClasses getter
    *
    * @return read classes
    */
  def originalClasses: Array[Any] = _originalClasses

  /** resultData getter
    *
    * @return read data
    */
  def resultData: Array[Array[Any]] = _resultData

  /** resultClasses getter
    *
    * @return read classes
    */
  def resultClasses: Array[Any] = _resultClasses

  /** _minorityClass getter
    *
    * @return _minorityClass class
    */
  def minorityClass: Any = _minorityClass

  /** index of kept elements getter
    *
    * @return read classes
    */
  def index: Array[Int] = _index
}
