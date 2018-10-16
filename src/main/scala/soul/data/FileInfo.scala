package soul.data

import scala.collection.mutable

/** Data structure used by the arff classes
  *
  * @param _file             file containing the data
  * @param _comment          string indicating that a line is a comment
  * @param _columnClass      indicates which column represents the class in the file
  * @param _delimiter        string separating two elements
  * @param _missing          string indicating a element is missed
  * @param _header           header of the file. If it is _, there was no header
  * @param _attributes       map with the form: index -> attributeName
  * @param _attributesValues map with the form attributeName -> type (it it's nominal, possible values instead of type)
  * @param nominal           array to know which attributes are nominal
  * @author Néstor Rodríguez Vico
  */
class FileInfo private[soul](private[soul] val _file: String, private[soul] val _comment: String,
                             private[soul] val _columnClass: Int = -1,
                             private[soul] val _delimiter: String, private[soul] val _missing: String,
                             private[soul] val _header: Array[String], private[soul] val _relationName: String,
                             private[soul] val _attributes: mutable.Map[Int, String],
                             private[soul] val _attributesValues: mutable.Map[String, String],
                             private[soul] val nominal: Array[Int]) {

  // data necessary to denormalize the data
  private[soul] var maxAttribs: Array[Double] = _
  private[soul] var minAttribs: Array[Double] = _

}