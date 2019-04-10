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
