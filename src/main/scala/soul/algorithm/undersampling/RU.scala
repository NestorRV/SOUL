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
package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities._

/** Compute a random algorithm. Original paper: "A study of the behavior of several methods for balancing machine
  * learning training data" by Batista, Gustavo EAPA and Prati, Ronaldo C and Monard, Maria Carolina.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param replacement whether or not to sample randomly with replacement or not. false by default
  * @param verbose     choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class RU(data: Data, seed: Long = System.currentTimeMillis(), ratio: Double = 1.0, replacement: Boolean = false, verbose: Boolean = false) {

  /** Compute the RU algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)

    val minorityIndex: Array[Int] = data.y.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val majorityIndex: Array[Int] = random.shuffle(data.y.zipWithIndex.collect { case (label, i)
      if label != untouchableClass => i
    }.toList).toArray
    val selectedMajorityIndex: Array[Int] = if (!replacement) majorityIndex.take((minorityIndex.length * ratio).toInt) else
      majorityIndex.indices.map(_ => random.nextInt(majorityIndex.length)).toArray map majorityIndex
    val finalIndex: Array[Int] = minorityIndex ++ selectedMajorityIndex
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map data.y).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / data.x.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}