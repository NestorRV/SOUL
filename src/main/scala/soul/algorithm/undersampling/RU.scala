package soul.algorithm.undersampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

/** Compute a random algorithm.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param replacement whether or not to sample randomly with replacement or not. false by default
  * @param normalize   normalize the data or not
  * @param randomData  iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class RU(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(),
         ratio: Double = 1.0, replacement: Boolean = false, val normalize: Boolean = false,
         val randomData: Boolean = false) extends LazyLogging {

  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data

  /** Compute the RU algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val random: scala.util.Random = new scala.util.Random(seed)

    val minorityIndex: Array[Int] = data.y.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val majorityIndex: Array[Int] = random.shuffle(data.y.zipWithIndex.collect { case (label, i)
      if label != untouchableClass => i
    }.toList).toArray
    val selectedMajorityIndex: Array[Int] = if (!replacement) majorityIndex.take((minorityIndex.length * ratio).toInt) else
      majorityIndex.indices.map((_: Int) => random.nextInt(majorityIndex.length)).toArray map majorityIndex
    val finalIndex: Array[Int] = minorityIndex ++ selectedMajorityIndex
    val finishTime: Long = System.nanoTime()

    val newData: Data = new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)

    logger.whenInfoEnabled {
      val newCounter: Map[Any, Int] = (finalIndex map data.y).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.info("ORIGINAL SIZE: %d".format(data.x.length))
      logger.info("NEW DATA SIZE: %d".format(finalIndex.length))
      logger.info("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / data.x.length) * 100))
      logger.info("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.info("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}