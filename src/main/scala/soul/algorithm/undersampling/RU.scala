package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** Compute a random algorithm.
  *
  * @param data        data to work with
  * @param seed        seed to use. If it is not provided, it will use the system time
  * @param file        file to store the log. If its set to None, log process would not be done
  * @param ratio       ratio to know how many majority class examples to preserve. By default it's set to 1 so there
  *                    will be the same minority class examples as majority class examples. It will take
  *                    numMinorityInstances * ratio
  * @param replacement whether or not to sample randomly with replacement or not. false by default
  * @author Néstor Rodríguez Vico
  */
class RU(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         ratio: Double = 1.0, replacement: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it
  private[soul] val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val randomIndex: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, _) = processData(data)
  // Use randomized data
  val dataToWorkWith: Array[Array[Double]] = (randomIndex map processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (randomIndex map data.y).toArray

  /** Compute the RU algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val minorityIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == untouchableClass => i }
    val random: Random = new Random(seed)
    val majorityIndex: Array[Int] = random.shuffle(classesToWorkWith.zipWithIndex.collect { case (label, i)
      if label != untouchableClass => i
    }.toList).toArray
    val selectedMajorityIndex: Array[Int] = if (!replacement) majorityIndex.take((minorityIndex.length * ratio).toInt) else
      majorityIndex.indices.map((_: Int) => random.nextInt(majorityIndex.length)).toArray map majorityIndex
    val finalIndex: Array[Int] = minorityIndex ++ selectedMajorityIndex
    val finishTime: Long = System.nanoTime()

    val index: Array[Int] = (finalIndex map randomIndex).sorted
    val newData: Data = new Data(index map data.x, index map data.y, Some(index), data.fileInfo)

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}