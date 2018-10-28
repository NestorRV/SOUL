package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** Tomek Link core. Original paper: "Two Modifications of CNN" by Ivan Tomek.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param file       file to store the log. If its set to None, log process would not be done
  * @param distance   distance to use when calling the NNRule
  * @param ratio      indicates the instances of the Tomek Links that are going to be remove. "all" will remove all instances,
  *                   "minority" will remove instances of the minority class and "not minority" will remove all the instances
  *                   except the ones of the minority class.
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @author Néstor Rodríguez Vico
  */
class TL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         distance: Distances.Distance = Distances.EUCLIDEAN, ratio: String = "not minority",
         val normalize: Boolean = false, val randomData: Boolean = false) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues((_: Array[Any]).length)
  private[this] var untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1

  /** untouchableClass setter
    *
    * @param value new untouchableClass
    */
  private[soul] def untouchableClass_=(value: Any): Unit = {
    untouchableClass = value
  }

  /** Compute the TL algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val random: scala.util.Random = new scala.util.Random(seed)

    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    var randomIndex: List[Int] = data.x.indices.toList
    val classesToWorkWith: Array[Any] = if (randomData) {
      // Index to shuffle (randomize) the data
      randomIndex = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val candidates: Map[Any, Array[Int]] = classesToWorkWith.distinct.map {
      c: Any =>
        c -> classesToWorkWith.zipWithIndex.collect {
          case (a, b) if a != c => b
        }
    }.toMap

    val distances: Array[Array[Double]] = Array.fill[Array[Double]](dataToWorkWith.length)(new Array[Double](dataToWorkWith.length))

    dataToWorkWith.indices.par.foreach { i: Int =>
      dataToWorkWith.indices.drop(i).par.foreach { j: Int =>
        distances(i)(j) = euclideanDistance(dataToWorkWith(i), dataToWorkWith(j))
        distances(j)(i) = distances(i)(j)
      }
    }

    // Look for the nearest neighbour in the rest of the classes
    val nearestNeighbour: Array[Int] = distances.zipWithIndex.map((row: (Array[Double], Int)) => row._1.indexOf((candidates(classesToWorkWith(row._2)) map row._1).min))
    // For each instance, I: If my nearest neighbour is J and the nearest neighbour of J it's me, I, I and J form a Tomek link
    val tomekLinks: Array[(Int, Int)] = nearestNeighbour.zipWithIndex.filter((pair: (Int, Int)) => nearestNeighbour(pair._1) == pair._2)
    val targetInstances: Array[Int] = tomekLinks.flatMap((x: (Int, Int)) => List(x._1, x._2)).distinct
    // but the user can choose which of them should be removed
    val removedInstances: Array[Int] = if (ratio == "all") targetInstances else if (ratio == "minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a == untouchableClass => b
      } else if (ratio == "not minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a != untouchableClass => b
      } else
      throw new Exception("Incorrect value of ratio. Possible options: all, minority, not minority")
    val finalIndex: Array[Int] = dataToWorkWith.indices.diff(removedInstances).toArray
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
      logger.addMsg("REMOVED INSTANCES: %s".format(ratio))
      logger.storeFile(file.get)
    }

    newData
  }
}
