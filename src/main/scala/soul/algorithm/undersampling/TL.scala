package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** Tomek Link core. Original paper: "Two Modifications of CNN" by Ivan Tomek.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param distance distance to use when calling the NNRule core
  * @param ratio    indicates the instances of the Tomek Links that are going to be remove. "all" will remove all instances,
  *                 "minority" will remove instances of the minority class and "not minority" will remove all the instances
  *                 except the ones of the minority class.
  * @param dists    distances among the elements. If provided, they won't be recalculated.
  * @author Néstor Rodríguez Vico
  */
class TL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
         distance: Distances.Distance = Distances.EUCLIDEAN, ratio: String = "not minority", dists: Option[Array[Array[Double]]] = None) {

  private[soul] val minorityClass: Any = -1
  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = this.data.originalClasses.groupBy(identity).mapValues((_: Array[Any]).length)
  private[this] var untouchableClass: Any = this.counter.minBy((c: (Any, Int)) => c._2)._1

  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.originalClasses.indices.toList)
  // Use normalized data for EUCLIDEAN distance and randomized data
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (this.index map zeroOneNormalization(this.data)).toArray else (this.index map this.data.processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.data.originalClasses).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = if (dists.isDefined) dists.get else computeDistances(dataToWorkWith, distance, this.data.nominal, this.data.originalClasses)

  /** untouchableClass setter
    *
    * @param value new untouchableClass
    */
  private[soul] def untouchableClass_=(value: Any): Unit = {
    this.untouchableClass = value
  }

  /** Undersampling method based in removing Tomek Links
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val candidates: Map[Any, Array[Int]] = classesToWorkWith.distinct.map {
      c: Any =>
        c -> classesToWorkWith.zipWithIndex.collect {
          case (a, b) if a != c => b
        }
    }.toMap

    // Look for the nearest neighbour in the rest of the classes
    val nearestNeighbour: Array[Int] = distances.zipWithIndex.map((row: (Array[Double], Int)) => row._1.indexOf((candidates(classesToWorkWith(row._2)) map row._1).min))
    // For each instance, I: If my nearest neighbour is J and the nearest neighbour of J it's me, I, I and J form a Tomek link
    val tomekLinks: Array[(Int, Int)] = nearestNeighbour.zipWithIndex.filter((pair: (Int, Int)) => nearestNeighbour(pair._1) == pair._2)
    val targetInstances: Array[Int] = tomekLinks.flatMap((x: (Int, Int)) => List(x._1, x._2)).distinct
    // but the user can choose which of them should be removed
    val removedInstances: Array[Int] = if (ratio == "all") targetInstances else if (ratio == "minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a == this.untouchableClass => b
      } else if (ratio == "not minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a != this.untouchableClass => b
      } else
      throw new Exception("Incorrect value of ratio. Possible options: all, minority, not minority")
    val finalIndex: Array[Int] = dataToWorkWith.indices.diff(removedInstances).toArray
    val finishTime: Long = System.nanoTime()

    this.data.index = (finalIndex map this.index).sorted
    this.data.resultData = this.data.index map this.data.originalData
    this.data.resultClasses = this.data.index map this.data.originalClasses

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      this.logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(this.counter, this.untouchableClass)))
      this.logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, this.untouchableClass)))
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      this.logger.addMsg("REMOVED INSTANCES: %s".format(ratio))
      this.logger.storeFile(file.get)
    }

    this.data
  }
}
