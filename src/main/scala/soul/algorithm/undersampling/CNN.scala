package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** Condensed Nearest Neighbor decision rule. Original paper: "The Condensed Nearest Neighbor Rule" by P. Hart.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param distance distance to use when calling the NNRule core
  * @author Néstor Rodríguez Vico
  */
class CNN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
          distance: Distances.Distance = Distances.EUCLIDEAN) {

  private[soul] val minorityClass: Any = -1
  // Remove NA values and change nominal values to numeric values
  private[soul] val x: Array[Array[Double]] = this.data._processedData
  private[soul] val y: Array[Any] = data._originalClasses
  // Logger object to log the execution of the algorithms
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = this.y.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if minorityClass is set to -1.
  // Otherwise, minorityClass will be used as the minority one
  private[soul] val untouchableClass: Any = this.counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.y.indices.toList)
  // Use normalized data for EUCLIDEAN distance and randomized data
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (this.index map zeroOneNormalization(this.data)).toArray else
    (this.index map this.x).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.y).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, this.data._nominal, this.y)

  /** Compute the Condensed Nearest Neighbor decision rule (CNN rule)
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    // Start the time
    val initTime: Long = System.nanoTime()

    if (file.isDefined) {
      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
    }

    // Indicate the corresponding group: 1 for store, 0 for unknown, -1 for grabbag
    val location: Array[Int] = List.fill(dataToWorkWith.length)(0).toArray
    var iteration: Int = 0
    // The first element is added to store
    location(0) = 1
    var changed = true

    // Iterate the data, x (except the first instance)
    dataToWorkWith.zipWithIndex.tail.foreach { element: (Array[Double], Int) =>
      // and classify each element with the actual content of store
      val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
      val label: (Any, Array[Int]) = nnRule(distances = distances(element._2), selectedElements = index, labels = classesToWorkWith, k = 1)
      // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
      location(element._2) = if (label._1 != classesToWorkWith(element._2)) 1 else -1
    }

    if (file.isDefined) {
      this.logger.addMsg("ITERATION %d: GRABBAG SIZE: %d, STORE SIZE: %d.".format(iteration, location.count((z: Int) => z == -1),
        location.count((z: Int) => z == 1)))
    }

    // After a first pass, iterate grabbag until is exhausted:
    // 1. There is no element in grabbag or
    // 2. There is no data change between grabbag and store after a full iteration
    while (location.count((z: Int) => z == -1) != 0 && changed) {
      iteration += 1
      changed = false

      // Now, instead of iterating x, we iterate grabbag
      location.zipWithIndex.filter((x: (Int, Int)) => x._1 == -1).foreach { element: (Int, Int) =>
        val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
        val label: (Any, Array[Int]) = nnRule(distances = distances(element._2), selectedElements = index, labels = classesToWorkWith, k = 1)
        // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
        location(element._2) = if (label._1 != classesToWorkWith(element._2)) {
          changed = true
          1
        } else -1
      }

      if (file.isDefined) {
        this.logger.addMsg("ITERATION %d: GRABBAG SIZE: %d, STORE SIZE: %d.".format(iteration, location.count((z: Int) => z == -1),
          location.count((z: Int) => z == 1)))
      }
    }

    // The final data is the content of store
    val storeIndex: Array[Int] = location.zipWithIndex.filter((x: (Int, Int)) => x._1 == 1).collect { case (_, a) => a }

    // Stop the time
    val finishTime: Long = System.nanoTime()

    this.data._resultData = (storeIndex map this.index).sorted map this.data._originalData
    this.data._resultClasses = (storeIndex map this.index).sorted map this.data._originalClasses
    this.data._index = (storeIndex map this.index).sorted

    if (file.isDefined) {
      // Recount of classes
      val newCounter: Map[Any, Int] = (storeIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)

      this.logger.addMsg("NEW DATA SIZE: %d".format(storeIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (storeIndex.length.toFloat / dataToWorkWith.length) * 100))

      this.logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(this.counter, this.untouchableClass)))
      // Recompute the Imbalanced Ratio
      this.logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, this.untouchableClass)))

      // Save the time
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))

      // Save the log
      this.logger.storeFile(file.get)
    }

    this.data
  }
}
