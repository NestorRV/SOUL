package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** Edited Nearest Neighbour rule. Original paper: "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"
  * by Dennis L. Wilson.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param distance distance to use when calling the NNRule core
  * @param k        number of neighbours to use when computing k-NN rule (normally 3 neighbours)
  * @author Néstor Rodríguez Vico
  */
class ENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None, distance: Distances.Distance = Distances.EUCLIDEAN, k: Int = 3) {

  private[soul] val minorityClass: Any = -1
  // Remove NA values and change nominal values to numeric values
  private[soul] val x: Array[Array[Double]] = this.data.processedData
  private[soul] val y: Array[Any] = data.originalClasses
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
    (this.index map zeroOneNormalization(this.data)).toArray else (this.index map this.x).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.y).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, this.data.nominal, this.y)


  /** Compute the Edited Nearest Neighbour rule (ENN rule)
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    // Start the time
    val initTime: Long = System.nanoTime()

    val finalIndex: Array[Int] = classesToWorkWith.distinct.flatMap { targetClass: Any =>
      if (targetClass != this.untouchableClass) {
        val sameClassIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (c, i) if c == targetClass => i }
        boolToIndex(sameClassIndex.map { i: Int =>
          nnRule(distances = distances(i), selectedElements = sameClassIndex.indices.diff(List(i)).toArray, labels = classesToWorkWith, k = k)._1
        }.map((c: Any) => targetClass == c)) map sameClassIndex
      } else {
        classesToWorkWith.zipWithIndex.collect { case (c, i) if c == targetClass => i }
      }
    }

    // Stop the time
    val finishTime: Long = System.nanoTime()

    this.data.resultData = (finalIndex map this.index).sorted map this.data.originalData
    this.data.resultClasses = (finalIndex map this.index).sorted map this.data.originalClasses
    this.data.index = (finalIndex map this.index).sorted

    if (file.isDefined) {
      // Recount of classes
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)

      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))

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
