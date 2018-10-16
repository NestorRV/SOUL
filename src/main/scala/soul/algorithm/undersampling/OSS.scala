package soul.algorithm.undersampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

/** One-Side Selection core. Original paper: "Addressing the Curse of Imbalanced
  * Training Sets: One-Side Selection" by Miroslav Kubat and Stan Matwin.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param distance distance to use when calling the NNRule core
  * @author Néstor Rodríguez Vico
  */
class OSS(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Count the number of instances for each class
  private[soul] val counter: Map[Any, Int] = this.data.originalClasses.groupBy(identity).mapValues((_: Array[Any]).length)
  // In certain algorithms, reduce the minority class is forbidden, so let's detect what class is it if minorityClass is set to -1.
  // Otherwise, minorityClass will be used as the minority one
  private[soul] val untouchableClass: Any = this.counter.minBy((c: (Any, Int)) => c._2)._1
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.originalClasses.indices.toList)
  // Use normalized data for EUCLIDEAN distance and randomized data
  val dataToWorkWith: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN)
    (this.index map zeroOneNormalization(this.data)).toArray else (this.index map this.data.processedData).toArray
  // and randomized classes to match the randomized data
  val classesToWorkWith: Array[Any] = (this.index map this.data.originalClasses).toArray
  // Distances among the elements
  val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, distance, this.data.fileInfo.nominal, this.data.originalClasses)

  /** Compute the One-Side Selection core.
    *
    * @return data structure with all the important information
    */
  def compute(): Data = {
    // Note: the notation used to refers the subsets of data is the used in the original paper.
    val initTime: Long = System.nanoTime()
    val positives: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (label, i) if label == this.untouchableClass => i }
    val randomElement: Int = classesToWorkWith.indices.diff(positives)(new util.Random(this.seed).nextInt(classesToWorkWith.length - positives.length))
    val c: Array[Int] = positives ++ Array(randomElement)
    val labels: Seq[(Int, Any)] = dataToWorkWith.indices.map { index: Int =>
      (index, nnRule(distances = distances(index), selectedElements = c.diff(List(index)), labels = classesToWorkWith, k = 1)._1)
    }
    val misclassified: Array[Int] = labels.collect { case (i, label) if label != classesToWorkWith(i) => i }.toArray
    val finalC: Array[Int] = (misclassified ++ c).distinct

    val auxData: Data = new Data(originalData = toXData(finalC map dataToWorkWith),
      originalClasses = finalC map classesToWorkWith, fileInfo = this.data.fileInfo)
    val tl = new TL(auxData, file = None, distance = distance, dists = Some((finalC map this.distances).map(finalC map _)))
    tl.untouchableClass_=(this.untouchableClass)
    val resultTL: Data = tl.compute()

    val finalIndex: Array[Int] = (resultTL.index.toList map finalC).toArray
    val finishTime: Long = System.nanoTime()

    this.data.index = (finalIndex map this.index).sorted
    this.data.resultData = (this.index map this.data.originalData).toArray
    this.data.resultClasses = (this.index map this.data.originalClasses).toArray

    if (file.isDefined) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues((_: Array[Any]).length)
      this.logger.addMsg("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(finalIndex.length))
      this.logger.addMsg("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      this.logger.addMsg("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(this.counter, this.untouchableClass)))
      this.logger.addMsg("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, this.untouchableClass)))
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      this.logger.storeFile(file.get)
    }

    this.data
  }
}
