package soul.algorithm.oversampling

import soul.algorithm.Algorithm
import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Spider2 algorithm. Original paper: "Learning from Imbalanced Data in Presence of Noisy and Borderline Examples" by
  * Krystyna Napiera la, Jerzy Stefanowski and Szymon Wilk.
  *
  * @param data data to work with
  * @param seed seed to use. If it is not provided, it will use the system time
  * @author David López Pretel
  */
class Spider2(private[soul] val data: Data,
              override private[soul] val seed: Long = System.currentTimeMillis()) extends Algorithm {

  // array with the index of the minority class
  private var minorityClassIndex: Array[Int] = minority(data._originalClasses)
  data._minorityClass = data._originalClasses(minorityClassIndex(0))
  // array with the index of the majority class
  private var majorityClassIndex: Array[Int] = data._processedData.indices.diff(minorityClassIndex.toList).toArray
  // the samples computed by the algorithm
  private val output: ArrayBuffer[Array[Double]] = ArrayBuffer()
  private var samples: Array[Array[Double]] = data._processedData
  private var distanceType: Distances.Distance = Distances.HVDM

  /**
    * @param c array of index of samples that belongs to a determined class
    * @param f array of flags that represent if a sample is safe or not
    * @return return a subset of examples that belong to class c and are flagged as f
    */
  def flagged(c: Array[Int], f: Array[Boolean]): Array[Int] = {
    c.map(classes => {
      if (!f(classes)) Some(classes) else None
    }).filterNot(_.forall(_ == None)).map(_.get)
  }

  /** amplifies example x by creating its n-copies
    *
    * @param x index of the element
    * @param k Number of neighbors
    * @return x amplified
    */
  def amplify(x: Int, k: Int): Unit = {
    // compute the neighborhood for the majority and minority class
    val majNeighbors: Array[Int] = kNeighbors(majorityClassIndex map output, output(x), k, distanceType, data._nominal.length == 0, (output.toArray, data._resultClasses))
    val minNeighbors: Array[Int] = kNeighbors(minorityClassIndex map output, output(x), k, distanceType, data._nominal.length == 0, (output.toArray, data._resultClasses))
    // compute the number of copies to create
    val S: Int = Math.abs(majNeighbors.length - minNeighbors.length) + 1
    // need to know the size of the output to save the index of the elements inserted
    val outputSize: Int = output.length
    (0 until S).foreach(_ => {
      output ++= Traversable(output(x))
    })
    // add n copies to the output
    if (data._resultClasses(x) == data._minorityClass) {
      minorityClassIndex = minorityClassIndex ++ (outputSize until outputSize + S)
    } else {
      majorityClassIndex = majorityClassIndex ++ (outputSize until outputSize + S)
    }
    data._resultClasses = data._resultClasses ++ Array.fill(S)(data._resultClasses(x))
  }

  /** classifies example x using its k-nearest neighbors and returns true
    * or false for correct and incorrect classification respectively
    *
    * @param x   index of the element
    * @param k   Number of minority class nearest neighbors
    * @param out indicate if use the output or the input data, true = input data
    * @return true or false
    */
  def correct(x: Int, k: Int, out: Boolean): Boolean = {
    // compute the neighbors
    val neighbors: Array[Int] = kNeighbors(if (out) samples else output.toArray, if (out) samples(x) else output(x), k, distanceType, data._nominal.length == 0, if (out) (samples, data._originalClasses) else (output.toArray, data._resultClasses))
    val classes: scala.collection.mutable.Map[Any, Int] = scala.collection.mutable.Map()
    // compute the number of samples for each class in the neighborhood
    neighbors.foreach(neighbor => classes += data._originalClasses(neighbor) -> 0)
    neighbors.foreach(neighbor => classes(data._originalClasses(neighbor)) += 1)

    // if the majority class in neighborhood is the minority class return true
    if (classes.reduceLeft((x: (Any, Int), y: (Any, Int)) => if (x._2 > y._2) x else y)._1 == data.originalClasses(x))
      true
    else
      false
  }

  /** Compute the Smote algorithm
    *
    * @param file    file to store the log. If its set to None, log process would not be done
    * @param relabel relabeling option
    * @param ampl    amplification option
    * @param k       Number of minority class nearest neighbors
    * @param dType   the type of distance to use, hvdm or euclidean
    * @return synthetic samples generated
    */
  def compute(file: Option[String] = None, relabel: String = "yes", ampl: String = "weak", k: Int = 5, dType: Distances.Distance = Distances.EUCLIDEAN): Unit = {
    if (relabel != "no" && relabel != "yes") {
      throw new Exception("relabel must be yes or no.")
    }
    if (ampl != "weak" && ampl != "strong" && ampl != "no") {
      throw new Exception("amplification must be weak or strong or no.")
    }

    if (dType != Distances.EUCLIDEAN && dType != Distances.HVDM) {
      throw new Exception("The distance must be euclidean or hvdm")
    }

    // Start the time
    val initTime: Long = System.nanoTime()

    distanceType = dType
    if (dType == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    // array with the index of each sample
    var DS: Array[Int] = Array.range(0, samples.length)
    // at the beginning there are not safe samples
    var safeSamples: Array[Boolean] = Array.fill(samples.length)(false)

    // for each sample in majority class check if the neighbors has the same class
    majorityClassIndex.foreach(index => if (correct(index, k, out = true)) safeSamples(index) = true)

    // return a subset of samples that are not safe and belong to the majority class
    val RS: Array[Int] = flagged(majorityClassIndex, safeSamples)
    if (relabel == "yes") {
      //add the RS samples to the minority set
      minorityClassIndex = minorityClassIndex ++ RS
      data._resultClasses = data._originalClasses
      RS.foreach(data.resultClasses(_) = data._minorityClass)
    } else {

      // eliminate the samples from the initial set, first we recalculate the index for min and maj class
      var newIndex: Int = 0
      minorityClassIndex = minorityClassIndex.map(minor => {
        newIndex = minor
        RS.foreach(index => if (index < minor) newIndex -= 1)
        newIndex
      })
      majorityClassIndex = majorityClassIndex.map(major => {
        newIndex = major
        RS.foreach(index => if (index < major) newIndex -= 1)
        newIndex
      })
      DS = DS.diff(RS)
      safeSamples = DS map safeSamples
      data._resultClasses = DS map data._originalClasses
    }

    // the output is DS if ampl is not weak or strong
    output ++= (DS map samples)

    // if the neighbors of each sample in minority class belong to it, flag as safe
    minorityClassIndex.foreach(index => if (correct(index, k, out = false)) safeSamples(index) = true)

    if (ampl == "weak") {
      // for each sample returned by flagged amplify the data creating n copies (n calculated in amplify)
      flagged(minorityClassIndex, safeSamples).foreach(amplify(_, k))
    } else if (ampl == "strong") {
      // if the sample is correct amplify with k, else amplify with k + 2 (k is not n)
      flagged(minorityClassIndex, safeSamples).foreach(x => {
        if (correct(x, k + 2, out = false)) amplify(x, k) else amplify(x, k + 2)
      })
    }

    val r: Random = new Random(this.seed)
    val dataShuffled: Array[Int] = r.shuffle(output.indices.toList).toArray
    // check if the data is nominal or numerical
    if (data._nominal.length == 0) {
      data._resultData = dataShuffled map to2Decimals(if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output.toArray, data._maxAttribs, data._minAttribs) else output.toArray)
    } else {
      data._resultData = dataShuffled map toNominal(if (dType == Distances.EUCLIDEAN) zeroOneDenormalization(output.toArray, data._maxAttribs, data._minAttribs) else output.toArray, data._nomToNum)
    }

    data._resultClasses = dataShuffled map data._resultClasses

    // Stop the time
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      this.logger.addMsg("ORIGINAL SIZE: %d".format(data._originalData.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(data._resultData.length))
      this.logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) this.logger.addMsg("%d".format(index._2)))
      // Save the time
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))

      // Save the log
      this.logger.storeFile(file.get + "_Spider2")
    }
  }
}
