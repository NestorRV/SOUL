package soul.algorithm.oversampling

import com.typesafe.scalalogging.LazyLogging
import soul.data.Data
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer

/** Spider2 algorithm. Original paper: "Learning from Imbalanced Data in Presence of Noisy and Borderline Examples" by
  * Krystyna Napiera la, Jerzy Stefanowski and Szymon Wilk.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param relabel   relabeling option
  * @param ampl      amplification option
  * @param k         number of minority class nearest neighbors
  * @param dist      object of DistanceType representing the distance to be used
  * @param normalize normalize the data or not
  * @author David López Pretel
  */
class Spider2(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
              relabel: String = "yes", ampl: String = "weak", k: Int = 5, dist: DistanceType = Distance(euclideanDistance),
              val normalize: Boolean = false) extends LazyLogging {

  // array with the index of the minority class
  private var minorityClassIndex: Array[Int] = minority(data.y)
  private val minorityClass: Any = data.y(minorityClassIndex(0))
  // array with the index of the majority class
  private var majorityClassIndex: Array[Int] = data.processedData.indices.diff(minorityClassIndex.toList).toArray
  // the samples computed by the algorithm
  private val output: ArrayBuffer[Array[Double]] = ArrayBuffer()
  private var resultClasses: Array[Any] = _

  /** Compute the Spider2 algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    if (relabel != "no" && relabel != "yes") {
      throw new Exception("relabel must be yes or no.")
    }

    if (ampl != "weak" && ampl != "strong" && ampl != "no") {
      throw new Exception("amplification must be weak or strong or no.")
    }

    val initTime: Long = System.nanoTime()
    val samples: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData

    val (attrCounter, attrClassesCounter, sds) = if (dist.isInstanceOf[HVDM]) {
      (samples.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        samples.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        samples.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    def flagged(c: Array[Int], f: Array[Boolean]): Array[Int] = {
      c.map(classes => {
        if (!f(classes)) Some(classes) else None
      }).filterNot(_.forall(_ == None)).map(_.get)
    }

    def amplify(x: Int, k: Int): Unit = {
      // compute the neighborhood for the majority and minority class
      val majNeighbors: Array[Int] = dist match {
        case distance: Distance =>
          kNeighbors(majorityClassIndex map output, output(x), k, distance)
        case _ =>
          kNeighborsHVDM(majorityClassIndex map output, output(x), k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }

      val minNeighbors: Array[Int] = dist match {
        case distance: Distance =>
          kNeighbors(minorityClassIndex map output, output(x), k, distance)
        case _ =>
          kNeighborsHVDM(minorityClassIndex map output, output(x), k, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }

      // compute the number of copies to create
      val S: Int = Math.abs(majNeighbors.length - minNeighbors.length) + 1
      // need to know the size of the output to save the randomIndex of the elements inserted
      val outputSize: Int = output.length
      (0 until S).foreach(_ => {
        output ++= Traversable(output(x))
      })
      // add n copies to the output
      if (resultClasses(x) == minorityClass) {
        minorityClassIndex = minorityClassIndex ++ (outputSize until outputSize + S)
      } else {
        majorityClassIndex = majorityClassIndex ++ (outputSize until outputSize + S)
      }
      resultClasses = resultClasses ++ Array.fill(S)(resultClasses(x))
    }

    def correct(x: Int, k: Int, out: Boolean): Boolean = {
      // compute the neighbors
      val neighbors: Array[Int] = dist match {
        case distance: Distance =>
          kNeighbors(if (out) samples else output.toArray, if (out) samples(x) else output(x), k, distance)
        case _ =>
          kNeighborsHVDM(if (out) samples else output.toArray, if (out) samples(x) else output(x), k,
            data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
      }
      val classes: scala.collection.mutable.Map[Any, Int] = scala.collection.mutable.Map()
      // compute the number of samples for each class in the neighborhood
      neighbors.foreach(neighbor => classes += data.y(neighbor) -> 0)
      neighbors.foreach(neighbor => classes(data.y(neighbor)) += 1)

      // if the majority class in neighborhood is the minority class return true
      if (classes.reduceLeft((x: (Any, Int), y: (Any, Int)) => if (x._2 > y._2) x else y)._1 == data.y(x))
        true
      else
        false
    }

    // array with the randomIndex of each sample
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
      resultClasses = data.y
      RS.foreach(resultClasses(_) = minorityClass)
    } else {

      // eliminate the samples from the initial set, first we recalculate the randomIndex for min and maj class
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
      resultClasses = DS map data.y
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

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(if (normalize) zeroOneDenormalization(output.toArray, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output.toArray)
    } else {
      toNominal(if (normalize) zeroOneDenormalization(output.toArray, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output.toArray, data.nomToNum)
    }, resultClasses, None, data.fileInfo)

    val finishTime: Long = System.nanoTime()

    logger.whenInfoEnabled {
      logger.info("ORIGINAL SIZE: %d".format(data.x.length))
      logger.info("NEW DATA SIZE: %d".format(newData.x.length))
      logger.info("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    newData
  }
}
