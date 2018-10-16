package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SafeLevel-SMOTE algorithm. Original paper: "Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling Technique
  * for Handling the Class Imbalanced Problem" by Chumphol Bunkhumpornpat, Krung Sinapiromsaran, and Chidchanok Lursinsap.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param k        Number of nearest neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class SafeLevelSMOTE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
                     k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, nomToNum) = processData(data)

  /** Compute the SafeLevelSMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    var samples: Array[Array[Double]] = processedData
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data, processedData)
    }
    // compute minority class
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    // output with a size of |D|-t samples
    val output: Array[Array[Double]] = Array.fill(minorityClassIndex.length, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    var sl_ratio: Double = 0.0
    val r: Random.type = scala.util.Random
    r.setSeed(seed)
    // for each minority class sample
    minorityClassIndex.foreach(i => {
      // compute k neighbors from p and save number of positive instances
      neighbors = kNeighbors(samples, i, k, distance, data.fileInfo.nominal.length == 0, (samples, data.y))
      val n: Int = neighbors(r.nextInt(neighbors.length))
      val slp: Int = neighbors.map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
          1
        } else {
          0
        }
      }).sum
      // compute k neighbors from n and save number of positive instances
      val sln: Int = kNeighbors(samples, n, k, distance, data.fileInfo.nominal.length == 0, (samples, data.y)).map(neighbor => {
        if (data.y(neighbor) == minorityClass) {
          1
        } else {
          0
        }
      }).sum
      if (sln != 0) { //sl is safe level
        sl_ratio = slp / sln
      } else {
        sl_ratio = 99999999
      }
      if (!(sl_ratio == 99999999 && slp == 0)) {
        // calculate synthetic sample
        var gap: Double = 0.0 // 2 case
        if (sl_ratio == 1) { // 3 case
          gap = r.nextFloat
        } else if (sl_ratio > 1 && sl_ratio != 99999999) { // 4 case
          gap = r.nextFloat * (1 / sl_ratio)
        } else if (sl_ratio < 1) { // 5 case
          gap = r.nextFloat()
          if (gap < 1 - sl_ratio) {
            gap = gap + 1 - sl_ratio
          }
        }
        samples(i).indices.foreach(atrib => {
          val diff: Double = samples(n)(atrib) - samples(i)(atrib)
          output(newIndex)(atrib) = samples(i)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      }
    })

    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + output.length).indices.toList).toArray
    // check if the data is nominal or numerical
    if (data.fileInfo.nominal.length == 0) {
      data.resultData = dataShuffled map to2Decimals(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      data.resultData = dataShuffled map toNominal(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), nomToNum)
    }
    data.resultClasses = dataShuffled map Array.concat(data.y, Array.fill(output.length)(minorityClass))
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      logger.addMsg("NEW DATA SIZE: %d".format(data.resultData.length))
      logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) logger.addMsg("%d".format(index._2)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    data
  }
}
