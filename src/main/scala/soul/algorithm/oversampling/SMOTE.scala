package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SMOTE algorithm. Original paper: "SMOTE: Synthetic Minority Over-sampling Technique" by Nitesh V. Chawla, Kevin W.
  * Bowyer, Lawrence O. Hall and W. Philip Kegelmeyer.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param percent  amount of SMOTE N%
  * @param k        number of minority class nearest neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class SMOTE(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
            percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(seed).shuffle(data.y.indices.toList)
  // Data without NA values and with nominal values transformed to numeric values
  private[soul] val (processedData, nomToNum) = processData(data)
  // Samples to work with
  private[soul] val samples: Array[Array[Double]] = if (distance == Distances.EUCLIDEAN) zeroOneNormalization(data, processedData) else processedData

  /** Compute the SMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    val initTime: Long = System.nanoTime()
    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    // check if the percent is correct
    var T: Int = minorityClassIndex.length
    var N: Int = percent

    if (N < 100) {
      T = N / 100 * T
      N = 100
    }
    N = N / 100

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.fill(N * T, samples(0).length)(0.0)

    // index array to save the neighbors of each sample
    var neighbors: Array[Int] = new Array[Int](minorityClassIndex.length)

    var newIndex: Int = 0
    val r: Random = new Random(seed)
    // for each minority class sample
    minorityClassIndex.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data.fileInfo.nominal.length == 0,
        (samples, data.y)).map(minorityClassIndex(_))
      // compute populate for the sample
      (0 until N).foreach(_ => {
        val nn: Int = r.nextInt(neighbors.length)
        // compute attributes of the sample
        samples(i._1).indices.foreach(atrib => {
          val diff: Double = samples(neighbors(nn))(atrib) - samples(i._1)(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(i._1)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      })
    })

    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + output.length).indices.toList).toArray

    // check if the data is nominal or numerical
    val newData: Data = new Data(if (data.fileInfo.nominal.length == 0) {
      dataShuffled map to2Decimals(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output))
    } else {
      dataShuffled map toNominal(Array.concat(processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(output, data.fileInfo.maxAttribs, data.fileInfo.minAttribs) else output), nomToNum)
    }, dataShuffled map Array.concat(data.y, Array.fill(output.length)(minorityClass)),
      Some(dataShuffled.zipWithIndex.collect { case (c, i) if c >= samples.length => i }), data.fileInfo)
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      logger.addMsg("NEW DATA SIZE: %d".format(newData.x.length))
      logger.addMsg("NEW SAMPLES ARE:")
      dataShuffled.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) logger.addMsg("%d".format(index._2)))
      logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      logger.storeFile(file.get)
    }

    newData
  }
}
