package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SMOTEENN algorithm. Original paper: "A Study of the Behavior of Several Methods for Balancing Machine Learning
  * Training Data" by Gustavo E. A. P. A. Batista, Ronaldo C. Prati and Maria Carolina Monard.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param percent  amount of Smote N%
  * @param k        number of minority class nearest neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class SMOTEENN(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
               percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.originalClasses.indices.toList)

  /** Compute the SMOTEENN algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    var samples: Array[Array[Double]] = data.processedData
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    val initTime: Long = System.nanoTime()
    val minorityClassIndex: Array[Int] = minority(data.originalClasses)
    val minorityClass: Any = data.originalClasses(minorityClassIndex(0))

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
    val r: Random = new Random(this.seed)
    // for each minority class sample
    minorityClassIndex.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data.nominal.length == 0,
        (samples, data.originalClasses)).map(minorityClassIndex(_))
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

    val result: Array[Array[Double]] = Array.concat(samples, output)
    val resultClasses: Array[Any] = Array.concat(data.originalClasses, Array.fill(output.length)(minorityClass))
    // The following code correspond to ENN and it has been made by Néstor Rodríguez Vico

    val shuffle: List[Int] = r.shuffle(resultClasses.indices.toList)

    val dataToWorkWith: Array[Array[Double]] = (shuffle map result).toArray
    // and randomized classes to match the randomized data
    val classesToWorkWith: Array[Any] = (shuffle map resultClasses).toArray

    // Distances among the elements
    val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, Distances.EUCLIDEAN, this.data.fileInfo.nominal, resultClasses)

    val finalIndex: Array[Int] = classesToWorkWith.distinct.flatMap { targetClass: Any =>
      if (targetClass != data.minorityClass) {
        val sameClassIndex: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (c, i) if c == targetClass => i }
        boolToIndex(sameClassIndex.map { i: Int =>
          nnRule(distances = distances(i), selectedElements = sameClassIndex.indices.diff(List(i)).toArray, labels = classesToWorkWith, k = k)._1
        }.map((c: Any) => targetClass == c)) map sameClassIndex
      } else {
        classesToWorkWith.zipWithIndex.collect { case (c, i) if c == targetClass => i }
      }
    }

    // check if the data is nominal or numerical
    if (data.nomToNum(0).isEmpty) {
      data.resultData = to2Decimals(zeroOneDenormalization((finalIndex map shuffle).sorted map result, data.maxAttribs, data.minAttribs))
    } else {
      data.resultData = toNominal(zeroOneDenormalization((finalIndex map shuffle).sorted map result, data.maxAttribs, data.minAttribs), data.nomToNum)
    }
    this.data.resultClasses = (finalIndex map shuffle).sorted map resultClasses
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      this.logger.addMsg("ORIGINAL SIZE: %d".format(data.originalData.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(data.resultData.length))
      this.logger.addMsg("NEW SAMPLES ARE:")
      shuffle.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) this.logger.addMsg("%d".format(index._2)))
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      this.logger.storeFile(file.get)
    }

    this.data
  }
}
