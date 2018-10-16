package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.util.Random

/** SMOTETL algorithm. Original paper: "A Study of the Behavior of Several Methods for Balancing Machine Learning
  * Training Data" by Gustavo E. A. P. A. Batista, Ronaldo C. Prati and Maria Carolina Monard.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param percent  Amount of Smote N%
  * @param k        Number of minority class nearest neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class SMOTETL(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
              percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithm
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.y.indices.toList)

  /** Compute the SMOTETL algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    val initTime: Long = System.nanoTime()
    var samples: Array[Array[Double]] = data.processedData
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    // compute minority class
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
    val r: Random = new Random(this.seed)
    // for each minority class sample
    minorityClassIndex.zipWithIndex.foreach(i => {
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, this.data.fileInfo.nominal.length == 0,
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

    val result: Array[Array[Double]] = Array.concat(samples, output)
    val resultClasses: Array[Any] = Array.concat(data.y, Array.fill(output.length)(minorityClass))

    // The following code correspond to TL and it has been made by Néstor Rodríguez Vico
    val shuffle: List[Int] = r.shuffle(resultClasses.indices.toList)

    val dataToWorkWith: Array[Array[Double]] = (shuffle map result).toArray
    // and randomized classes to match the randomized data
    val classesToWorkWith: Array[Any] = (shuffle map resultClasses).toArray
    // Distances among the elements
    val distances: Array[Array[Double]] = computeDistances(dataToWorkWith, Distances.EUCLIDEAN, this.data.fileInfo.nominal, resultClasses)

    // Take the index of the elements that have a different class
    val candidates: Map[Any, Array[Int]] = classesToWorkWith.distinct.map { c: Any =>
      c -> classesToWorkWith.zipWithIndex.collect { case (a, b) if a != c => b }
    }.toMap

    // Look for the nearest neighbour in the rest of the classes
    val nearestNeighbour: Array[Int] = distances.zipWithIndex.map((row: (Array[Double], Int)) => row._1.indexOf((candidates(classesToWorkWith(row._2)) map row._1).min))

    // For each instance, I: If my nearest neighbour is J and the nearest neighbour of J it's me, I, I and J form a Tomek link
    val tomekLinks: Array[(Int, Int)] = nearestNeighbour.zipWithIndex.filter((pair: (Int, Int)) => nearestNeighbour(pair._1) == pair._2)

    // Instances that form a Tomek link are going to be removed
    val targetInstances: Array[Int] = tomekLinks.flatMap((x: (Int, Int)) => List(x._1, x._2)).distinct
    // but the user can choose which of them should be removed
    val removedInstances: Array[Int] = targetInstances
    // Get the final index
    val finalIndex: Array[Int] = dataToWorkWith.indices.diff(removedInstances).toArray

    // check if the data is nominal or numerical
    if (data.nomToNum(0).isEmpty) {
      data.resultData = to2Decimals(zeroOneDenormalization((finalIndex map shuffle).sorted map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs))
    } else {
      data.resultData = toNominal(zeroOneDenormalization((finalIndex map shuffle).sorted map result, data.fileInfo.maxAttribs, data.fileInfo.minAttribs), data.nomToNum)
    }
    this.data.resultClasses = (finalIndex map shuffle).sorted map resultClasses
    val finishTime: Long = System.nanoTime()

    if (file.isDefined) {
      this.logger.addMsg("ORIGINAL SIZE: %d".format(data.x.length))
      this.logger.addMsg("NEW DATA SIZE: %d".format(data.resultData.length))
      this.logger.addMsg("NEW SAMPLES ARE:")
      shuffle.zipWithIndex.foreach((index: (Int, Int)) => if (index._1 >= samples.length) this.logger.addMsg("%d".format(index._2)))
      this.logger.addMsg("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      this.logger.storeFile(file.get)
    }

    this.data
  }
}
