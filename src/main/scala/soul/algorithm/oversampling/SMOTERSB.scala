package soul.algorithm.oversampling

import soul.data.Data
import soul.io.Logger
import soul.util.Utilities._

import scala.Array._
import scala.util.Random

/** SMOTERSB algorithm. Original paper: "kNN Approach to Unbalanced Data Distribution: SMOTE-RSB: a hybrid preprocessing
  * approach based on oversampling and undersampling for high imbalanced data-sets using SMOTE and rough sets theory"
  * by Enislay Ramentol, Yailé Caballero, Rafael Bello and Francisco Herrera.
  *
  * @param data     data to work with
  * @param seed     seed to use. If it is not provided, it will use the system time
  * @param file     file to store the log. If its set to None, log process would not be done
  * @param percent  amount of Smote N%
  * @param k        number of minority class nearest neighbors
  * @param distance the type of distance to use, hvdm or euclidean
  * @author David López Pretel
  */
class SMOTERSB(private[soul] val data: Data, private[soul] val seed: Long = System.currentTimeMillis(), file: Option[String] = None,
               percent: Int = 500, k: Int = 5, distance: Distances.Distance = Distances.EUCLIDEAN) {

  // Logger object to log the execution of the algorithms
  private[soul] val logger: Logger = new Logger
  // Index to shuffle (randomize) the data
  private[soul] val index: List[Int] = new util.Random(this.seed).shuffle(this.data.originalClasses.indices.toList)

  /** Compute the SMOTERSB algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Unit = {
    if (percent > 100 && percent % 100 != 0) {
      throw new Exception("Percent must be a multiple of 100")
    }

    // Start the time
    val initTime: Long = System.nanoTime()

    var samples: Array[Array[Double]] = data._processedData
    if (distance == Distances.EUCLIDEAN) {
      samples = zeroOneNormalization(data)
    }

    // calculate minority class
    val minorityClassIndex: Array[Int] = minority(data._originalClasses)
    data._minorityClass = data._originalClasses(minorityClassIndex(0))

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
      neighbors = kNeighbors(minorityClassIndex map samples, i._2, k, distance, data._nominal.length == 0,
        (samples, data._originalClasses)).map(minorityClassIndex(_))
      // calculate populate for the sample
      (0 until N).foreach(_ => {
        val nn: Int = r.nextInt(neighbors.length)
        // calculate attributes of the sample
        samples(i._1).indices.foreach(atrib => {
          val diff: Double = samples(neighbors(nn))(atrib) - samples(i._1)(atrib)
          val gap: Float = r.nextFloat
          output(newIndex)(atrib) = samples(i._1)(atrib) + gap * diff
        })
        newIndex = newIndex + 1
      })
    })

    //compute the majority class
    val majorityClassIndex: Array[Int] = samples.indices.diff(minorityClassIndex.toList).toArray

    // minimum and maximum value for each attrib
    val Amax: Array[Double] = Array.concat(majorityClassIndex map samples, output).transpose.map(column => column.max)
    val Amin: Array[Double] = Array.concat(majorityClassIndex map samples, output).transpose.map(column => column.min)

    //compute the similarity matrix
    val similarityMatrix: Array[Array[Double]] = output.map(i => {
      (majorityClassIndex map samples).map(j => {
        i.indices.map(k => {
          if (data._nomToNum(0).isEmpty) {
            1 - (Math.abs(i(k) - j(k)) / (Amax(k) - Amin(k))) // this expression must be multiplied by wk
          } else { // but all the features are included, so wk is 1
            if (i(k) == j(k)) 1 else 0
          }
        }).sum / i.length
      })
    })

    var result: Array[Int] = Array()
    var similarityValue: Double = 0.4
    var lowerApproximation: Boolean = false

    while (similarityValue < 0.9) {
      output.indices.foreach(i => {
        lowerApproximation = false
        majorityClassIndex.indices.foreach(j => {
          if (similarityMatrix(i)(j) > similarityValue)
            lowerApproximation = true
        })
        if (!lowerApproximation) result = result :+ i
      })
      similarityValue += 0.05
    }

    //if there are not synthetic samples with lower approximation, return all synthetic samples
    if (result.length == 0) {
      result = Array.range(0, output.length)
    }

    val dataShuffled: Array[Int] = r.shuffle((0 until samples.length + result.length).indices.toList).toArray
    // check if the data is nominal or numerical
    if (data._nominal.length == 0) {
      data._resultData = dataShuffled map to2Decimals(Array.concat(data._processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(result map output, data._maxAttribs, data._minAttribs) else result map output))
    } else {
      data._resultData = dataShuffled map toNominal(Array.concat(data._processedData, if (distance == Distances.EUCLIDEAN)
        zeroOneDenormalization(result map output, data._maxAttribs, data._minAttribs) else result map output), data._nomToNum)
    }
    data._resultClasses = dataShuffled map Array.concat(data._originalClasses, Array.fill((result map output).length)(data._minorityClass))

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
      this.logger.storeFile(file.get)
    }
  }
}
