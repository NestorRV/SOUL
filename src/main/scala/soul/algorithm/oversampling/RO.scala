package soul.algorithm.oversampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.util.Random

/** Random Oversampling algorithm.
  *
  * @param data      data to work with
  * @param seed      seed to use. If it is not provided, it will use the system time
  * @param percent   number of samples to create
  * @param verbose   choose to display information about the execution or not
  * @author David López Pretel
  */
class RO(data: Data, seed: Long = System.currentTimeMillis(), percent: Int = 500, verbose: Boolean = false) {

  /** Compute the SMOTE algorithm
    *
    * @return synthetic samples generated
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    if (percent <  0) {
      throw new Exception("Percent must be a greather than 0")
    }

    val minorityClassIndex: Array[Int] = minority(data.y)
    val minorityClass: Any = data.y(minorityClassIndex(0))

    // output with a size of T*N samples
    val output: Array[Array[Double]] = Array.ofDim[Double](percent, data.processedData(0).length)

    val r: Random = new Random(seed)

    // for each minority class sample
    (0 until percent).par.map((i: Int) => {
      output(i) = data.processedData(minorityClassIndex(r.nextInt(minorityClassIndex.length)))
    })

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      println("ORIGINAL SIZE: %d".format(data.x.length))
      println("NEW DATA SIZE: %d".format(data.x.length + output.length))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(if (data.fileInfo.nominal.length == 0) {
      to2Decimals(Array.concat(data.processedData, output))
    } else {
      toNominal(Array.concat(data.processedData, output), data.nomToNum)
    }, Array.concat(data.y, Array.fill(output.length)(minorityClass)), None, data.fileInfo)
  }
}