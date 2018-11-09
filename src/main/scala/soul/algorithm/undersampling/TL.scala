package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

/** Tomek Link. Original paper: "Two Modifications of CNN" by Ivan Tomek.
  *
  * @param data          data to work with
  * @param seed          seed to use. If it is not provided, it will use the system time
  * @param dist          object of Distance enumeration representing the distance to be used
  * @param ratio         indicates the instances of the Tomek Links that are going to be remove. "all" will remove all instances,
  *                      "minority" will remove instances of the minority class and "not minority" will remove all the instances
  *                      except the ones of the minority class.
  * @param minorityClass minority class. If set to None, it will be computed
  * @param normalize     normalize the data or not
  * @param randomData    iterate through the data randomly or not
  * @param verbose       choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class TL(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN, ratio: String = "not minority",
         val minorityClass: Option[Any] = None, normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the TL algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()
    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = if (minorityClass.isDefined) minorityClass.get else counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)

    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    var randomIndex: List[Int] = data.x.indices.toList
    val classesToWorkWith: Array[Any] = if (randomData) {
      // Index to shuffle (randomize) the data
      randomIndex = random.shuffle(data.y.indices.toList)
      dataToWorkWith = (randomIndex map dataToWorkWith).toArray
      (randomIndex map data.y).toArray
    } else {
      data.y
    }

    val (attrCounter, attrClassesCounter, sds) = if (dist == Distance.HVDM) {
      (dataToWorkWith.transpose.map((column: Array[Double]) => column.groupBy(identity).mapValues(_.length)),
        dataToWorkWith.transpose.map((attribute: Array[Double]) => occurrencesByValueAndClass(attribute, data.y)),
        dataToWorkWith.transpose.map((column: Array[Double]) => standardDeviation(column)))
    } else {
      (null, null, null)
    }

    val candidates: Map[Any, Array[Int]] = classesToWorkWith.distinct.map {
      c: Any =>
        c -> classesToWorkWith.zipWithIndex.collect {
          case (a, b) if a != c => b
        }
    }.toMap

    val distances: Array[Array[Double]] = Array.fill[Array[Double]](dataToWorkWith.length)(new Array[Double](dataToWorkWith.length))

    if (dist == Distance.EUCLIDEAN) {
      dataToWorkWith.indices.par.foreach { i: Int =>
        dataToWorkWith.indices.drop(i).par.foreach { j: Int =>
          distances(i)(j) = euclidean(dataToWorkWith(i), dataToWorkWith(j))
          distances(j)(i) = distances(i)(j)
        }
      }
    } else {
      dataToWorkWith.indices.par.foreach { i: Int =>
        dataToWorkWith.indices.drop(i).par.foreach { j: Int =>
          distances(i)(j) = HVDM(dataToWorkWith(i), dataToWorkWith(j), data.fileInfo.nominal, sds, attrCounter, attrClassesCounter)
          distances(j)(i) = distances(i)(j)
        }
      }
    }

    // Look for the nearest neighbour in the rest of the classes
    val nearestNeighbour: Array[Int] = distances.zipWithIndex.map((row: (Array[Double], Int)) => row._1.indexOf((candidates(classesToWorkWith(row._2)) map row._1).min))
    // For each instance, I: If my nearest neighbour is J and the nearest neighbour of J it's me, I, I and J form a Tomek link
    val tomekLinks: Array[(Int, Int)] = nearestNeighbour.zipWithIndex.filter((pair: (Int, Int)) => nearestNeighbour(pair._1) == pair._2)
    val targetInstances: Array[Int] = tomekLinks.flatMap((x: (Int, Int)) => List(x._1, x._2)).distinct
    // but the user can choose which of them should be removed
    val removedInstances: Array[Int] = if (ratio == "all") targetInstances else if (ratio == "minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a == untouchableClass => b
      } else if (ratio == "not minority")
      targetInstances.zipWithIndex.collect {
        case (a, b) if a != untouchableClass => b
      } else
      throw new Exception("Incorrect value of ratio. Possible options: all, minority, not minority")
    val finalIndex: Array[Int] = dataToWorkWith.indices.diff(removedInstances).toArray
    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
      println("REMOVED INSTANCES: %s".format(ratio))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}
