package soul.algorithm.undersampling

import soul.data.Data
import soul.util.KDTree
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ListBuffer

/** Condensed Nearest Neighbor decision rule. Original paper: "The Condensed Nearest Neighbor Rule" by P. Hart.
  *
  * @param data       data to work with
  * @param seed       seed to use. If it is not provided, it will use the system time
  * @param dist       object of Distance enumeration representing the distance to be used
  * @param normalize  normalize the data or not
  * @param randomData iterate through the data randomly or not
  * @param verbose    choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class CNN(data: Data, seed: Long = System.currentTimeMillis(), dist: Distance = Distance.EUCLIDEAN,
          normalize: Boolean = false, randomData: Boolean = false, verbose: Boolean = false) {

  /** Compute the CNN algorithm
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
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

    val finalIndex: Array[Int] = if (dist == Distance.HVDM) {
      // Indicate the corresponding group: 1 for store, 0 for unknown, -1 for grabbag
      val location: Array[Int] = List.fill(dataToWorkWith.length)(0).toArray
      // The first element is added to store
      location(0) = 1
      var changed = true

      // Iterate the data, x (except the first instance)
      dataToWorkWith.zipWithIndex.tail.foreach { element: (Array[Double], Int) =>
        // and classify each element with the actual content of store
        val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
        val neighbours: Array[Array[Double]] = index map dataToWorkWith
        val classes: Array[Any] = index map classesToWorkWith
        val label: (Any, Array[Int], Array[Double]) = nnRuleHVDM(neighbours, element._1, -1, classes, 1, data.fileInfo.nominal,
          sds, attrCounter, attrClassesCounter, "nearest")

        // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
        location(element._2) = if (label._1 != classesToWorkWith(element._2)) 1 else -1
      }

      // After a first pass, iterate grabbag until is exhausted:
      // 1. There is no element in grabbag or
      // 2. There is no data change between grabbag and store after a full iteration
      while (location.count((z: Int) => z == -1) != 0 && changed) {
        changed = false
        // Now, instead of iterating x, we iterate grabbag
        location.zipWithIndex.filter((x: (Int, Int)) => x._1 == -1).foreach { element: (Int, Int) =>
          val index: Array[Int] = location.zipWithIndex.collect { case (a, b) if a == 1 => b }
          val neighbours: Array[Array[Double]] = index map dataToWorkWith
          val classes: Array[Any] = index map classesToWorkWith
          val label: Any = nnRuleHVDM(neighbours, dataToWorkWith(element._2), -1, classes, 1, data.fileInfo.nominal,
            sds, attrCounter, attrClassesCounter, "nearest")._1
          // If it is misclassified or is a element of the untouchable class it is added to store; otherwise, it is added to grabbag
          location(element._2) = if (label != classesToWorkWith(element._2)) {
            changed = true
            1
          } else -1
        }
      }

      location.zipWithIndex.filter((x: (Int, Int)) => x._1 == 1).collect { case (_, a) => a }
    } else {
      val store: KDTree = new KDTree(Array(dataToWorkWith(0)), Array(classesToWorkWith(0)), dataToWorkWith(0).length)
      var grabbag: ListBuffer[(Array[Double], Int)] = new ListBuffer[(Array[Double], Int)]()
      var newGrabbag: ListBuffer[(Array[Double], Int)] = new ListBuffer[(Array[Double], Int)]()

      // Iterate the data, x (except the first instance)
      dataToWorkWith.zipWithIndex.tail.foreach { instance: (Array[Double], Int) =>
        val label = mode(store.nNeighbours(instance._1, k = 1, leaveOneOut = false)._2.toArray)
        if(label != classesToWorkWith(instance._2)){
          store.addElement(instance._1, classesToWorkWith(instance._2))
        } else {
          grabbag += instance
        }
      }

      var changed = true
      while (grabbag.nonEmpty && changed) {
        changed = false
        grabbag.foreach { instance =>
          val label = mode(store.nNeighbours(instance._1, k = 1, leaveOneOut = false)._2.toArray)
          if(label != classesToWorkWith(instance._2)){
            store.addElement(instance._1, classesToWorkWith(instance._2))
            changed = true
          } else {
            newGrabbag += instance
          }
        }

        grabbag = newGrabbag
        newGrabbag = new ListBuffer[(Array[Double], Int)]()
      }

      store.kDTreeMap.values.unzip._2.toArray
    }

    val finishTime: Long = System.nanoTime()

    if (verbose) {
      val newCounter: Map[Any, Int] = (finalIndex map classesToWorkWith).groupBy(identity).mapValues(_.length)
      println("ORIGINAL SIZE: %d".format(dataToWorkWith.length))
      println("NEW DATA SIZE: %d".format(finalIndex.length))
      println("REDUCTION PERCENTAGE: %s".format(100 - (finalIndex.length.toFloat / dataToWorkWith.length) * 100))
      println("ORIGINAL IMBALANCED RATIO: %s".format(imbalancedRatio(counter, untouchableClass)))
      println("NEW IMBALANCED RATIO: %s".format(imbalancedRatio(newCounter, untouchableClass)))
      println("TOTAL ELAPSED TIME: %s".format(nanoTimeToString(finishTime - initTime)))
    }

    new Data(finalIndex map data.x, finalIndex map data.y, Some(finalIndex), data.fileInfo)
  }
}
