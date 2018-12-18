package soul.util

import java.util
import java.util.concurrent.TimeUnit

import soul.data.{Data, FileInfo}
import weka.core.{Attribute, DenseInstance, Instance, Instances}

import scala.Array.range
import scala.collection.mutable.ArrayBuffer
import scala.collection.{immutable, mutable}
import scala.math.{abs, pow, sqrt}
import scala.util.Random

/** Set of utilities functions
  *
  * @author Néstor Rodríguez Vico
  */
object Utilities {

  /** Enumeration to represent the possible distances
    *
    * EUCLIDEAN: Euclidean distance
    * HVDM: Proposed in "Improved Heterogeneous Distance Functions" by "D. Randall Wilson and Tony R. Martinez"
    *
    */
  object Distance extends Enumeration {
    type Distance = Value
    val HVDM, EUCLIDEAN = Value
  }

  /** Return an array of the indices that are true
    *
    * @param data boolean array to convert
    * @return indices
    */
  def boolToIndex(data: Array[Boolean]): Array[Int] = {
    data.zipWithIndex.collect { case (v, i) if v => i }
  }

  /** Build a weka Instances object for custom data
    *
    * @param data     set of "instances"
    * @param classes  response of instances
    * @param fileInfo additional information
    * @return weka instances
    */
  def buildInstances(data: Array[Array[Double]], classes: Array[Any], fileInfo: FileInfo): Instances = {
    val possibleClasses: Array[String] = fileInfo._attributesValues("Class").replaceAll("[{} ]", "").split(",")
    var counter: Int = -1
    val dict: Map[Any, Int] = possibleClasses.map { value: Any => counter += 1; value -> counter }.toMap

    val attributes: util.ArrayList[Attribute] = new util.ArrayList[Attribute]
    val classesValues: util.ArrayList[String] = new util.ArrayList[String]
    possibleClasses.foreach((e: Any) => classesValues.add(e.toString))
    attributes.add(new Attribute("RESPONSE", classesValues))

    if (fileInfo._header == null) {
      data(0).indices.foreach((i: Int) => attributes.add(new Attribute("attribute_" + i)))
    } else {
      fileInfo._header.foreach((i: String) => attributes.add(new Attribute(i)))
    }

    val instances = new Instances("Instances", attributes, 0)

    (data zip classes).foreach { i: (Array[Double], Any) =>
      val instance: Instance = new DenseInstance(i._1.length + 1)
      instance.setValue(0, dict(i._2))
      i._1.zipWithIndex.foreach((e: (Double, Int)) => instance.setValue(e._2 + 1, e._1))
      instance.setDataset(instances)
      instances.add(instance)
    }

    instances.setClassIndex(0)

    instances
  }

  /** return the element by their probabilities
    *
    * @param probs   the probabilities
    * @param probSum the sum of all probabilities
    * @param rand    the random generator
    * @return the chosen element
    */
  def chooseByProb(probs: Array[(Double, Int)], probSum: Double, rand: Random): Int = {
    val index: Double = 0 + rand.nextDouble() * (probSum - 0) // random between 0 and probSum
    var sum: Double = 0
    var i: Int = 0
    while (sum < index) {
      sum = sum + probs(i)._1
      i += 1
    }
    probs(Math.max(0, i - 1))._2
  }

  /** Compute the number of true positives (tp), false positives (fp), true negatives (tn) and false negatives (fn)
    *
    * @param originalLabels  original labels
    * @param predictedLabels labels predicted by a classifier
    * @param minorityClass   positive class
    * @return (tp, fp, tn, fn)
    */
  def confusionMatrix(originalLabels: Array[Any], predictedLabels: Array[Any], minorityClass: Any): (Int, Int, Int, Int) = {
    var tp: Int = 0
    var fp: Int = 0
    var fn: Int = 0
    var tn: Int = 0

    (predictedLabels zip originalLabels).foreach { cs: (Any, Any) =>
      if (cs._1 == minorityClass && cs._2 == minorityClass)
        tp += 1
      else if (cs._1 == minorityClass && cs._2 != minorityClass)
        fp += 1
      else if (cs._1 != minorityClass && cs._2 == minorityClass)
        fn += 1
      else if (cs._1 != minorityClass && cs._2 != minorityClass)
        tn += 1
    }

    (tp, fp, tn, fn)
  }

  /** Compute the Euclidean Distance between two points
    *
    * @param x first instance
    * @param y second instance
    * @return euclidean distance between the instances
    */
  def euclidean(x: Array[Double], y: Array[Double]): Double = {
    var d: Double = 0.0
    var i: Int = 0
    while (i < x.length) {
      val toPow2 = x(i) - y(i)
      d += toPow2 * toPow2
      i += 1
    }
    sqrt(d)
  }

  /** Compute HVDM distance of two nodes
    *
    * @param n1                 sample1
    * @param n2                 sample2
    * @param nominal            indicate nominal attributes in the instances
    * @param sds                standard deviations
    * @param attrCounter        counter attributes occurrences
    * @param attrClassesCounter number of occurrences for each value and output class c, for each class
    * @return HVDM distance of two nodes
    */
  def HVDM(n1: Array[Double], n2: Array[Double], nominal: Array[Int], sds: Array[Double],
           attrCounter: Array[Map[Double, Int]], attrClassesCounter: Array[Map[Double, Map[Any, Int]]]): Double = {
    def normalized_diff(x: Double, y: Double, sd: Double): Double = abs(x - y) / (4 * sd)

    def normalized_vdm(nax: Double, nay: Double, naxClasses: Map[Any, Int], nayClasses: Map[Any, Int]): Double = {
      sqrt((naxClasses.values zip nayClasses.values).map((element: (Int, Int)) => pow(abs(element._1 / nax - element._2 / nay), 2)).sum)
    }

    (n1 zip n2).zipWithIndex.map((element: ((Double, Double), Int)) =>
      if (nominal.contains(element._2))
        normalized_vdm(nax = attrCounter(element._2)(element._1._1), nay = attrCounter(element._2)(element._1._1),
          naxClasses = attrClassesCounter(element._2)(element._1._1), nayClasses = attrClassesCounter(element._2)(element._1._2)) else
        normalized_diff(element._1._1, element._1._2, sds(element._2))).sum
  }

  /** Compute the soul ratio (number of instances of all the classes except the minority one divided by number of
    * instances of the minority class)
    *
    * @param counter       Array containing a pair representing: (class, number of elements)
    * @param minorityClass indicates which is the minority class
    * @return the soul ratio
    */
  def imbalancedRatio(counter: Map[Any, Int], minorityClass: Any): Double = {
    try {
      (counter.values.sum.toFloat - counter(minorityClass)) / counter(minorityClass)
    } catch {
      case _: Exception => Double.NaN
    }
  }

  /** Split the data into nFolds folds and predict the labels using the test
    *
    * @param data   target data
    * @param labels labels associated to each point in data
    * @param k      number of neighbours to consider
    * @param nFolds number of subsets to create
    * @param which  "nearest" to return the nearest neighbours, otherwise, return the farthest ones
    * @return the predictedLabels with less error
    */
  def kFoldPrediction(data: Array[Array[Double]], labels: Array[Any], k: Int, nFolds: Int, which: String): Array[Any] = {
    val indices: List[List[Int]] = scala.util.Random.shuffle(labels.indices.toList).grouped((labels.length.toFloat / nFolds).ceil.toInt).toList
    val predictedLabels: Array[(Int, Array[Any])] = indices.par.map { test: List[Int] =>
      val targetData: Array[Array[Double]] = (test map data).toArray
      val targetClasses: Array[Any] = (test map labels).toArray
      val KDTree: KDTree = new KDTree(targetData, targetClasses, targetData(0).length)
      val predictedLabels: Array[(Int, Any)] = test.map { e =>
        (e, mode(KDTree.nNeighbours(data(e), k, leaveOneOut = false)._2.toArray))
      }.toArray
      val error: Int = predictedLabels.count((e: (Int, Any)) => e._2 != labels(e._1))
      (error, predictedLabels.unzip._2)
    }.toArray

    predictedLabels.minBy(_._1)._2
  }

  /** Split the data into nFolds folds and predict the labels using the test
    *
    * @param data               target data
    * @param labels             labels associated to each point in data
    * @param k                  number of neighbours to consider
    * @param nFolds             number of subsets to create
    * @param nominal            indicate nominal attributes in the instances
    * @param sds                standard deviations
    * @param attrCounter        counter attributes occurrences
    * @param attrClassesCounter number of occurrences for each value and output class c, for each class
    * @param which              "nearest" to return the nearest neighbours, otherwise, return the farthest ones
    * @return the predictedLabels with less error
    */
  def kFoldPredictionHVDM(data: Array[Array[Double]], labels: Array[Any], k: Int, nFolds: Int, nominal: Array[Int], sds: Array[Double],
                          attrCounter: Array[Map[Double, Int]], attrClassesCounter: Array[Map[Double, Map[Any, Int]]], which: String): Array[Any] = {

    val indices: List[List[Int]] = scala.util.Random.shuffle(labels.indices.toList).grouped((labels.length.toFloat / nFolds).ceil.toInt).toList
    val predictedLabels: Array[(Int, Array[Any])] = indices.map { test: List[Int] =>
      val neighbours: Array[Array[Double]] = (labels.indices.diff(test) map data).toArray
      val classes: Array[Any] = (labels.indices.diff(test) map labels).toArray
      val predictedLabels: Array[(Int, Any)] = labels.indices.diff(test).map { i: Int =>
        (i, nnRuleHVDM(neighbours, data(i), -1, classes, k, nominal, sds, attrCounter, attrClassesCounter, which)._1)
      }.toArray

      val error: Int = predictedLabels.count((e: (Int, Any)) => e._2 != labels(e._1))
      (error, predictedLabels.sortBy(_._1).unzip._2)
    }.toArray

    predictedLabels.minBy(_._1)._2
  }

  /** Compute KMeans core
    *
    * @param data          data to be clustered
    * @param nominal       array to know which attributes are nominal
    * @param numClusters   number of clusters to be created
    * @param restarts      number of times to relaunch the core
    * @param minDispersion stop if dispersion is lower than this value
    * @param maxIterations number of iterations to be done in KMeans core
    * @param seed          seed to initialize the random object
    * @return (dispersion, centroids of the cluster, a map of the form: clusterID -> Array of elements in this cluster,
    *         a map of the form: elementID -> cluster associated)
    */
  def kMeans(data: Array[Array[Double]], nominal: Array[Int], numClusters: Int, restarts: Int, minDispersion: Double,
             maxIterations: Int, seed: Long): (Double, Array[Array[Double]], mutable.Map[Int, Array[Int]]) = {

    def run(centroids: Array[Array[Double]], minChangeInDispersion: Double, maxIterations: Int): (Double, Array[Array[Double]], mutable.Map[Int, Array[Int]]) = {

      def clusterIndex(data: Array[Array[Double]], centroids: Array[Array[Double]]): (Double, Array[Int]) = {
        val (distances, memberships) = data.par.map { element: Array[Double] =>
          val distances: Array[Double] = centroids.map((c: Array[Double]) => euclidean(c, element))
          val (bestDistance, bestCentroid) = distances.zipWithIndex.min
          (bestDistance * bestDistance, bestCentroid)
        }.toArray.unzip
        (distances.sum, memberships)
      }

      def getCentroids(data: Array[Array[Double]], memberships: Array[Int], numClusters: Int): (Array[Array[Double]],
        mutable.Map[Int, Array[Int]]) = {
        val assignment: mutable.Map[Int, ArrayBuffer[Int]] = mutable.LinkedHashMap[Int, ArrayBuffer[Int]]()
        (0 until numClusters).toList.foreach((c: Int) => assignment(c) = new ArrayBuffer[Int](0))

        for (index <- data.indices) {
          val clusterId = memberships(index)
          if (clusterId > -1)
            assignment(clusterId) += index
        }

        val centroids: Array[Array[Double]] = assignment.map((e: (Int, ArrayBuffer[Int])) => for (column <- (e._2.toArray map data).transpose) yield {
          column.sum / column.length
        }).toArray

        (centroids, assignment.map((element: (Int, ArrayBuffer[Int])) => element._1 -> element._2.toArray))
      }

      val numClusters: Int = centroids.length
      var iteration: Int = 0
      var lastDispersion: Double = Double.PositiveInfinity
      var dispersionDiff: Double = Double.PositiveInfinity
      var newCentroids: Array[Array[Double]] = centroids
      var assignment: mutable.Map[Int, Array[Int]] = mutable.LinkedHashMap[Int, Array[Int]]()

      while (iteration < maxIterations && dispersionDiff > minChangeInDispersion) {
        val (dispersion: Double, memberships: Array[Int]) = clusterIndex(data, newCentroids)
        val aux: (Array[Array[Double]], mutable.Map[Int, Array[Int]]) = getCentroids(data, memberships, numClusters)
        newCentroids = aux._1
        assignment = aux._2
        dispersionDiff = math.abs(lastDispersion - dispersion)
        lastDispersion = dispersion
        iteration += 1
      }
      (lastDispersion, newCentroids, assignment)
    }

    val centroids: Array[Array[Double]] = new scala.util.Random(seed).shuffle(data.indices.toList).toArray.take(numClusters) map data
    val results: immutable.IndexedSeq[(Double, Array[Array[Double]], mutable.Map[Int, Array[Int]])] = (1 to restarts).map(_ => run(centroids, minDispersion, maxIterations))
    val (bestDispersion, bestCentroids, bestAssignment) = results.minBy(_._1)
    (bestDispersion, bestCentroids, bestAssignment)
  }

  /** Compute kNN core
    *
    * @param data               array of samples
    * @param node               array with the attributes of the node
    * @param k                  number of neighbors
    * @param nominal            indicate nominal attributes in the instances
    * @param sds                standard deviations
    * @param attrCounter        counter attributes occurrences
    * @param attrClassesCounter number of occurrences for each value and output class c, for each class
    * @return index of the neighbors of node
    */
  def kNeighborsHVDM(data: Array[Array[Double]], node: Array[Double], k: Int, nominal: Array[Int], sds: Array[Double],
                     attrCounter: Array[Map[Double, Int]], attrClassesCounter: Array[Map[Double, Map[Any, Int]]]): Array[Int] = {
    val distances: Array[Double] = new Array[Double](data.length)

    data.indices.foreach(i => {
      distances(i) = HVDM(node, data(i), nominal, sds, attrCounter, attrClassesCounter)
      if (distances(i) == 0) {
        distances(i) = 9999999
      }
    })
    val result = distances.toList.view.zipWithIndex.sortBy(_._1)
    val index = result.unzip._2

    var kk: Int = k
    if (k > distances.length) {
      kk = distances.length
    }
    range(0, kk).map(i => index.toList(i))
  }


  /** Compute kNN core
    *
    * @param data               array of samples
    * @param node               index whom neighbors are going to be evaluated
    * @param k                  number of neighbors
    * @param nominal            indicate nominal attributes in the instances
    * @param sds                standard deviations
    * @param attrCounter        counter attributes occurrences
    * @param attrClassesCounter number of occurrences for each value and output class c, for each class
    * @return index of the neighbors of node
    */
  def kNeighborsHVDM(data: Array[Array[Double]], node: Int, k: Int, nominal: Array[Int], sds: Array[Double],
                     attrCounter: Array[Map[Double, Int]], attrClassesCounter: Array[Map[Double, Map[Any, Int]]]): Array[Int] = {
    val distances: Array[Double] = Array.fill[Double](data.length)(99999999)

    val kNeighbors: Array[Int] = (0 until k).toArray
    var i: Int = 0
    var j: Int = 0
    var found: Boolean = false
    //compute distances
    while (i < data.length) {
      if (i != node) {
        j = 0
        found = false
        distances(i) = HVDM(data(i), data(node), nominal, sds, attrCounter, attrClassesCounter)
        //save the best neighbors
        while (j < kNeighbors.length && !found) {
          if (distances(i) < distances(kNeighbors(j))) {
            found = true
          } else {
            j = j + 1
          }
        }
        if (found) {
          kNeighbors(j) = i
        }
      }
      i = i + 1
    }

    kNeighbors
  }

  /** Compute kNN core
    *
    * @param data array of samples
    * @param node array with the attributes of the node
    * @param k    number of neighbors
    * @return index of the neighbors of node
    */
  def kNeighbors(data: Array[Array[Double]], node: Array[Double], k: Int): Array[Int] = {
    val kNeighbors: Array[(Int, Double)] = (0 until k).map((_, 99999999.0)).toArray
    var i: Int = 0
    var j: Int = 0
    var maxDistance: (Double, Int) = (0.0, 0)
    var distance: Double = 0
    //compute distances
    while (i < data.length) {
      j = 0
      maxDistance = (0.0, 0)
      distance = euclidean(data(i), node)
      //save the best neighbors
      while (j < kNeighbors.length) {
        if (maxDistance._1 < kNeighbors(j)._2) {
          maxDistance = (kNeighbors(j)._2, j)
        }
        j = j + 1
      }
      if (distance < maxDistance._1) {
        kNeighbors(maxDistance._2) = (i, distance)
      }
      i = i + 1
    }

    kNeighbors.map(_._1)
  }

  /** Compute kNN core
    *
    * @param data array of samples
    * @param node index whom neighbors are going to be evaluated
    * @param k    number of neighbors
    * @return index of the neighbors of node
    */
  def kNeighbors(data: Array[Array[Double]], node: Int, k: Int): Array[Int] = {
    val kNeighbors: Array[(Int, Double)] = (0 until k).map((_, 99999999.0)).toArray
    var i: Int = 0
    var j: Int = 0
    var maxDistance: (Double, Int) = (0.0, 0)
    var distance: Double = 0
    //compute distances
    while (i < data.length) {
      j = 0
      maxDistance = (0.0, 0)
      distance = euclidean(data(i), data(node))
      //save the best neighbors
      while (j < kNeighbors.length) {
        if (maxDistance._1 < kNeighbors(j)._2) {
          maxDistance = (kNeighbors(j)._2, j)
        }
        j = j + 1
      }
      if (distance < maxDistance._1) {
        kNeighbors(maxDistance._2) = (i, distance)
      }
      i = i + 1
    }

    kNeighbors.map(_._1)
  }

  /** Calculate minority class
    *
    * @return the elements in the minority class
    */
  def minority(data: Array[Any]): Array[Int] = {
    val classes: mutable.Map[Any, Int] = mutable.Map[Any, Int]()
    var i: Int = 0

    while (i < data.length) {
      if ((classes get data(i)).isEmpty) {
        classes.update(data(i), 1)
      } else {
        classes.update(data(i), classes(data(i)) + 1)
      }
      i = i + 1
    }

    val min = (x: (Any, Int), y: (Any, Int)) => {
      if (x._2 < y._2) {
        x
      } else {
        y
      }
    }

    val minorityClass: (Any, Int) = classes.reduceLeft(min)
    val minority: Array[Int] = new Array(minorityClass._2)
    i = 0
    var classIndex: Int = 0
    while (i < data.length) {
      if (data(i) == minorityClass._1) {
        minority(classIndex) = i
        classIndex = classIndex + 1
      }
      i = i + 1
    }
    minority
  }

  /** Compute the mode of an array
    *
    * @param data array to compute the mode
    * @return the mode of the array
    */
  def mode(data: Array[Any]): Any = {
    data.groupBy(identity).mapValues(_.length).toArray.maxBy(_._2)._1
  }

  /** Convert nanoseconds to minutes, seconds and milliseconds
    *
    * @param elapsedTime nanoseconds to be converted
    * @return String representing the conversion
    */
  def nanoTimeToString(elapsedTime: Long): String = {
    val minutes: Long = TimeUnit.NANOSECONDS.toMinutes(elapsedTime)
    val seconds: Long = TimeUnit.NANOSECONDS.toSeconds(elapsedTime) - TimeUnit.MINUTES.toSeconds(minutes)
    val millis: Long = TimeUnit.NANOSECONDS.toMillis(elapsedTime) - TimeUnit.MINUTES.toMillis(minutes) - TimeUnit.SECONDS.toMillis(seconds)
    "%03d min, %03d sec, %03d millis".format(minutes, seconds, millis)
  }

  /** Decide the label using the NNRule considering k neighbours of dataset
    *
    * @param neighbours neighbours of the element
    * @param instance   target instance
    * @param id         id of the instance
    * @param labels     labels associated to each point in data
    * @param k          number of neighbours to consider
    * @param which      if it's set to "nearest", return the nearest neighbours, if it sets "farthest", return the farthest ones
    * @return the label associated to newPoint and the index of the k-nearest which
    */
  def nnRule(neighbours: Array[Array[Double]], instance: Array[Double], id: Int, labels: Array[Any], k: Int,
             which: String): (Any, Array[Int], Array[Double]) = {
    val distances: Array[Double] = new Array[Double](neighbours.length)

    var i = 0
    while (i < neighbours.length) {
      distances(i) = euclidean(instance, neighbours(i))
      i += 1
    }
    if (id != -1)
      distances(id) = Double.MaxValue

    val kBest = List.fill(k)(0).toArray
    val distancesKBest = if (which == "nearest") List.fill(k)(Double.PositiveInfinity).toArray else List.fill(k)(Double.NegativeInfinity).toArray
    val compare: (Double, Double) => Boolean = if (which == "nearest") (a: Double, b: Double) => a < b else (a: Double, b: Double) => a > b

    var j = 0
    while (j < distances.length) {
      var worstDistance = distances(j)
      var worstIndex = 0

      var k = 0
      var assigned = false
      while (k < distancesKBest.length) {
        if (!compare(distancesKBest(k), worstDistance)) {
          worstDistance = distancesKBest(k)
          worstIndex = k
          assigned = true
        }
        k += 1
      }

      if (assigned) {
        kBest(worstIndex) = j
        distancesKBest(worstIndex) = distances(j)
      }

      j += 1
    }

    (mode(kBest map labels), kBest, distances)
  }

  /** Decide the label using the NNRule considering k neighbours of dataset
    *
    * @param neighbours         neighbours of the element
    * @param instance           target instance
    * @param id                 id of the instance
    * @param labels             labels associated to each point in data
    * @param k                  number of neighbours to consider
    * @param nominal            indicate nominal attributes in the instances
    * @param sds                standard deviations
    * @param attrCounter        counter attributes occurrences
    * @param attrClassesCounter number of occurrences for each value and output class c, for each class
    * @param which              "nearest" to return the nearest neighbours, otherwise, return the farthest ones
    * @return the label associated to newPoint and the index of the k-nearest which
    */
  def nnRuleHVDM(neighbours: Array[Array[Double]], instance: Array[Double], id: Int, labels: Array[Any], k: Int,
                 nominal: Array[Int], sds: Array[Double], attrCounter: Array[Map[Double, Int]],
                 attrClassesCounter: Array[Map[Double, Map[Any, Int]]], which: String): (Any, Array[Int], Array[Double]) = {
    val distances: Array[Double] = new Array[Double](neighbours.length)

    var i = 0
    while (i < neighbours.length) {
      distances(i) = HVDM(instance, neighbours(i), nominal, sds, attrCounter, attrClassesCounter)
      i += 1
    }
    if (id != -1)
      distances(id) = Double.MaxValue

    val kBest = List.fill(k)(0).toArray
    val distancesKBest = if (which == "nearest") List.fill(k)(Double.PositiveInfinity).toArray else List.fill(k)(Double.NegativeInfinity).toArray
    val compare: (Double, Double) => Boolean = if (which == "nearest") (a: Double, b: Double) => a < b else (a: Double, b: Double) => a > b

    var j = 0
    while (j < distances.length) {
      var worstDistance = distances(j)
      var worstIndex = 0

      var k = 0
      var assigned = false
      while (k < distancesKBest.length) {
        if (!compare(distancesKBest(k), worstDistance)) {
          worstDistance = distancesKBest(k)
          worstIndex = k
          assigned = true
        }
        k += 1
      }

      if (assigned) {
        kBest(worstIndex) = j
        distancesKBest(worstIndex) = distances(j)
      }

      j += 1
    }

    (mode(kBest map labels), kBest, distances)
  }

  /** Compute the number of occurrences for each value x for attribute represented by array attribute and output class c, for each class c in classes
    *
    * @param attribute attribute to be used
    * @param classes   classes present in the dataset
    * @return map of maps with the form: (value -> (class -> number of elements))
    */
  def occurrencesByValueAndClass(attribute: Array[Double], classes: Array[Any]): Map[Double, Map[Any, Int]] = {
    val auxMap: Map[Double, Array[Any]] = (attribute zip classes).groupBy((element: (Double, Any)) => element._1).map((element: (Double, Array[(Double, Any)])) =>
      (element._1, element._2.map((value: (Double, Any)) => value._2)))
    auxMap.map((element: (Double, Array[Any])) => Map(element._1 -> element._2.groupBy(identity).mapValues(_.length))).toList.flatten.toMap
  }

  /** Convert a data object into a matrix of doubles, taking care of missing values and nominal columns.
    * Missing data was treated using the most frequent value for nominal variables and the median for numeric variables.
    * Nominal columns are converted to doubles.
    *
    * @param data data to process
    */
  def processData(data: Data): (Array[Array[Double]], Array[mutable.Map[Double, Any]]) = {
    val nomToNum: Array[mutable.Map[Double, Any]] = Array.fill(data.x(0).length)(mutable.Map[Double, Any]())
    var nomToNumIndex = 0
    val processedData: Array[Array[Double]] = data.x.transpose.zipWithIndex.map { column: (Array[Any], Int) =>
      // let's look for the NA values
      val naIndex: Array[Int] = column._1.zipWithIndex.filter(_._1 == "soul_NA").map(_._2)
      // If they exist
      if (naIndex.length != 0) {
        // Take the index of the elements that are not NA
        val nonNAIndex: Array[Int] = column._1.zipWithIndex.filter(_._1 != "soul_NA").map(_._2)
        // If the column is not a nominal value
        if (!data.fileInfo.nominal.contains(column._2)) {
          // compute the mean of the present values
          val arrayDouble: Array[Double] = (nonNAIndex map column._1).map(_.asInstanceOf[Double])
          val mean: Double = arrayDouble.sum / arrayDouble.length
          val array: Array[Any] = column._1.clone()
          // replace all the NA values with the mean
          naIndex.foreach((index: Int) => array(index) = mean)

          array.map(_.asInstanceOf[Double])
        } else {
          // compute the mode of the present values
          val m: Any = mode(nonNAIndex map column._1)
          val array: Array[Any] = column._1.clone()
          // replace all the NA values with the mode
          naIndex.foreach((index: Int) => array(index) = m)

          // After replacing the NA values, we change them to numerical values (0, 1, 2, ..., N)
          val uniqueValues: Array[Any] = array.distinct
          var counter: Double = -1.0
          val dict: collection.Map[Any, Double] = uniqueValues.map { value: Any => counter += 1.0; value -> counter }.toMap
          array.indices.foreach((i: Int) => array(i) = dict(array(i)))
          // make the dictionary to convert numerical to nominal
          dict.foreach(pair => if (pair._1 != "soul_NA") {
            nomToNum(nomToNumIndex).update(pair._2, pair._1)
          })
          nomToNumIndex += 1
          array.map(_.asInstanceOf[Double])
        }
      } else {
        // If there is no NA values
        // If the column is not a nominal value
        if (data.fileInfo.nominal.contains(column._2)) {
          val array: Array[Any] = column._1.clone()
          // we change them to numerical values (0, 1, 2, ..., N)
          val uniqueValues: Array[Any] = array.distinct
          var counter: Double = -1.0
          val dict: collection.Map[Any, Double] = uniqueValues.map { value: Any => counter += 1.0; value -> counter }.toMap
          array.indices.foreach((i: Int) => array(i) = dict(array(i)))
          // make the dictionary to convert numerical to nominal
          dict.foreach(pair => nomToNum(nomToNumIndex).update(pair._2, pair._1))
          nomToNumIndex += 1
          array.map(_.asInstanceOf[Double])
        } else {
          // Store the data as is
          column._1.map(_.asInstanceOf[Double])
        }
      }
    }
    (processedData.transpose, nomToNum)
  }

  /** Compute the standard deviation for an array
    *
    * @param xs array to be used
    * @return standard deviation of x
    */
  def standardDeviation(xs: Array[Double]): Double = {
    val mean: Double = xs.sum / xs.length
    sqrt(xs.map((a: Double) => math.pow(a - mean, 2)).sum / xs.length)
  }

  /** Transform numerical data to nominal data
    *
    * @param data array with the results
    * @param dict dictionary with the keys to do the transformation
    * @return the result converted
    */
  def toNominal(data: Array[Array[Double]], dict: Array[mutable.Map[Double, Any]]): Array[Array[Any]] = {
    data.map(elem => {
      var index: Int = -1
      elem.map(atrib => {
        index += 1
        // if the value of the attribute is bigger than the max value in the dixtionary,
        // select the biggest in dict
        if (atrib.round < 0) {
          dict(index)(0)
        } else if (atrib.round <= dict(index).keys.max) {
          dict(index)(atrib.round)
        } else {
          dict(index)(dict(index).keys.max)
        }
      })
    })
  }

  /** Convert a double matrix to a matrix of Any
    *
    * @param d data to be converted
    * @return matrix of Any
    */
  def toXData(d: Array[Array[Double]]): Array[Array[Any]] = {
    d.map((row: Array[Double]) => row.map((_: Double).asInstanceOf[Any]))
  }

  /** Transform the numerical data to the nominal data
    *
    * @param data array with the results
    * @return the result converted
    */
  def to2Decimals(data: Array[Array[Double]]): Array[Array[Any]] = {
    data.map(_.map(BigDecimal(_).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble.asInstanceOf[Any]))
  }

  /** Denormalize the data
    *
    * @param data the data to denormalize
    * @param max  the max value of the samples for each column
    * @param min  the min value of the samples for each column
    * @return the data denormalized
    */
  def zeroOneDenormalization(data: Array[Array[Double]], max: Array[Double], min: Array[Double]): Array[Array[Double]] = {
    data.map(x => x.indices.map(index => x(index) * (max(index) - min(index)) + min(index)).toArray)
  }

  /** Normalize the data as follow: for each column, x, (x-min(x))/(max(x)-min(x))
    * This method only normalize not nominal columns
    *
    * @return normalized data
    */
  def zeroOneNormalization(d: Data, x: Array[Array[Double]]): Array[Array[Double]] = {
    val maxV: Array[Double] = x.transpose.map((col: Array[Double]) => col.max)
    val minV: Array[Double] = x.transpose.map((col: Array[Double]) => col.min)
    d.fileInfo.maxAttribs = maxV
    d.fileInfo.minAttribs = minV
    val result: Array[Array[Double]] = x.transpose.clone()

    x.transpose.indices.diff(d.fileInfo.nominal).par.foreach { index: Int =>
      val aux: Array[Double] = result(index).map((element: Double) => (element - minV(index)).toFloat / (maxV(index) - minV(index)))
      result(index) = if (aux.count(_.isNaN) == 0) aux else Array.fill[Double](aux.length)(0.0)
    }
    result.transpose
  }

  /** Return an array of the indices that are one
    *
    * @param data zero/one array to convert
    * @return indices
    */
  def zeroOneToIndex(data: Array[Int]): Array[Int] = {
    data.zipWithIndex.collect { case (v, i) if v == 1 => i }
  }
}