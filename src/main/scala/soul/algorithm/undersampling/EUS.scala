package soul.algorithm.undersampling

import soul.data.Data
import soul.util.Utilities.Distance.Distance
import soul.util.Utilities._

import scala.collection.mutable.ArrayBuffer
import scala.math.{abs, sqrt}

/** Evolutionary Under Sampling. Original paper: "Evolutionary Under-Sampling for Classification with Imbalanced Data
  * Sets: Proposals and Taxonomy" by Salvador Garcia and Francisco Herrera.
  *
  * @param data           data to work with
  * @param seed           seed to use. If it is not provided, it will use the system time
  * @param populationSize number of chromosomes to generate
  * @param maxEvaluations number of evaluations
  * @param algorithm      version of core to execute. One of: EBUSGSGM, EBUSMSGM, EBUSGSAUC, EBUSMSAUC,
  *                       EUSCMGSGM, EUSCMMSGM, EUSCMGSAUC or EUSCMMSAUC
  * @param dist           object of Distance enumeration representing the distance to be used
  * @param probHUX        probability of changing a gen from 0 to 1 (used in crossover)
  * @param recombination  recombination threshold (used in reinitialization)
  * @param prob0to1       probability of changing a gen from 0 to 1 (used in reinitialization)
  * @param normalize      normalize the data or not
  * @param randomData     iterate through the data randomly or not
  * @param verbose        choose to display information about the execution or not
  * @author Néstor Rodríguez Vico
  */
class EUS(data: Data, seed: Long = System.currentTimeMillis(), populationSize: Int = 50, maxEvaluations: Int = 1000,
          algorithm: String = "EBUSMSGM", dist: Distance = Distance.EUCLIDEAN, probHUX: Double = 0.25,
          recombination: Double = 0.35, prob0to1: Double = 0.05, normalize: Boolean = false, randomData: Boolean = false,
          verbose: Boolean = false) {

  /** Compute the EUS algorithm.
    *
    * @return undersampled data structure
    */
  def compute(): Data = {
    val initTime: Long = System.nanoTime()

    val counter: Map[Any, Int] = data.y.groupBy(identity).mapValues(_.length)
    val untouchableClass: Any = counter.minBy((c: (Any, Int)) => c._2)._1
    val random: scala.util.Random = new scala.util.Random(seed)
    var dataToWorkWith: Array[Array[Double]] = if (normalize) zeroOneNormalization(data, data.processedData) else data.processedData
    val classesToWorkWith: Array[Any] = if (randomData) {
      val randomIndex: List[Int] = random.shuffle(data.y.indices.toList)
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

    val majoritySelection: Boolean = algorithm.contains("MS")
    val targetInstances: Array[Int] = classesToWorkWith.indices.toArray
    val minorityElements: Array[Int] = classesToWorkWith.zipWithIndex.collect { case (c, i) if c == untouchableClass => i }

    def fitnessFunction(instance: Array[Int]): Double = {
      val index: Array[Int] = zeroOneToIndex(instance) map targetInstances
      val neighbours: Array[Array[Double]] = index map dataToWorkWith
      val classes: Array[Any] = index map classesToWorkWith
      val predicted: Array[Any] = dataToWorkWith.indices.map { e: Int =>
        if (dist == Distance.EUCLIDEAN) {
          nnRule(neighbours, dataToWorkWith(e), e, classes, 1, "nearest")._1
        } else {
          nnRuleHVDM(neighbours, dataToWorkWith(e), e, classes, 1, data.fileInfo.nominal, sds, attrCounter, attrClassesCounter, "nearest")._1
        }
      }.toArray

      val matrix: (Int, Int, Int, Int) = confusionMatrix(originalLabels = index map classesToWorkWith,
        predictedLabels = predicted, minorityClass = untouchableClass)

      val tp: Int = matrix._1
      val fp: Int = matrix._2
      val fn: Int = matrix._3
      val tn: Int = matrix._4

      val nPositives: Int = (index map classesToWorkWith).count(_ == untouchableClass)
      val nNegatives: Int = (index map classesToWorkWith).length - nPositives

      val tpr: Double = tp / ((tp + fn) + 0.00000001)
      val fpr: Double = fp / ((fp + tn) + 0.00000001)
      val auc: Double = (1.0 + tpr - fpr) / 2.0
      val tnr: Double = tn / ((tn + fp) + 0.00000001)
      val g: Double = sqrt(tpr * tnr)

      val fitness: Double = if (algorithm == "EBUSGSGM") {
        g - abs(1 - (nPositives.toFloat / nNegatives)) * 20
      } else if (algorithm == "EBUSMSGM") {
        g - abs(1 - (counter(untouchableClass).toFloat / nNegatives)) * 20
      } else if (algorithm == "EUSCMGSGM") {
        g
      } else if (algorithm == "EUSCMMSGM") {
        g
      } else if (algorithm == "EBUSGSAUC") {
        auc - abs(1 - (nPositives.toFloat / nNegatives)) * 0.2
      } else if (algorithm == "EBUSMSAUC") {
        auc - abs(1 - (counter(untouchableClass).toFloat / nNegatives)) * 0.2
      } else if (algorithm == "EUSCMGSAUC") {
        auc
      } else if (algorithm == "EUSCMMSAUC") {
        auc
      } else {
        Double.NaN
      }

      if (fitness.isNaN)
        throw new Exception("Invalid argument: core should be: EBUSGSGM, EBUSMSGM, EBUSGSAUC, EBUSMSAUC, EUSCMGSGM, " +
          "EUSCMMSGM, EUSCMGSAUC or EUSCMMSAUC")

      fitness
    }

    val population: Array[Array[Int]] = new Array[Array[Int]](populationSize)
    (0 until populationSize).foreach { i: Int =>
      val individual: Array[Int] = targetInstances.indices.map(_ => random.nextInt(2)).toArray
      if (majoritySelection) {
        minorityElements.foreach((i: Int) => individual(i) = 1)
      }
      population(i) = individual
    }

    val evaluations: Array[Double] = new Array[Double](population.length)
    population.zipWithIndex.par.foreach { chromosome: (Array[Int], Int) =>
      evaluations(chromosome._2) = fitnessFunction(chromosome._1)
    }

    var incestThreshold: Int = targetInstances.length / 4
    var actualEvaluations: Int = populationSize

    while (actualEvaluations < maxEvaluations) {
      val randomPopulation: Array[Array[Int]] = random.shuffle(population.indices.toList).toArray map population
      val newPopulation: ArrayBuffer[Array[Int]] = new ArrayBuffer[Array[Int]](0)

      (randomPopulation.indices by 2).foreach { i: Int =>
        val hammingDistance: Int = (randomPopulation(i) zip randomPopulation(i + 1)).count((pair: (Int, Int)) => pair._1 != pair._2)

        if ((hammingDistance / 2) > incestThreshold) {
          val desc1: Array[Int] = randomPopulation(i).clone
          val desc2: Array[Int] = randomPopulation(i + 1).clone

          desc1.indices.foreach { i: Int =>
            if (desc1(i) != desc2(i) && random.nextFloat < 0.5) {
              desc1(i) = if (desc1(i) == 1) 0 else if (random.nextFloat < probHUX) 1 else desc1(i)
              desc2(i) = if (desc2(i) == 1) 0 else if (random.nextFloat < probHUX) 1 else desc2(i)

              if (majoritySelection) {
                minorityElements.foreach((i: Int) => desc1(i) = 1)
                minorityElements.foreach((i: Int) => desc2(i) = 1)
              }
            }
          }

          newPopulation += desc1
          newPopulation += desc2
        }
      }

      val newEvaluations: Array[Double] = new Array[Double](newPopulation.length)
      newPopulation.zipWithIndex.par.foreach { chromosome: (Array[Int], Int) =>
        newEvaluations(chromosome._2) = fitnessFunction(chromosome._1)
      }

      actualEvaluations += newPopulation.length

      // We order the population. The best ones (greater evaluation value) are the first
      val populationOrder: Array[(Double, Int, String)] = evaluations.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).map((e: (Double, Int)) => (e._1, e._2, "OLD"))
      val newPopulationOrder: Array[(Double, Int, String)] = newEvaluations.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).map((e: (Double, Int)) => (e._1, e._2, "NEW"))

      if (newPopulationOrder.length == 0 || populationOrder.last._1 > newPopulationOrder.head._1) {
        incestThreshold -= 1
      } else {
        val finalOrder: Array[(Double, Int, String)] = (populationOrder ++ newPopulationOrder).sortBy(_._1)(Ordering[Double].reverse).take(populationSize)

        finalOrder.zipWithIndex.foreach { e: ((Double, Int, String), Int) =>
          population(e._2) = if (e._1._3 == "OLD") population(e._1._2) else newPopulation(e._1._2)
          evaluations(e._2) = if (e._1._3 == "OLD") evaluations(e._1._2) else newEvaluations(e._1._2)
        }
      }

      if (incestThreshold <= 0) {
        population.indices.tail.foreach { i: Int =>
          val individual: Array[Int] = population(i).map(_ => if (random.nextFloat < recombination)
            if (random.nextFloat < prob0to1) 1 else 0 else population(0)(i))

          if (majoritySelection) {
            minorityElements.foreach((i: Int) => individual(i) = 1)
          }

          population(i) = individual
        }

        population.zipWithIndex.tail.par.foreach { e: (Array[Int], Int) =>
          evaluations(e._2) = fitnessFunction(e._1)
        }

        actualEvaluations += (population.length - 1)

        incestThreshold = (recombination * (1.0 - recombination) * targetInstances.length.toFloat).toInt
      }
    }

    val bestChromosome: Array[Int] = population(evaluations.zipWithIndex.sortBy(_._1)(Ordering[Double].reverse).head._2)
    val finalIndex: Array[Int] = zeroOneToIndex(bestChromosome) map targetInstances
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
