package glintlda

import akka.util.Timeout
import breeze.linalg.SparseVector
import com.typesafe.scalalogging.slf4j.Logger
import glint.Client
import glint.models.client.buffered.BufferedBigMatrix
import glintlda.mh.MHSolver
import glintlda.naive.NaiveSolver
import glintlda.util.{FastRNG, SimpleLock, RDDImprovements}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.concurrent.duration._
import scala.concurrent.{Future, ExecutionContext}

/**
  * A solver that can compute an LDA model based on data
  *
  * @param model The LDA model
  * @param id The identifier
  */
abstract class Solver(model: LDAModel, id: Int) {

  // Construct execution context, timeout and logger
  implicit protected val ec = ExecutionContext.Implicits.global
  implicit protected val timeout = new Timeout(600 seconds)
  protected val logger: Logger = Logger(LoggerFactory getLogger s"${getClass.getSimpleName}-$id")

  /**
    * Initializes the count table on the parameter servers specified in model with given partition of the data
    *
    * @param samples The samples to initialize with
    */
  private def initialize(samples: Array[GibbsSample]): Unit = {

    // Initialize buffered matrix for word topic counts
    val buffer = new BufferedBigMatrix[Long](model.wordTopicCounts, 100000)
    val topics = new Array[Long](model.config.topics)
    val pushLock = new SimpleLock(16, logger)
    var flushEntry = 0

    // Iterate over all samples and the corresponding features, counting them and pushing to the parameter servers
    logger.info(s"Constructing count table from samples")
    var i = 0
    while (i < samples.length) {
      val sample = samples(i)

      var j = 0
      while (j < sample.features.length) {

        if (buffer.isFull) {
          flushEntry += 1
          val flushEntryNow = flushEntry
          logger.info(s"Acquiring push lock for entry $flushEntryNow...")
          pushLock.acquire()
          logger.info(s"Flushing buffer $flushEntryNow")
          val flush = buffer.flush()
          flush.onComplete { case _ =>
            logger.info(s"Releasing push lock $flushEntryNow")
            pushLock.release()
          }
          flush.onFailure { case ex => logger.error(s"${ex.getMessage}\n${ex.getStackTraceString}") }

        }
        buffer.pushToBuffer(sample.features(j), sample.topics(j), 1)
        topics(sample.topics(j)) += 1

        j += 1
      }

      i += 1
    }

    // Perform final flush and await results to guarantee everything has been processed on the parameter servers
    logger.info(s"Flushing buffer to parameter server")
    pushLock.acquire()
    buffer.flush().onComplete(_ => pushLock.release())
    pushLock.acquire()
    model.topicCounts.push((0L until model.config.topics).toArray, topics).onComplete(_ => pushLock.release())

    // Wait for everything to finish
    logger.info(s"Waiting for all transfers to finish")
    pushLock.acquireAll()
    pushLock.releaseAll()

    // Print time information
    logger.info(s"Time spend waiting for lock: ${pushLock.waitTime}ms")

  }

  /**
    * Runs the LDA inference algorithm on given partition of the data
    *
    * @param samples The samples to run the algorithm on
    * @param iteration The iteration number
    */
  protected def fit(samples: Array[GibbsSample], iteration: Int): Unit

}

/**
  * The solver
  */
object Solver {

  /**
    * Trains an LDA model using a metropolis-hastings based algorithm
    *
    * @param gc The glint client
    * @param samples The samples
    * @param config The configuration
    * @return The trained LDA model
    */
  def fitMetropolisHastings(gc: Client, samples: RDD[SparseVector[Int]], config: LDAConfig): LDAModel = {
    fit(gc, samples, config, (model, id) => new MHSolver(model, id))
  }

  /**
    * Trains an LDA model using a naive algorithm
    *
    * @param gc The glint client
    * @param samples The samples
    * @param config The configuration
    * @return
    */
  def fitNaive(gc: Client, samples: RDD[SparseVector[Int]], config: LDAConfig): LDAModel = {
    fit(gc, samples, config, (model, id) => new NaiveSolver(model, id))
  }

  /**
    * Runs the solver
    *
    * @param gc The glint client
    * @param samples The samples as word-frequency vectors
    * @param config The LDA configuration
    * @param solver A function that creates a solver
    * @return A trained LDA model
    */
  def fit(gc: Client,
          samples: RDD[SparseVector[Int]],
          config: LDAConfig,
          solver: (LDAModel, Int) => Solver): LDAModel = {

    // Transform data to gibbs samples
    val gibbsSamples = transform(samples, config)

    // Execution context and timeouts for asynchronous operations
    implicit val ec = ExecutionContext.Implicits.global
    implicit val timeout = new Timeout(60 seconds)

    // Construct LDA model and initialize it on the parameter server
    val model = LDAModel(gc, config)
    gibbsSamples.foreachPartitionWithIndex { case (id, it) =>
      val s = solver(model, id)
      s.initialize(it.toArray)
    }

    // Construct evaluation
    val eval = new Evaluation(config)

    // Iterate
    var rdd = gibbsSamples
    var prevRdd = gibbsSamples
    for (t <- 0 until config.iterations) {

      // Perform training for this iteration
      rdd = rdd.mapPartitionsWithIndex { case (id, it) =>
        val s = solver(model, id)
        val partitionSamples = it.toArray
        s.fit(partitionSamples, t)
        partitionSamples.toIterator
      }.persist(StorageLevel.MEMORY_AND_DISK)

      // Compute evaluation
      val (docllh, counts) = rdd.aggregate[(Double, Long)]((0.0, 0L))(eval.aggregateDocument, eval.aggregateResults)
      Future {
        eval.logCurrentState(t, docllh, counts, model)
      }

      // Unpersist previous RDD
      prevRdd.unpersist()
      prevRdd = rdd

    }

    // Return trained model
    model
  }

  /**
    * Transforms given samples into (randomly initialized) Gibbs samples
    *
    * @param samples The samples as word-frequency vectors
    * @param config The LDA configuration
    * @return An RDD containing Gibbs samples
    */
  private def transform(samples: RDD[SparseVector[Int]], config: LDAConfig): RDD[GibbsSample] = {

    // Map partitions to Gibbs samples that have a random initialization
    val gibbsSamples = samples.mapPartitionsWithIndex { case (id, it) =>
      val random = new FastRNG(config.seed + id)
      it.map(s => GibbsSample(s, random, config.topics))
    }.repartition(config.partitions).persist(StorageLevel.MEMORY_AND_DISK)

    // Trigger empty action to materialize the mapping and persist it
    gibbsSamples.foreachPartition(_ => ())
    gibbsSamples
  }

}
