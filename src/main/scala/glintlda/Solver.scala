package glintlda

import java.util.concurrent.atomic.AtomicBoolean

import akka.util.Timeout
import breeze.linalg.SparseVector
import com.typesafe.scalalogging.slf4j.Logger
import glint.Client
import glint.models.client.buffered.BufferedBigMatrix
import glintlda.mh.MHSolver
import glintlda.naive.NaiveSolver
import glintlda.util.{AggregateBuffer, FastRNG, SimpleLock, RDDImprovements}
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.spark.{SparkException, Success, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.scheduler._
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.concurrent.duration._
import scala.concurrent.{Await, Future, ExecutionContext}

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
    val aggregateBuffer = new AggregateBuffer(model.config.powerlawCutoff, model.config)
    val topics = new Array[Long](model.config.topics)
    val pushLock = new SimpleLock(16, logger)

    // Iterate over all samples and the corresponding features, counting them and pushing to the parameter servers
    logger.info(s"Constructing count table from samples")
    var i = 0
    while (i < samples.length) {
      val sample = samples(i)

      var j = 0
      while (j < sample.features.length) {

        if (buffer.isFull) {
          pushLock.acquire()
          val flush = buffer.flush()
          flush.onComplete { case _ =>
            pushLock.release()
          }
          flush.onFailure { case ex => logger.error(s"${ex.getMessage}\n${ex.getStackTraceString}") }

        }
        if (sample.features(j) < aggregateBuffer.cutoff) {
          aggregateBuffer.add(sample.features(j), sample.topics(j), 1)
        } else {
          buffer.pushToBuffer(sample.features(j), sample.topics(j), 1)
        }
        topics(sample.topics(j)) += 1

        j += 1
      }

      i += 1
    }

    // Flush power law buffer
    pushLock.acquire()
    aggregateBuffer.flush(model.wordTopicCounts).onComplete(_ => pushLock.release())

    // Perform final flush and await results to guarantee everything has been processed on the parameter servers
    pushLock.acquire()
    buffer.flush().onComplete(_ => pushLock.release())
    pushLock.acquire()
    model.topicCounts.push((0L until model.config.topics).toArray, topics).onComplete(_ => pushLock.release())

    // Wait for everything to finish
    logger.info(s"Waiting for all transfers to finish")
    pushLock.acquireAll()
    pushLock.releaseAll()

    // Print time information
    logger.info(s"Total time spend waiting for lock: ${pushLock.waitTime}ms")

  }

  /**
    * Runs the LDA inference algorithm on given partition of the data
    *
    * @param samples The samples to run the algorithm on
    * @param iteration The iteration number
    */
  protected def fit(samples: Array[GibbsSample], iteration: Int): Unit

  /**
    * Runs the LDA inference algorithm on given partition of the data without
    * updating topic-word counts
    *
    * @param samples The samples to run the algorithm on
    */
  protected def test(samples: Array[GibbsSample]): Unit

}

/**
  * The solver
  */
object Solver {

  private val logger: Logger = Logger(LoggerFactory getLogger s"${getClass.getSimpleName}")
  private val datasetStorageLevel = StorageLevel.DISK_ONLY

  def test(sc: SparkContext, gc: Client, model: LDAModel, samples: RDD[SparseVector[Int]], iterations: Int): Unit = {

    // Execution context and timeouts for asynchronous operations
    implicit val ec = ExecutionContext.Implicits.global
    implicit val timeout = new Timeout(60 seconds)

    // Construct evaluation
    val eval = new Evaluation(model.config)

    // Evaluate
    val gibbsSamples = transform(samples, model.config)
    var rdd = gibbsSamples
    var prevRdd = rdd
    var prevFuture = Future { }
    var t = 1
    while (t <= iterations) {

      rdd = rdd.mapPartitionsWithIndex { case (id, it) =>
        val s = new MHSolver(model, id).asInstanceOf[Solver]
        val partitionSamples = it.toArray
        s.test(partitionSamples)
        partitionSamples.toIterator
      }.persist(datasetStorageLevel)

      // Compute document-specific evaluations
      val (documentLogLikelihood, tokenCounts) = try {
        rdd.aggregate[(Double, Long)]((0.0, 0L))(eval.aggregateDocument, eval.aggregateResults)
      } catch {
        case e: Exception => (0.0, 0L)
      }

      // Build count table for rdd
      val countModel = build(gc, model.config, rdd, (m, i) => new MHSolver(m, i))

      // Compute log likelihood and perplexity
      prevFuture = Future {
        val iteration = t
        eval.logCurrentState(iteration, documentLogLikelihood, tokenCounts, countModel)

        // Free up space by destroying the count model
        countModel.wordTopicCounts.destroy()
        countModel.topicCounts.destroy()
      }

      // Unpersist previous RDD
      prevRdd.unpersist()
      prevRdd = rdd

      // Go to next iteration
      t += 1
    }

    // Wait for final evaluation to finish
    Await.result(prevFuture, 300 seconds)
  }

  /**
    * Trains an LDA model using a metropolis-hastings based algorithm
    *
    * @param sc The spark context
    * @param gc The glint client
    * @param samples The samples
    * @param config The configuration
    * @return The trained LDA model
    */
  def fitMetropolisHastings(sc: SparkContext, gc: Client, samples: RDD[SparseVector[Int]], config: LDAConfig): LDAModel = {
    fit(sc, gc, samples, config, (model, id) => new MHSolver(model, id))
  }

  /**
    * Trains an LDA model using a naive algorithm
    *
    * @param sc The spark context
    * @param gc The glint client
    * @param samples The samples
    * @param config The configuration
    * @return
    */
  def fitNaive(sc: SparkContext, gc: Client, samples: RDD[SparseVector[Int]], config: LDAConfig): LDAModel = {
    fit(sc, gc, samples, config, (model, id) => new NaiveSolver(model, id))
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
  def fit(sc: SparkContext,
          gc: Client,
          samples: RDD[SparseVector[Int]],
          config: LDAConfig,
          solver: (LDAModel, Int) => Solver): LDAModel = {

    // Transform data to gibbs samples
    val gibbsSamples: RDD[GibbsSample] = if (config.checkpointRead.isEmpty) {
      transform(samples, config)
    } else {
      sc.objectFile[GibbsSample](config.checkpointRead)
    }

    // Set checkpoint directory
    if (!config.checkpointSave.isEmpty) {
      sc.setCheckpointDir(config.checkpointSave)
    }

    // Execution context and timeouts for asynchronous operations
    implicit val ec = ExecutionContext.Implicits.global
    implicit val timeout = new Timeout(60 seconds)

    // Construct LDA model and initialize it on the parameter server
    var model = build(gc, config, gibbsSamples, solver)

    // Construct evaluation
    val eval = new Evaluation(config)

    // When tasks fail we have to reconstruct the count table due to mismatch of local data and parameter server storage
    // Check for task failure with a spark listener and set a boolean to true if we have to reconstruct the count table
    val rebuildCountTable: AtomicBoolean = new AtomicBoolean(false)
    sc.addSparkListener(new SparkListener() {
      override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
        taskEnd.reason match {
          case Success =>
          case _ => rebuildCountTable.set(true)
        }
      }
    })

    // Iterate
    var rdd = gibbsSamples
    var prevRdd = gibbsSamples
    var prevFuture: Future[Unit] = Future {}
    var lastCheckpointIteration: Int = 0
    var t = 1
    while (t <= config.iterations) {

      logger.info(s"Starting iteration $t")

      // Perform training for this iteration
      rdd = rdd.mapPartitionsWithIndex { case (id, it) =>
        val s = solver(model, id)
        val partitionSamples = it.toArray
        s.fit(partitionSamples, t)
        partitionSamples.toIterator
      }.persist(datasetStorageLevel)

      // Perform checkpointing
      if (!config.checkpointSave.isEmpty) {
        if (t % config.checkpointEvery == 0) {
          rdd.checkpoint()
        }
      }

      // Compute evaluation
      val (documentLogLikelihood, tokenCounts) = try {
        rdd.aggregate[(Double, Long)]((0.0, 0L))(eval.aggregateDocument, eval.aggregateResults)
      } catch {
        case e: Exception =>
          rebuildCountTable.set(true)
          (0.0, 0L)
      }

      if (rebuildCountTable.get()) {

        // Something went wrong, remove the current RDD and reset it to the previous iteration's RDD
        logger.warn(s"Iteration $t failed: rebuilding count table from samples and restarting iteration")
        removeRdd(rdd, sc)
        rdd = prevRdd

        // Rebuild count table after evaluation to restore valid state on the parameter servers
        Await.result(prevFuture, Duration.Inf)
        model.wordTopicCounts.destroy()
        model.topicCounts.destroy()
        model = build(gc, config, rdd, solver)
        rebuildCountTable.set(false)
        t = lastCheckpointIteration + 1

      } else {

        // Checkpoint was successfully computed, store the iteration number
        if (!config.checkpointSave.isEmpty) {
          if (t % config.checkpointEvery == 0) {
            lastCheckpointIteration = t
          }
        }

        // Nothing went wrong, compute evaluation as normal and continue
        prevFuture = Future {
          val iteration = t
          eval.logCurrentState(iteration, documentLogLikelihood, tokenCounts, model)
        }

        // Unpersist previous RDD and delete old checkpointed data
        removeRdd(prevRdd, sc)
        prevRdd = rdd

        // Go to next iteration
        t += 1

      }

    }

    // Wait for evaluation to finish
    Await.result(prevFuture, Duration.Inf)

    // Return trained model
    model
  }

  /**
    * Removes the old RDD and associated checkpoint data
    *
    * @param oldRdd The old RDD
    * @param sc The spark context (needed for deleting checkpoint data)
    */
  private def removeRdd(oldRdd: RDD[GibbsSample], sc: SparkContext): Unit = {
    if (oldRdd.isCheckpointed) {
      try {
        oldRdd.getCheckpointFile.foreach {
          case s => FileSystem.get(sc.hadoopConfiguration).delete(new Path(s), true)
        }
      } catch {
        case e: Exception => logger.error(s"Checkpoint deletion error: ${e.getMessage}\n${e.getStackTraceString}")
      }
    }
    oldRdd.unpersist()
  }

  /**
    * Rebuilds the count table in given model from given set of samples
    *
    * @param samples The samples
    */
  private def build(gc: Client, config: LDAConfig, samples: RDD[GibbsSample], solver: (LDAModel, Int) => Solver): LDAModel = {
    val model = LDAModel(gc, config)
    samples.foreachPartitionWithIndex { case (id, it) =>
      val s = solver(model, id)
      s.initialize(it.toArray)
    }
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
    }

    // Repartition (if possible through coalesce for performance reasons)
    val repartitionedGibbsSamples = if (gibbsSamples.getNumPartitions > config.partitions) {
      gibbsSamples.coalesce(config.partitions)
    } else {
      gibbsSamples.repartition(config.partitions)
    }

    // Persist samples to memory and disk
    val persistedGibbsSamples = repartitionedGibbsSamples.persist(datasetStorageLevel)

    // Trigger empty action to materialize the mapping and persist it
    persistedGibbsSamples.foreachPartition(_ => ())
    persistedGibbsSamples
  }

}
