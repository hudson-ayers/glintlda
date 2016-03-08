package glintlda

import java.io.{PrintWriter, BufferedWriter, File}

import akka.util.Timeout
import breeze.io.TextWriter.FileWriter
import glint.Client
import glint.iterators.RowBlockIterator
import glint.models.client.granular.GranularBigMatrix
import glint.models.client.retry.{RetryBigMatrix, RetryBigVector}
import glint.models.client.{BigMatrix, BigVector}
import glint.partitioning.cyclic.CyclicPartitioner
import glintlda.util.{BoundedPriorityQueue, ranges}

import scala.concurrent.{Await, ExecutionContext}

/**
  * A distributed LDA Model
  *
  * @param wordTopicCounts A big matrix representing the topic word counts
  * @param topicCounts A big vector representing the global topic counts
  * @param config The LDA configuration
  */
class LDAModel(var wordTopicCounts: BigMatrix[Long],
               var topicCounts: BigVector[Long],
               val config: LDAConfig) extends Serializable {

  /**
    * Gets the top n words for each topic in the topic model
    *
    * @param words The number of words to return
    * @return Array of a list of words for each topic
    */
  def describe(words: Int)(implicit ec: ExecutionContext, timeout: Timeout): Array[List[(Long, Double)]] = {

    // Construct per-topic bounded priority queues
    val topWords = Array.fill(config.topics)(new BoundedPriorityQueue[(Double, Long)](words))
    val globalCounts = Await.result(topicCounts.pull((0L until config.topics).toArray), timeout.duration)

    // Iterate over blocks of rows and add entries to the bounded priority queue
    var start = 0L
    new RowBlockIterator[Long](wordTopicCounts, 10000).foreach {
      case rowBlock =>
        var i = 0
        while (i < rowBlock.length) {
          val row = rowBlock(i)
          var t = 0
          while (t < config.topics) {
            val p = (config.β + row(t).toDouble) / (config.vocabularyTerms * config.β + globalCounts(t).toDouble)
            topWords(t).enqueue((p, start + i))
            t += 1
          }
          i += 1
        }
        start += rowBlock.length
    }

    // Get final data from the bounded priority queue and convert it to the correct format
    topWords.map {
      case bpq => bpq.iterator.toList.sorted.reverse.map { case (prob, f) => (f, prob) }
    }
  }

  /**
    * Writes the topic model to given file
    *
    * @param file The file
    * @param ec The execution context in which to execute requests to the parameter server
    * @param timeout The timeout of operations to the parameter server
    */
  def writeToFile(file: File)(implicit ec: ExecutionContext, timeout: Timeout): Unit = {

    // Get global counts
    val globalCounts = Await.result(topicCounts.pull((0L until config.topics).toArray), timeout.duration)
    val writer = new PrintWriter(file)

    // Iterate over blocks of rows and add entries to the bounded priority queue
    var start = 0L
    new RowBlockIterator[Long](wordTopicCounts, 10000).foreach {
      case rowBlock =>
        var i = 0
        while (i < rowBlock.length) {
          writer.print(s"${start + i}")
          val row = rowBlock(i)
          var topic = 0
          while (topic < config.topics) {
            val probability = (config.β + row(topic).toDouble) / (config.vocabularyTerms * config.β + globalCounts(topic).toDouble)
            writer.print(s" $probability")
            topic += 1
          }
          writer.println()
          i += 1
        }
        start += rowBlock.length
    }

    // Close file
    writer.close()
  }

}

object LDAModel {

  /**
    * Constructs an empty LDA model based on given configuration
    *
    * @param config The LDA configuration
    */
  def apply(gc: Client, config: LDAConfig): LDAModel = {
    val topicWordCounts = gc.matrix[Long](config.vocabularyTerms, config.topics, 2, (x,y) => CyclicPartitioner(x, y))
    val granularTopicWordCounts = new GranularBigMatrix[Long](topicWordCounts, 120000)
    val globalCounts = new RetryBigVector[Long](gc.vector[Long](config.topics, 1), 5)
    new LDAModel(new RetryBigMatrix[Long](granularTopicWordCounts, 5), globalCounts, config)
  }

}

