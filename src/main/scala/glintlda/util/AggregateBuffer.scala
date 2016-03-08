package glintlda.util

import akka.util.Timeout
import glint.models.client.BigMatrix
import glintlda.LDAConfig

import scala.concurrent.{ExecutionContext, Future}

/**
  * This is a specialized buffer that keeps aggregating values locally
  * We wish to use this for words that are most common according to the power law distribution
  *
  * @param cutoff The cutoff feature. Keep this relatively small as we need cutoff * topics space to store the buffer
  * @param config The LDA configuration
  */
class AggregateBuffer(val cutoff: Int, config: LDAConfig) {

  val size: Int = cutoff * config.topics
  val buffer = new Array[Long](size)

  /**
    * Adds a value to the buffer
    *
    * @param feature The feature
    * @param topic The topic
    * @param value The value
    */
  @inline
  def add(feature: Long, topic: Int, value: Int): Unit = {
    buffer(feature.toInt * config.topics + topic) += value
  }

  /**
    * Flushes to given lda model
    *
    * @param matrix The matrix to flush to
    * @param ec The execution context in which the push requests will be executed
    * @param timeout The timeout for the push requests
    * @return
    */
  def flush(matrix: BigMatrix[Long])(implicit ec: ExecutionContext, timeout: Timeout): Future[Boolean] = {

    if (cutoff > 0) {
      val rows = new Array[Long](size)
      val cols = new Array[Int](size)
      var i = 0
      while (i < size) {
        rows(i) = i / config.topics
        cols(i) = i % config.topics
        i += 1
      }
      matrix.push(rows, cols, buffer)
    } else {
      Future { true }
    }

  }

}
