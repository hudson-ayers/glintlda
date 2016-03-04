package glintlda.naive

import breeze.linalg.{DenseVector, Vector}
import breeze.stats.distributions.Multinomial
import glintlda.LDAConfig
import glintlda.util.FastRNG

/**
  * A naive sampler using the basic collapsed Gibbs sampling probabilities and draws from the unnormalized distribution
  * using a regular cumulative approach.
  *
  * @param config The LDA configuration
  * @param random The random number generator
  */
class Sampler(config: LDAConfig, random: FastRNG) {

  private val α = config.α
  private val β = config.β
  private val αSum = config.topics * α
  private val βSum = config.vocabularyTerms * config.β

  var infer: Int = 1
  var wordCounts: Vector[Long] = null
  var globalCounts: Array[Long] = null
  var documentCounts: Vector[Int] = null

  /**
    * Produces a new topic for given feature and old topic
    *
    * @param feature The feature
    * @param oldTopic The old topic
    * @return
    */
  def sampleFeature(feature: Int, oldTopic: Int): Int = {
    var i = 0
    val p = DenseVector.zeros[Double](config.topics)
    var sum = 0.0
    while (i < config.topics) {
      p(i) = (documentCounts(i) + α) * ((wordCounts(i) + β) / (globalCounts(i) + βSum))
      sum += p(i)
      i += 1
    }
    p /= sum
    Multinomial(p).draw()
  }

}
