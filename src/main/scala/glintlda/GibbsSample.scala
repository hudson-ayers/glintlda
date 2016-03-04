package glintlda

import breeze.linalg.{DenseVector, SparseVector, sum}
import glintlda.util.FastRNG

/**
  * A Gibbs sample
  * @param features The (sequential and ordered) features
  * @param topics The assigned topics
  */
class GibbsSample(val features: Array[Int],
                  val topics: Array[Int]) extends Serializable {

  /**
    * Returns a dense topic count
    * @param nrOfTopics The total number of topics
    * @return The dense topic count
    */
  def denseCounts(nrOfTopics: Int): DenseVector[Int] = {
    val result = DenseVector.zeros[Int](nrOfTopics)
    var i = 0
    while (i < topics.length) {
      result(topics(i)) += 1
      i += 1
    }
    result
  }

  /**
    * Returns a sparse topic count
    *
    * @param nrOfTopics The total number of topics
    * @return The sparse topic count
    */
  def sparseCounts(nrOfTopics: Int): SparseVector[Int] = {
    val result = SparseVector.zeros[Int](nrOfTopics)
    var i = 0
    while (i < topics.length) {
      result.update(topics(i), result(topics(i)) + 1)
      i += 1
    }
    result
  }

}

object GibbsSample {

  /**
    * Initializes a Gibbs sample with random (uniform) topic assignments
    *
    * @param sv The sparse vector representing the document
    * @param random The random number generator
    * @param topics The number of topics
    * @return An initialized Gibbs sample with random (uniform) topic assignments
    */
  def apply(sv: SparseVector[Int], random: FastRNG, topics: Int): GibbsSample = {
    val totalTokens = sum(sv)
    val sample = new GibbsSample(new Array[Int](totalTokens), new Array[Int](totalTokens))

    var i = 0
    var current = 0
    while (i < sv.activeSize) {
      val index = sv.indexAt(i)
      var value = sv.valueAt(i)
      while (value > 0) {
        sample.features(current) = index
        sample.topics(current) = random.nextPositiveInt() % topics
        current += 1
        value -= 1
      }
      i += 1
    }

    sample
  }

}
