package glintlda

/**
  * Configuration for the LDA solver
  *
  * @param α Prior on the document-topic distributions
  * @param β Prior on the topic-word distributions
  * @param iterations The number of iterations
  * @param topics The number of topics
  * @param vocabularyTerms The number of vocabulary terms
  * @param partitions The number of partitions
  * @param blockSize The size of a block of coordinates to process at a time
  * @param seed The random seed
  */
case class LDAConfig(var α: Double = 0.5,
                     var β: Double = 0.01,
                     var τ: Int = 1,
                     var mhSteps: Int = 2,
                     var iterations: Int = 100,
                     var topics: Int = 10,
                     var vocabularyTerms: Int = 100000,
                     var powerlawCutoff: Int = 0,
                     var partitions: Int = 240,
                     var blockSize: Int = 1000,
                     var checkpointRead: String = "",
                     var checkpointSave: String = "",
                     var seed: Int = 42) {

  def setα(α: Double) = this.α = α
  def setβ(β: Double) = this.β = β
  def setτ(τ: Int) = this.τ = τ
  def setMhSteps(mhSteps: Int) = this.mhSteps = mhSteps
  def setIterations(iterations: Int) = this.iterations = iterations
  def setTopics(topics: Int) = this.topics = topics
  def setVocabularyTerms(vocabularyTerms: Int) = this.vocabularyTerms = vocabularyTerms
  def setPowerlawCutoff(powerlawCutoff: Int) = this.powerlawCutoff = powerlawCutoff
  def setPartitions(partitions: Int) = this.partitions = partitions
  def setBlockSize(blockSize: Int) = this.blockSize = blockSize
  def setCheckpointSave(checkpointSave: String) = this.checkpointSave = checkpointSave
  def setCheckpointRead(checkpointRead: String) = this.checkpointRead = checkpointRead
  def setSeed(seed: Int) = this.seed = seed

  override def toString: String = {
    s"""LDAConfig {
       |  α = $α
       |  β = $β
       |  τ = $τ
       |  mhSteps = $mhSteps
       |  iterations = $iterations
       |  topics = $topics
       |  vocabularyTerms = $vocabularyTerms
       |  powerlawCutoff = $powerlawCutoff
       |  partitions = $partitions
       |  blockSize = $blockSize
       |  checkpointSave = $checkpointSave
       |  checkpointRead = $checkpointRead
       |  seed = $seed
       |}
    """.stripMargin
  }
}
