# Glint LDA
Scalable Distributed LDA implementation for Spark & Glint

This implementation is based on [LightLDA](https://github.com/Microsoft/lightlda).

## Usage

Make sure you have Glint running, for a simple localhost test with 2 servers, you can locally run Glint as follows:

    sbt "run master"
    sbt "run server"
    sbt "run server"
    
Next, load in a dataset in Spark with an RDD:

    // Preprocessing of data ...
    // End result should be an RDD of breeze sparse vectors that represent bag-of-words term frequency vectors
    rdd = sc.textFile(...).map(x => SparseVector[Int](...))
    
Construct the Glint client that acts as an interface to the running parameter servers

    // Open glint client with a path to a specific configuration file
    val gc = Client(ConfigFactory.parseFile(new java.io.File(configFile)))
    
Set the LDA parameters and call the `fitMetropolisHastings` function to run the LDA algorithm

    // LDA topic model with 100,000 terms and 100 topics
    val ldaConfig = new LDAConfig()
    ldaConfig.setα(0.5)
    ldaConfig.setβ(0.01)
    ldaConfig.setTopics(100)
    ldaConfig.setVocabularyTerms(100000)
    val model = Solver.fitMetropolisHastings(sc, gc, rdd, ldaConfig, 100)
    
