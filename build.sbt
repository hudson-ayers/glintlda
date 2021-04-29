name := "GlintLDA"

version := "0.1-SNAPSHOT"

organization := "ch.ethz.inf.da"

scalaVersion := "2.11.8"


// Glint parameter server

libraryDependencies += "ch.ethz.inf.da" %% "glint" % "0.1-SNAPSHOT"


// Spark

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "1.6.0" % "provided"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "1.6.0" % "provided"


// Breeze native BLAS support

libraryDependencies += "org.scalanlp" %% "breeze" % "0.11.2"

libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.11.2"


// Unit tests

libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"

fork in Test := true

javaOptions in Test += "-Xmx2048m"

