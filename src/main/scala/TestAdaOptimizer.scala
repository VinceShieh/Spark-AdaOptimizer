import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD



object TestAdaOptimizer extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTADAOPTIMIZER"))
    val training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)
    val testing = MLUtils.loadLibSVMFile(sc, "data/a9a.t")
    val lr = new LogisticRegressionWithAda().setIntercept(false)
    lr.optimizer
      .setStepSize(0.1)
      .setRegParam(0.0)
      .setNumIterations(1000)
      .setConvergenceTol(0.0005)

    val currentTime = System.currentTimeMillis()
    val model = lr.run(training)
    val elapsedTime = System.currentTimeMillis() - currentTime
    // Compute raw scores on the test set.
    val predictionAndLabels = testing.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    training.unpersist()
    println(s"Accuracy = $accuracy, time elapsed: $elapsedTime millisecond.")
    sc.stop()
  }
}
