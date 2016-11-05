# Spark-AdaOptimizer
A Spark-based implementation of Adam optimizer, which is a method for Stochastic Optimization.
See https://arxiv.org/abs/1412.6980

Comparing with SGD, Adam has better performance, especially in case of sparse features, Adam can converge faster than
normal SGD.

## How to try
After getting Spark build environment ready,
run 'sbt package' under the root of the source to build the package
then excute '$SPARKHOME/bin/spark-submit
                --class TestAdaOptimizer
                --master local[1] target/scala-2.10/spark-adaoptimizer_2.10-0.0.1.jar'

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
