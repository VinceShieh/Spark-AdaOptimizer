# Spark-AdaOptimizer
A Spark-based implementation of Adam and AdaGrad optimizer, methods for Stochastic Optimization.
See https://arxiv.org/abs/1412.6980 and http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
Comparing with SGD, Adam and AdaGrad have better performance. Especially in case of sparse features, Adam can converge faster than
normal SGD.

## How to try
After getting Spark build environment ready,
run 'sbt package' under the root of the source to build the package
then excute '$SPARKHOME/bin/spark-submit
                --class TestAdaOptimizer
                --master local[*] target/scala-2.10/spark-adaoptimizer_2.10-0.0.1.jar'
##FAQ
"Some APIs are not found or parameters not match"

Please modify the settings in build.sbt to match your environments.

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
