/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.optimization

import scala.math._

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

trait AdaUpdater extends Serializable {
  val epsilon = 1e-8
}
/**
 * :: DeveloperApi ::
 * Adam updater for gradient descent with L2 regularization.
 */
@DeveloperApi
class AdamUpdater extends AdaUpdater {
  private var beta1: Double = 0.9
  private var beta2: Double = 0.999

  def compute(
               weightsOld: Vector,
               gradient: Vector,
               mPre: Vector,
               vPre: Vector,
               beta1PowerOld: Double,
               beta2PowerOld: Double,
               iter: Int,
               regParam: Double): (Vector, Vector, Vector, Double, Double, Double) = {
    val decayRate = 1/sqrt(iter)
    val m = beta1 * mPre.asBreeze + (1-beta1) * gradient.asBreeze
    val v = beta2 * vPre.asBreeze + (1-beta2) * (gradient.asBreeze :* gradient.asBreeze)
    val beta1power = beta1PowerOld * beta1
    val beta2power = beta2PowerOld * beta2
    val mc = m / (1.0 - beta1power)
    val vc = v / (1.0 - beta2power)
    val sqv = vc.map(k => sqrt(k))
    val myWeight = weightsOld.asBreeze - decayRate * (mc :/ (sqv + epsilon))
    (Vectors.fromBreeze(myWeight), Vectors.fromBreeze(m), Vectors.fromBreeze(v),
      beta1power, beta2power, 0)
  }
}

/**
 * :: DeveloperApi ::
 * A simple AdaGrad updater for gradient descent *without* any regularization.
 * Uses a step-size decreasing with the square root of the number of iterations.
 * accum += grad * grad
 * var -= lr * grad * (1 / sqrt(accum))
 */
@DeveloperApi
class AdagradUpdater extends AdaUpdater {

  def compute(
               weightsOld: Vector,
               gradient: Vector,
               accumOld: Vector,
               learningRate: Double,
               iter: Int,
               regParam: Double): (Vector, Vector, Double) = {
    val accum = accumOld.asBreeze + (gradient.asBreeze :* gradient.asBreeze)
    val sqrtHistGrad = accum.map(k => sqrt(k + epsilon))
    val Weights = weightsOld.asBreeze - learningRate * (gradient.asBreeze :/ sqrtHistGrad)
    (Vectors.fromBreeze(Weights), Vectors.fromBreeze(accum), 0)
  }
}
