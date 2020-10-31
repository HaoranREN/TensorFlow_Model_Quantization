# This Repository

When using the Tensorflow Python API methods above to implement model quantization, I encountered several problems with both post-training quantization and quantization-aware training. I also found online that some of the problems are very common. A lot of people were seeing the exact same errors as I was. After finally getting the codes run correctly, I decided to open this repository to post what I have learnt.

**This is a tutorial of model quantization using TensorFlow, with suggestions based on personal experience.** The Tensorflow version here in examples is **1.14**, and models are built with **`tf.keras`**. Since the Tensorflow team said they are working on a new package [`Tensorflow Model Optimiaztion`](https://www.tensorflow.org/model_optimization), which includes some new implementations of model quantization per their [roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap), I will keep looking for their updates and merge them into this repository if possible.

The last modification here was on **10/30/2020**, where in the roadmap, the [Post training quantization for dynamic-range kernels](https://blog.tensorflow.org/2018/09/introducing-model-optimization-toolkit.html) (Post-training weight quantization), [Post training quantization for (8b) fixed-point kernels](https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html) (Post-training integer quantization), and [Quantization aware training for (8b) fixed-point kernels and experimentation for <8b](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html) are launched.

**You are welcome to comment any issues, concerns, and suggestions, as well as anything regarding to Tensorflow updates. If you found this repository to be useful, I would like to thank you for your generosity to star** :star2: **this repository.**

## Update 10/30/2020

There is another quantization tool [Qkeras](https://github.com/google/qkeras) launched by a team from Google. It has a Keras like interface, and supports both common CNN and RNN. This is a good tool for advanced quantization research, but not compatible with `tf.lite` for deployment.

## Update 6/8/2020

Post Training Quantization for Hybrid Kernels now has a new official name: Post training quantization for dynamic-range kernels.

The Tensorflow Model Optimiaztion package now contains a new tool to perform **quantization-aware training**, and here is the [guide](https://www.tensorflow.org/model_optimization/guide/quantization/training_example). By default, this new tool produces a quantization-aware trained model with hybrid kernels, where only weights are stored in fixed-point value. The bias are stroed in float and the model takes float input. Based on my experiences, it only supports a subset of the layers/operations. For example, batch normalization layer(fused & unfused) is not supported. However, I think (did not try) someone can write a customized layer that perform a simplified BN operation and intergrate it with the model based on this [comprehensive guide](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide), which also includes some experimental features of the new tool.

To train a fixed-point model, consider the [quantization-aware training method here in this repo](/quantization_aware_training/README.md). It is still working with **Tensorflow 1.14**, but will not work with a new version, due to the removal of `tf.contrib` and some changes on `tf.lite.TFLiteConverter`.

## TensorFlow Model Quantization (modified on 10/24/2019)

The efficiency at inference time is critial when deploying machine learning models to devices with limited resources, such as IoT edge nodes and mobile devices. **Model quantization** is a tool to improve **inference efficiency**, by converting the variable data types inside a model (usually float32) into some data types with fewer numbers of bits (uint8, int8, float16, etc.), to overcome the constraints such as energy consumption, storage capacity, and computation power.

TensorFlow supports two levels of model quantizations in general (see [this link](https://www.tensorflow.org/lite/performance/model_optimization)):

- [Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
  - [Post-training weight quantization (hybrid)](https://www.tensorflow.org/lite/performance/post_training_quant)
  - [Post-training integer quantization (full)](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

- [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize) (training with quantization)

In general, quantization in Tensforflow uses [`tf.lite.TFLiteConverter`](https://github.com/tensorflow/docs/blob/r1.14/site/en/api_docs/python/tf/lite/TFLiteConverter.md) to convert a float-point model to a [`tf.lite`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite) model. A `tf.lite` model contains some/only fixed-point values. Some parameters of `tf.lite.TFLiteConverter` can be tuned to indicate the expected quantization method.

**Post-training quantization** directly converts a trained model into a hybrid or fully-quantized `tf.lite` model using `tf.lite.TFLiteConverter`, with degradation in model accuracy. Post-training weight quantization only quantize model weights (convolution layer kernels, dense layer weights, etc.) to reduce model size and speedup computations by allowing hybrid operations (mix of fixed- and floating-point math). Post-training integer quantization fully quantize the model to support fixed-point-only hardware accelerators.

**Quantization-aware training** trains a model that can be fully quantized by `tf.lite.TFLiteConverter` with minimal accuracy loss. It uses [`tf.contrib.quantize`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize) to rewrite the training/eval graph to add fake quantization nodes. Fake quantization nodes simulate the errors introduced by quantization during training, so that the errors can be calibrated in the following training process. They also contain min/max values that are required by the `tf.lite.TFLiteConverter`. (Why min/max matter? See [here](/post_training_integer_quantization/README.md).)

Comparing with quantization-aware training, post-training quantization is simpler to use, and it only requires an already-trained floating-point mode. Based on the [roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap) release above, while quantization-aware training is still expected for some models that accuracy is strict required, the Tensorflow team is expecting it to be rare as they improve post-training quantization tools to a negligible accuracy loss.

Quoting from Tensorflow:

> In summary, a user should use “hybrid” post training quantization when targeting simple CPU size and latency improvements. When targeting greater CPU improvements or fixed-point accelerators, they should use this integer post training quantization tool, potentially using quantization-aware training if accuracy of a model suffers.

**Please go each directory for details about the three model quantization tools.**
