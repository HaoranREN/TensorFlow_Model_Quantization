# This Repository

When using the Tensorflow Python API methods above to implement model quantization, I encountered several problems with both post-training quantization and quantization-aware training. I also found online that some of the problems are very common. A lot of people were seeing the exact same errors as I was. After finally getting the codes run correctly, I decided to open this repository to post what I have learnt.

**This is a tutorial of model quantization using TensorFlow, with suggestions based on personal experience.** The Tensorflow version here in examples is **1.14**, and models are built with **`tf.keras`**. Since the Tensorflow team said they are working on a new package [`Tensorflow Model Optimiaztion`](https://www.tensorflow.org/model_optimization), which includes some new implementations of model quantization per their [roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap), I will keep looking for their updates and merge them into this repository if possible. The last modification here was on **10/6/2019**, where in the roadmap, the [Post Training Quantization for Hybrid Kernels](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3) (Post-training weight quantization) and [Post Training Quantization for (8b) Fixed-point Kernels](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba) (Post-training integer quantization) are launched.

**You are welcome to comment any issues, concerns, and suggestions, as well as anything regarding to Tensorflow updates. If you found this repository to be useful, I would like to thank you for your generosity to star** :star2: **this repository.**

# TensorFlow Model Quantization

The efficiency at inference time is critial when deploying machine learning models to devices with limited resources, such as IoT edge nodes and mobile devices. **Model quantization** is a tool to improve **inference efficiency**, by converting the variable data types inside a model (usually float32) into some data types with fewer numbers of bits (uint8, int8, float16, etc.), to overcome the constraints such as energy consumption, storage capacity, and computation power.

TensorFlow supports two levels of model quantizations in general (see [this link](https://www.tensorflow.org/lite/performance/model_optimization)):

- [Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
  - [Post-training weight quantization (hybrid)](https://www.tensorflow.org/lite/performance/post_training_quant)
  - [Post-training integer quantization (full)](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

- [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize) (training with quantization)

In general, quantization in Tensforflow uses [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter) to convert a float-point model to a [`tf.lite`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite) model. A `tf.lite` model contains some/only fixed-point values. Some parameters of `tf.lite.TFLiteConverter` can be tuned to indicate the expected quantization method.

**Post-training quantization** directly converts a trained model into a hybrid or fully-quantized `tf.lite` model using `tf.lite.TFLiteConverter`, with degradation in model accuracy. Post-training weight quantization only quantize model weights (convolution layer kernels, dense layer weights, etc.) to reduce model size and speedup computations by allowing hybrid operations (mix of fixed- and floating-point math). Post-training integer quantization fully quantize the model to support fixed-point-only hardware accelerators.

**Quantization-aware training** trains a model that can be quantized by `tf.lite.TFLiteConverter` with minimal accuracy loss. It uses [`tf.contrib.quantize`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize) to rewrite the training/eval graph to add fake quantization nodes. Fake quantization nodes simulate the errors introduced by quantization during training, so that the errors can be calibrated in the following training process. They also contain min/max values that are required by the `tf.lite.TFLiteConverter`.

Comparing with quantization-aware training, post-training quantization is simpler to use, and it only requires an already-trained floating-point mode. Based on the [roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap) release above, while quantization-aware training is still expected for some models that accuracy is strict required, the Tensorflow team is expecting it to be rare as they improve post-training quantization tools to a negligible accuracy loss.

Quoting from Tensorflow:

> In summary, a user should use “hybrid” post training quantization when targeting simple CPU size and latency improvements. When targeting greater CPU improvements or fixed-point accelerators, they should use this integer post training quantization tool, potentially using quantization-aware training if accuracy of a model suffers.

**Please go each directory for details about the three model quantization tools.**
