# TensorFlow_Model_Quantization

The efficiency at inference time is critial when deploying machine learning models to devices with limited resources, such as IoT edge nodes and mobile devices. **Model quantization** is a tool to improve **inference efficiency**, by converting the variable data types inside a model (usually float32) into some data types with fewer numbers of bits (uint8, int8, float16, etc.), to overcome the constraints such as energy consumption, storage capacity, and computation power.

TensorFlow supports two levels of model quantizations in general (see [this link](https://www.tensorflow.org/lite/performance/model_optimization)):

- [Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize) (training with quantization)

**Post-training quantization** directly converts a trained model into a hybrid or fully-quantized `tf.lite` model using `tf.lite.converter`, with degradation in model accuracy.

**Quantization-aware training** trains a model that can be quantized by `tf.lite.converter` with minimal accuracy loss. It uses `tf.contrib.quantize` to rewrite the training/eval graph to add fake quantization nodes. Fake quantization nodes simulates the errors introduced by quantization during training, so that the errors can be calibrated in the following training process. It also generates tensor min/max values that are required by the `tf.lite.converter`.

# This Repository

When using the Tensorflow Python API methods above to implement model quantization, I encountered several problems with both post-training quantization and quantization-aware training. I also found online that some of the problems are very common. A lot of people were seeing the exact same errors as I was. After finally getting the codes run correctly, I decided to open this repository to post what I have learnt.

**This is a tutorial of model quantization using TensorFlow 1.14, with suggestions based on personal experience.** Since the Tensorflow team said they are working on a new package [`Tensorflow Model Optimiaztion`](https://www.tensorflow.org/model_optimization), which includes some new implementations of model quantization per their [roadmap](https://www.tensorflow.org/model_optimization/guide/roadmap), I will keep looking for their updates and merge them into this repository if possible. The last modification here was on **10/6/2019**, where in the roadmap, the Post Training Quantization for Hybrid Kernels and Post Training Quantization for (8b) Fixed-point Kernels are launched.

**You are welcome to comment any issues, concerns, and suggestions, as well as anything regarding to Tensorflow updates. If you found this repository to be usefull, I would like to thanks for your generosity to star this repository.**
