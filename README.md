# TensorFlow_Model_Quantization

The efficiency at inference time is critial when deploying machine learning models to devices with limited resources, such IoT edge nodes and mobile devices. **Model quantization** is a tool to improve **inference efficiency**, by converting the variable data types inside a model (usually float32) into some data types with fewer numbers of bits (uint8, int8, float16, etc.), to overcome the constraints such as energy consumption, storage capacity, and computation power.

TensorFlow supports two levels of model quantizations in general (see [this link](https://www.tensorflow.org/lite/performance/model_optimization)):

- [Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)

**Post-training quantization** directly converts a trained model into a hybrid or fully-quantized `tf.lite` model using `tf.lite.converter`, with degradation in model accuracy.

**Quantization-aware training** trains a model that can be quantized with minimal accuracy loss. It uses `tf.contrib.quantize` to rewrite the training/eval graph to add fake quantization nodes. Fake quantization nodes simulates the errors introduced by quantization during training, so that the errors can be calibrated in the following training process. It also provides tensor min/max values that are required by the `tf.lite.converter`.

This is a tutorial of model quantization using TensorFlow with suggestions based on personal experience.
