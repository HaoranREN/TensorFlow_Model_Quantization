# Quantization-aware Training

###### Information below is version sensitive, time sensitive, and empirical, check the main [README.md](https://github.com/HaoranREN/TensorFlow_Model_Quantization) for details

Setting `tf.lite.TFLiteConverter.inference_type` to `tf.uint8` signals conversion to a fully quantized model, **only** from a quantization-aware trained input model. A quantization-aware trained model contains fake quantization nodes added by [`tf.contrib.quantize`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize). Since `tf.contrib.quantize` rewrites the training/eval graph, the `tf.lite.TFLiteConverter` should be contrusted by `tf.lite.TFLiteConverter.from_frozen_graph()`. This setup also requires `tf.lite.TFLiteConverter.quantized_input_stats` to be set. This parameter contains a **scalar** value and a **displacement** value of how to map the input data to values in the range of the inference date type (i.e. 0 - 255 for uint8), with the equation `real_input_value = (quantized_input_value - **displacement**) / **scalar**`. See the description of [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter) for more information. 

This conversion aims for a fully quantized model, but any operations that do not have quantized implementations will throw errors.

## Specifications

- **Inference Inputs:** `unit8` type, map the original data to the range of 0 - 255
- **Activations:** `unit8` type
- **Outputs:** `uint8` type
- **Inference Computation:** all in fixed-point type
- **`tf.lite.TFLiteConverter` Parameters:**
  - Construct by `tf.lite.TFLiteConverter.from_frozen_graph()`
  - `tf.lite.TFLiteConverter.inference_type = tf.uint8`
  - `tf.lite.TFLiteConverter.quantized_input_stats = {'input_layer_name': (displacement, scalar)}`

## Keynotes

Quantization-aware training uses the `tf.contrib` module, so it is somehow an 'experimental' feauture of Tensorflow. This Tensorflow team [webpage](https://www.tensorflow.org/lite/performance/model_optimization) says it is only available for a subset of convolutional neural network architectures. An unsupported architecture, is usually a tensor, which `tf.lite.TFLiteConverter` requires range information of it for the conversion, but `tf.contrib.quantize` dose not have the fake quantization implementation for it, so that there is no min/max value associate with that tensor. In this case, a error message of lacking min/max data will be thrown, like:

```
F tensorflow/lite/toco/tooling_util.cc:1728] Array batch_normalization/FusedBatchNormV3_mul_0, which is an input to the Add operator producing the output array batch_normalization/FusedBatchNormV3, is lacking min/max data, which is necessary for quantization. If accuracy matters, either target a non-quantized output format, or run quantized training with your model from a floating point checkpoint to change the input graph to contain min/max information. If you don't care about accuracy, you can pass --default_ranges_min= and --default_ranges_max= for easy experimentation.
Aborted (core dumped)
```

For some of the unsupported architectures, there are some 'tricks' I found based on my experience, to work around some common circumstances (listed below). In general, like prompted in the error message, the min/max lacking problem can be bypassed by setting the default range parameter `tf.lite.TFLiteConverter.default_ranges_stats`, but with an accuracy loss. To achieve better accuracy, my suggestion is, make sure the default range covers all the value inside all the min/max lacking tensors.

Also, not all Tensorflow operations are supported by `tf.lite`. The compatibility is listed in this [webpage](https://www.tensorflow.org/lite/guide/ops_compatibility). According to this page, operations may be elided or fused, before the supported operations are mapped to their TensorFlow Lite counterparts. This means the operation sequence or layer order matters. I did encounter some problems of supported operations being not suppported in some operation combinations.

The supportability issues are often version sensitive, even some online resources are very helpful, some can also be outdated. Thus, when facing such a supportability issue, my suggestion would be, take some experiments to do whatever can be done to modify the model, even with some minor behavior change, such as skip a layer or change layer order. Below are some 'tricks' that I found, which worked well with my experiments.

###### Batch Normalization Layer

For the best compatibility, when using batch normalization layer and convolution layer combinations, the batch normalization layer should come after a convolution layer. If it thorws an error message similar to the one above, of `FusedBatchNormV?` is lacking min/max data, try to use an unfused batch normalization layer. For example in `tf.keras`, use

```python
tf.keras.layers.BatchNormalization(fused=False)
```

The differences are, a fused batch normalization layer is kind of a wrapper layer of several batch normalization operations, and an unfused batch normalization layer leaves those operations individually. There is no fake quantization implementation for that wrapper layer, but for the individual operations.
