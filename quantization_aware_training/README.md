# [Quantization-aware Training](https://github.com/tensorflow/tensorflow/tree/r1.14/tensorflow/contrib/quantize)

###### Information below is version sensitive, time sensitive, and empirical, check the main [README.md](https://github.com/HaoranREN/TensorFlow_Model_Quantization) for details
###### See [quantization_aware_training.md](quantization_aware_training.md) for some code-side comments
###### See this [Google Colab ipynb](https://colab.research.google.com/drive/1hD_G2qD3ptlH9zrpT4GtDCD0GwXjt7K-) for sample output
###### Sample code is avaiable [here](quantization_aware_training.py)

Setting `tf.lite.TFLiteConverter.inference_type` to `tf.uint8` signals conversion to a fully quantized model, **only** from a quantization-aware trained input model. A quantization-aware trained model contains fake quantization nodes added by [`tf.contrib.quantize`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize). Since `tf.contrib.quantize` rewrites the training/eval graph, the `tf.lite.TFLiteConverter` should be constructed by `tf.lite.TFLiteConverter.from_frozen_graph()`. This setup also requires `tf.lite.TFLiteConverter.quantized_input_stats` to be set. This parameter contains a **scalar** value and a **displacement** value of how to map the input data to values in the range of the inference datatype (i.e. 0 - 255 for `uint8`), with the equation `real_input_value = (quantized_input_value - **displacement**) / **scalar**`. See the description of [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter) for more information. 

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

Quantization-aware training uses the `tf.contrib` module, so it is somehow an 'experimental' feature of Tensorflow. This Tensorflow team [webpage](https://www.tensorflow.org/lite/performance/model_optimization) says it is only available for a subset of convolutional neural network architectures. An unsupported architecture, is usually a tensor, which `tf.lite.TFLiteConverter` requires range information of it for the conversion, but `tf.contrib.quantize` dose not have the fake quantization implementation for it, so that there is no min/max value associate with that tensor. In this case, a error message of lacking min/max data will be thrown, like:

```
F tensorflow/lite/toco/tooling_util.cc:1728] Array batch_normalization/FusedBatchNormV3_mul_0, which is an input to the Add operator producing the output array batch_normalization/FusedBatchNormV3, is lacking min/max data, which is necessary for quantization. If accuracy matters, either target a non-quantized output format, or run quantized training with your model from a floating point checkpoint to change the input graph to contain min/max information. If you don't care about accuracy, you can pass --default_ranges_min= and --default_ranges_max= for easy experimentation.
Aborted (core dumped)
```

For some of the unsupported architectures, there are some 'tricks' I found based on my experience, to work around some common circumstances (listed below). In general, like prompted in the error message, the min/max lacking problem can be bypassed by setting the default range parameter `tf.lite.TFLiteConverter.default_ranges_stats`, but with an accuracy loss. To achieve better accuracy, my suggestion is, make sure the default range covers all the values inside all the min/max lacking tensors.

Also, not all Tensorflow operations are supported by `tf.lite`. The compatibility is listed in this [webpage](https://www.tensorflow.org/lite/guide/ops_compatibility). According to this page, operations may be elided or fused, before the supported operations are mapped to their TensorFlow Lite counterparts. This means the operation sequence or layer order matters. I did encounter some problems of supported operations being not supported in some operation combinations.

The supportability issues are often version sensitive, even some online resources are very helpful, some can also be outdated. Thus, when facing such a supportability issue, my suggestion would be, take some experiments to do whatever can be done to modify the model, even with some minor behavior changes if accuracy is acceptable, such as skip a layer or change layer order. Below are some 'tricks' that I found, which worked well with my experiments.

Some ideas/code are retrieved from these [discussions](https://github.com/tensorflow/tensorflow/issues/27880).

###### Batch Normalization Layers

For the best compatibility, when using batch normalization layer and convolution layer combinations, the batch normalization layer should come after a convolution layer. If it throws an error message similar to the one above, of `FusedBatchNormV?` is lacking min/max data, try to use an unfused batch normalization layer. For example in `tf.keras`, use

```python
tf.keras.layers.BatchNormalization(fused=False)
```

The differences are, a fused batch normalization layer is kind of a wrapper layer of several batch normalization operations, and an unfused batch normalization layer leaves those operations individually. There is no fake quantization implementation for that wrapper layer, but for the individual operations.

| Fused | Unfused|
| --- | --- |
| ![Fused](/other/fused.png) | ![Unfused](/other/unfused.png) |

###### Activation Layers

The best practice of an activation layer is combining it with the preceding layer, since some stand alone activation layers are not supported. For example, the code below is part of ResNetV1. The activation layer on the last line throws a min/max lacking error. I tried skipping it as well as combining the activations into the two layers in line 6 and line 9. Both gave acceptable accuracies.

```python
1   for res_block in range(num_res_blocks):
2       strides = 1
3       if stack > 0 and res_block == 0:  # first layer but not first stack
4           strides = 2
5       y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
6       y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
7
8       if stack > 0 and res_block == 0:  # first layer but not first stack
9           x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
10                           strides=strides, activation=None, batch_normalization=False)
11      x = tf.keras.layers.add([x, y])
12      x = tf.keras.layers.Activation('relu')(x)
```

Some activation functions are not supported. For example, I also tried using `softplus` in the two layers, but with no success.

###### Convolution Layer Bias

Mostly, bias tensors are supported. However, in some models, for example an `uint8` target model, if the bias are greater than 255, which cannot be represented by `uint8` datatype, bias would be converted to `int32` type. Even it is still good for fixed-point only inference computation, if something unexpected happens with bias, try to set layer parameter `use_bias = False`.
