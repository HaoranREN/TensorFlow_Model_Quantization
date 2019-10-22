# Quantization-aware Training

Setting `tf.lite.TFLiteConverter.inference_type` to `tf.uint8` signals conversion to a fully quantized model, **only** from a quantization-aware trained input model. A quantization-aware trained model contains fake quantization nodes added by `tf.contrib.quantize`. Since `tf.contrib.quantize` rewrites the training/eval graph, the `tf.lite.TFLiteConverter` should be contrusted by `tf.lite.TFLiteConverter.from_frozen_graph()`. This setup also requires `tf.lite.TFLiteConverter.quantized_input_stats` to be set. This parameter contains a **scalar** value and a **displacement** value of how to map the input data to values in the range of the inference date type (i.e. 0 - 255 for uint8), with the equation `real_input_value = (quantized_input_value - **displacement**) / **scalar**`. See the description of `tf.lite.TFLiteConverter` for more information. 

This conversion aims for a fully quantized model, but any operations that do not have quantized implementations will throw errors.

## Specifications

**Inference Inputs:** `unit8` type, mapped to the range of 0 - 255

**Inference Computation:** all in fixed-point type

**`tf.lite.TFLiteConverter` Parameters:**
- Construct by `tf.lite.TFLiteConverter.from_frozen_graph()`
- `tf.lite.TFLiteConverter.inference_type = tf.uint8`
- `tf.lite.TFLiteConverter.quantized_input_stats = {'input_layer_name': (displacement, scalar)}`

## Tensorflow usage
- [`tf.contrib.quantize`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/quantize)
- [`tf.lite.TFLiteConverter`](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter)

## Keynotes
