# [Post-training Integer Quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)

###### Information below is version sensitive, time sensitive, and empirical, check the main [README.md](https://github.com/HaoranREN/TensorFlow_Model_Quantization) for details
###### See this [Google Colab ipynb](https://colab.research.google.com/drive/12tUYhjb8MbczoSgj2kjH5V2UYHrr7780) for sample output
###### Sample code is available [here](post_training_integer_quantization.py)

Setting `tf.lite.TFLiteConverter.optimizations = [tf.lite.Optimize.DEFAULT]` indicates the `tf.lite.TFLiteConverter` to perform a post-training integer quantization. By doing so, `tf.lite.TFLiteConverter.representative_dataset` requires a generator function that provides some sample data for calibration. The behavior of the `tf.lite.TFLiteConverter` can be specified with parameter settings, see Inference Specifications below for details.

To quantize an tensor, the main task is to calculate the two parameters `scalar` and `displacement` for value range mapping, by solving the equation set:
- float_min = (fixed_min - **displacement**) / **scalar**
- float_max = (fixed_max - **displacement**) / **scalar**

The fixed_min/max are known from the target data type nature, but float_min/max are still needed to solve for `scalar` and `displacement`.
After a regular training, the trained model only contains values in those weight tensors, i.e. only has min/max data for those tensors, so it can only be post-training weight quantized. However, by providing the `tf.lite.TFLiteConverter.representative_dataset`, sample data can flow through the entire model as it does during the train/eval processes, so that each of the other tensors can record a sample value set. With the min/max data provided by these sample sets, the model can be fully quantized. Similarly, the fake quantization nodes track the min/max data during a quantization-aware training.

## Inference Specifications

- **Inputs:** defined by parameter settings, map to the range of the target data type
- **Activations:** same as the data type in each of the operations
- **Outputs:** defined by parameter settings
- **Computation:** defined by parameter settings, check `tf.lite.TFLiteConverter.target_ops` below. If input or output data type does not match the target operation data type, an operation will be added after input or before output to cast data type and map data range.
- **`tf.lite.TFLiteConverter` Parameters:**
  - `tf.lite.TFLiteConverter.optimizations = [tf.lite.Optimize.DEFAULT]`
  - `tf.lite.TFLiteConverter.representative_dataset` required for calibration
  - `tf.lite.TFLiteConverter.inference_input_type` and `tf.lite.TFLiteConverter.inference_output_type` set to target data type, default to `tf.float32`, supports `tf.float32, tf.uint8, tf.int8`
  - `tf.lite.TFLiteConverter.target_ops` default to `[tf.lite.OpsSet.TFLITE_BUILTINS]`, supports `SELECT_TF_OPS, TFLITE_BUILTINS, TFLITE_BUILTINS_INT8`
    - `[tf.lite.OpsSet.TFLITE_BUILTINS]`: supported operations are quantized, others remain in float-point
    - `[tf.lite.OpsSet.TFLITE_BUILTINS_INT]`: aim for an `int8` fully quantized model, operations cannot be quantized throw errors
    - `[tf.lite.OpsSet.SELECT_TF_OPS]`: to avoid the limitation of operations are partially supported by TensorFlow Lite (not recommended)
    
## Keynotes

To make the `tf.lite.TFLiteConverter.representative_dataset` working, `tf.enable_eager_execution()` must be called immediate after importing Tensorflow. However, I found that there might be some unexpected behaviors during a regular training and evaluation process with eager execution enabled. The `tf.lite.Interpreter` seems working good. Although there should be some workarounds, I would suggest to reset the Python runtime both before the regular train/eval part and `tf.lite.TFLiteConverter` part, and be clear to enable eager execution or not.
