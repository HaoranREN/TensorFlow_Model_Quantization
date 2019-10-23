# Post-training Weight Quantization

###### Information below is version sensitive, time sensitive, and empirical, check the main [README.md](https://github.com/HaoranREN/TensorFlow_Model_Quantization) for details
###### See this [Google Colab ipynb](https://colab.research.google.com/drive/119GkmswoaO4GZV5rQ5W9q8W2BlPeedYr) for sample output
###### Sample code is avaiable [here](post_training_weight_quantization.py)

Setting `tf.lite.TFLiteConverter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]` indicates the `tf.lite.TFLiteConverter` to perform a post-training weight quantization. In the resulting `tf.lite` model, only weights of supported operations are quantized, such as convolution layer kernels and dense layer weights. However, the first layer and the last layer of the model will not be touched since they are very sensitive for accuracy.

At inference time, the input should be in float-point type and should be normalized to the same range as the training dataset. The inference computaion is in a hybrid manner. For hybrid-supported operations with quantized weights, the input tensor will be quantized and perform fixed-point computation, and convert the fixed-point output tensor back to float-point values. For all the other operations, it performs as a normal float-point model.

## Specifications

- **Inference Inputs:** float-point type, in the same range as the training dataset
- **Activations:** float-point type
- **Outputs:** float-point type
- **Inference Computation:** hybrid, fixed-point for hybrid operations, float-point for others
- **`tf.lite.TFLiteConverter` Parameters:** `tf.lite.TFLiteConverter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]`
