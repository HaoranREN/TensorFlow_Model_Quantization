# TensorFlow_Model_Quantization
This is a tutorial of model quantization using TensorFlow with suggestions based on personal experience.

The efficiency at inference time is critial when deploying machine learning models to devices with limited resources, such IoT edge nodes and mobile devices. <b>Model quantization</b> is a tool to improve <b>inference efficiency</b>, by converting the variable data types inside a model (usually float32) to some data types with fewer numbers of bits (uint8, int8, float16, etc.).
