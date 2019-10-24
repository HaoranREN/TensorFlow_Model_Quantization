# imports and load the MNIST data

import numpy as np
import tensorflow as tf

(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

train_data = (train_data.astype('float32') / 255.0).reshape(-1,28,28,1)
eval_data = (eval_data.astype('float32') / 255.0).reshape(-1,28,28,1)

# build tf.keras model

def build_keras_model():
    return tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), activation=tf.nn.relu, padding='same', input_shape=(28,28,1)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation=tf.nn.relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

# train the model as normal, without tf.enable_eager_execution()

train_batch_size = 50
train_epoch = 2

train_model = build_keras_model()

train_model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('\n------ Train ------\n')
train_model.fit(train_data, train_labels, batch_size = train_batch_size, epochs=train_epoch)

print('\n------ Test ------\n')
loss, acc = train_model.evaluate(eval_data, eval_labels)

train_model.save('/content/drive/My Drive/Colab Notebooks/quantization_github/post_training_integer_quantization_model/trained_model.h5')

'''
Run the code below after reseting the runtime.
The main goal here is to enable `tf.enable_eager_execution()` immediate after importing Tensorflow,
to generate the representative_dataset
'''

# imports and load the MNIST data

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

train_data = (train_data.astype('float32') / 255.0).reshape(-1,28,28,1)
eval_data = (eval_data.astype('float32') / 255.0).reshape(-1,28,28,1)

# convert to quantized tf.lite model

images = tf.cast(train_data, tf.float32)
mnist_ds = tf.data.Dataset.from_tensor_slices(images).batch(1)
def representative_data_gen():
    for input_value in mnist_ds.take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model_file('/content/drive/My Drive/Colab Notebooks/quantization_github/post_training_integer_quantization_model/trained_model.h5')

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)

converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open('/content/drive/My Drive/Colab Notebooks/quantization_github/post_training_integer_quantization_model/quantized_model.tflite', 'wb').write(tflite_model)

# load the quantized tf.lite model and test

interpreter = tf.lite.Interpreter(model_path='/content/drive/My Drive/Colab Notebooks/quantization_github/post_training_integer_quantization_model/quantized_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

acc = 0

eval_data = np.array(eval_data * 255 - 128, dtype = np.int8)

for i in range(eval_data.shape[0]):
    image = eval_data[i].reshape(1,28,28,1)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if (eval_labels[i]) == np.argmax(prediction):
        acc += 1

print('Post-training integer quantization accuracy: ' + str(acc / len(eval_data)))

'''
# check the tensor data type

tensor_details = interpreter.get_tensor_details()

for i in tensor_details:
    print(i['dtype'], i['name'], i['index'])
'''