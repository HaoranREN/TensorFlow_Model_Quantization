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
        tf.keras.layers.BatchNormalization(fused=False),

        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.BatchNormalization(fused=False),

        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.BatchNormalization(fused=False),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(64, activation=tf.nn.relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

# train the model, quantization-aware training (finetuning) after $[quant_delay] steps

train_batch_size = 50
train_batch_number = train_data.shape[0]
quant_delay_epoch = 1

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

tf.keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=int(train_batch_number / train_batch_size * quant_delay_epoch))

    train_sess.run(tf.global_variables_initializer())	 

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print('\n------ Train ------\n')
    train_model.fit(train_data, train_labels, batch_size = train_batch_size, epochs=quant_delay_epoch * 2)

    print('\n------ Test ------\n')
    loss, acc = train_model.evaluate(eval_data, eval_labels)

    saver = tf.train.Saver()
    saver.save(train_sess, 'path_to_checkpoints')

# save the frozen graph

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

tf.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
	tf.keras.backend.set_learning_phase(0)
	eval_model = build_keras_model()
	tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
	eval_graph_def = eval_graph.as_graph_def()
	saver = tf.train.Saver()
	saver.restore(eval_sess, 'path_to_checkpoints')
    
	frozen_graph_def = tf.graph_util.convert_variables_to_constants(
		eval_sess,
		eval_graph_def,
		[eval_model.output.op.name]
	)

	with open('path_to_frozen_graph.pb', 'wb') as f:
		f.write(frozen_graph_def.SerializeToString())

# convert to quantized tf.lite model

input_max = np.max(train_data)
input_min = np.min(train_data)
converter_std = 255 / (input_max - input_min)
converter_mean = -(input_min * converter_std)

converter = tf.lite.TFLiteConverter.from_frozen_graph('path_to_frozen_graph.pb',
                                                     ['conv2d_input'],
                                                     ['dense_1/Softmax'])
converter.inference_type = tf.uint8
converter.quantized_input_stats = {'conv2d_input':(converter_mean, converter_std)}
#converter.default_ranges_stats = (0,1)
tflite_model = converter.convert()
open('path_to_quantized_model.tflite', 'wb').write(tflite_model)

# load the quantized tf.lite model and test

interpreter = tf.lite.Interpreter(model_path='path_to_quantized_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

quantize_eval_data = np.array(eval_data * 255, dtype = np.uint8)
acc = 0

for i in range(quantize_eval_data.shape[0]):
	quantize_image = quantize_eval_data[i]
	quantize_image = quantize_image.reshape(1,28,28,1)

	interpreter.set_tensor(input_details[0]['index'], quantize_image)
	interpreter.invoke()
	prediction = interpreter.get_tensor(output_details[0]['index'])

	if (eval_labels[i]) == np.argmax(prediction):
		acc += 1

print('Quantization-aware training (finetuning) accuracy: ' + str(acc / len(eval_data)))

'''
# check the tensor data type

tensor_details = interpreter.get_tensor_details()

for i in tensor_details:
    print(i['dtype'], i['name'], i['index'])
'''
