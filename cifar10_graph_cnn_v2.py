# Peter Duggins, psipeter@gmail.com
# SYDE 552/750
# Final Project
# Winter 2016
# CNN Adapted from https://github.com/fchollet/keras/issues/762

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano

# Theano configuration
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
# import theano.sandbox.cuda
# theano.sandbox.cuda.use("gpu")

# Keras imports
from keras.datasets import cifar10
from keras.models import Graph
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

# Learning parameters
batch_size = 40
classes = 10
epochs = 50
learning_rate=0.01
decay=1e-6
momentum=0.9
nesterov=True

# Network Parameters
n_filters = [16, 24] # number of convolutional filters to use at layer_i
pool_size = [2, 2] # square size of pooling window at layer_i
kernel_size = [3, 3] # square size of kernel at layer_i
dropout_frac = [0.25, 0.25, 0.5] # dropout fraction at layer_i
dense_size = [512, classes] # output dimension for dense layers

# CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
shapex, shapey, shapez = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
image_dim=(shapez,shapex,shapey)
train_datapoints=samples_train
test_datapoints=samples_test

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Network
model = Graph()

#input
model.add_input(name='input', input_shape=image_dim)

#layer 1
model.add_node(Convolution2D(n_filters[0], kernel_size[0], kernel_size[0], 
				activation='relu', input_shape=image_dim),
				name='conv1a', input='input')
model.add_node(Convolution2D(n_filters[0], kernel_size[0], kernel_size[0], 
				activation='relu', border_mode='valid'),
				name='conv1b', input='conv1a')
model.add_node(MaxPooling2D(pool_size=(pool_size[0], pool_size[0])), 
				name='pool1', input='conv1b')
model.add_node(Dropout(dropout_frac[0]),
				name='drop1', input='pool1')

#layer 2
model.add_node(Convolution2D(n_filters[1], kernel_size[1], kernel_size[1],
				activation='relu', border_mode='valid'),
				name='conv2a', input='pool1')
model.add_node(Convolution2D(n_filters[1], kernel_size[1], kernel_size[1],
				activation='relu', border_mode='valid'),
				name='conv2b', input='conv2a')
model.add_node(MaxPooling2D(pool_size=(pool_size[1], pool_size[1])),
				name='pool2', input='conv2b')
model.add_node(Dropout(dropout_frac[1]),
				name='drop2', input='pool2')

#layer 3
model.add_node(Flatten(),
				name='flatten3', input='pool2')
model.add_node(Dense(dense_size[0],
				activation='relu', init='glorot_uniform'),
				name='dense3', input='flatten3')
model.add_node(Dropout(dropout_frac[2]),
				name='drop3', input='dense3')
model.add_node(Dense(dense_size[1],
				activation='softmax', init='glorot_uniform'),
				name='dense4', input='drop3')

#output
model.add_output(name='output', input='dense4', merge_mode='sum')


#optimize, compile, and print
sgd1 = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(sgd1, {'output':'categorical_crossentropy'})
# model.get_config(verbose=1)

#train
history=model.fit({'input':X_train[:train_datapoints], 'output':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input':X_test[:test_datapoints], 'output':Y_test[:test_datapoints]})

predictions = model.predict({'input':X_test[:test_datapoints]})

#save model arcitecture and weights
json_string = model.to_json()
open('cifar10_graph_cnn_v2.json', 'w').write(json_string)
model.save_weights('cifar10_graph_cnn_v2.h5')

#save layer stats
def get_outputs(model, input_name, layer_name, X_batch):
    get_outputs = theano.function([model.inputs[input_name].input],
    				model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    my_outputs = get_outputs(X_batch)
    return my_outputs

text_file = open("cifar10_graph_cnn_v2_stats.txt", "w")

print ("Node Name \t mean activation \t std activation \t min,max activation \t \
	 	||| across features: std(mean) \t std(std) \t std(min), std(max)\n\n")
text_file.write("Node Name \t mean activation \t std activation \t min,max activation \t \
	 	||| across features: std(mean) \t std(std) \t std(min), std(max)\n")

for the_node in model.nodes:
	A = get_outputs(model,'input',the_node,X_test[:test_datapoints])
	print (the_node,'\t')
	print (np.average(A),'\t',np.std(A),'\t'np.min(A),'\t'np.max(A),'\t')
	print ('|||')
	print (np.std(np.average(A,axis=1)),'\t')
	print (np.std(np.std(A,axis=1)),'\t')
	print (np.std(np.min(A,axis=1)),'\t')
	print (np.std(np.min(A,axis=1)),'\t')
	print (np.std(np.max(A,axis=1)),'\t')
	text_file.write(the_node,'\t')
	text_file.write(np.average(A),'\t',np.std(A),'\t',np.min(A),'\t',np.max(A),'\t')
	text_file.write('|||')
	text_file.write(np.std(np.average(A,axis=1)),'\t')
	text_file.write(np.std(np.std(A,axis=1)),'\t')
	text_file.write(np.std(np.min(A,axis=1)),'\t')
	text_file.write(np.std(np.min(A,axis=1)),'\t')
	text_file.write(np.std(np.max(A,axis=1)),'\t')
	# for i in range(len(A)):
	# 	print ("feature_map", i)
	# 	print ('mean', np.average(A[i]) ,'max', np.max(A[i]), 'std', np.std(A[i]))
text_file.close()
