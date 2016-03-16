# Peter Duggins, psipeter@gmail.com
# SYDE 552/750
# Final Project
# Winter 2016

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np
import theano

# Learning parameters
batch_size = 32
classes = 10
epochs = 2
train_datapoints=50
test_datapoints=10
learning_rate=0.01
decay=1e-6
momentum=0.9
nesterov=True

# Network Parameters
n_filters = [32, 64] # number of convolutional filters to use at layer_i
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

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Network
model = Graph()
model.add_input(name='input', input_shape=image_dim)

model.add_node(Convolution2D(n_filters[0],kernel_size[0],kernel_size[0],
				activation='relu',input_shape=image_dim),
				name='f_1',input='input')
model.add_node(MaxPooling2D(pool_size=(1,1),#feedforward identity
				trainable=False), 
				name='v_1', input='f_1')

model.add_node(Flatten(),
				name='flat_1', input='v_1')
model.add_node(Dense(dense_size[0],
				activation='relu'),
				name='dense_1', input='flat_1')
model.add_node(Dropout(dropout_frac[2]),
				name='drop_1', input='dense_1')
model.add_node(Dense(dense_size[1],
				activation='softmax'),
				name='dense_out', input='drop_1')

model.add_output(name='output',input='dense_out')

# Optimize and compile
my_sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(my_sgd, {'output':'categorical_crossentropy'})

# Train
history=model.fit({'input':X_train[:train_datapoints], 'output':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input':X_test[:test_datapoints], 'output':Y_test[:test_datapoints]})

# Print results and network configuration
print (history.history)
# model.get_config(verbose=1)
def get_activations(model, input_name, layer_name, X_batch):
    get_activations = theano.function([model.inputs[input_name].input], model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

f_1_output=get_activations(model,'input','f_1',X_test[:test_datapoints])
v_1_output=get_activations(model,'input','v_1',X_test[:test_datapoints])
print (f_1_output[1])
print (v_1_output[1])