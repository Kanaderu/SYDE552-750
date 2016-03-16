#courtesy https://github.com/fchollet/keras/issues/762

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

# Learning parameters
batch_size = 32
nb_classes = 10
nb_epoch = 2
data_augmentation = True
train_datapoints=500
test_datapoints=100

# Network parameters
nb_filters = [32, 64] # number of convolutional filters to use at each layer
nb_pool = [2, 2] # level of pooling to perform at each layer
nb_conv = [3, 3] # level of convolution to perform at each layer
nb_dropout = [0.25, 0.25, 0.5] # dropout fraction
nb_dense = [512, 10] # output dimension for dense layers

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# shape of the image (SHAPE x SHAPE)
shapex, shapey, shapez = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
dimensions=(shapez,shapex,shapey)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

'''Network'''
model = Graph()

#input
model.add_input(name='input1', input_shape=dimensions)

#layer 1
model.add_node(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0], 
				activation='relu', border_mode='valid', input_shape=dimensions),
				name='conv2', input='input1')
model.add_node(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0], 
				activation='relu', border_mode='valid'),
				name='conv3', input='conv2')
model.add_node(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[0])), 
				name='pool1', input='conv3')
model.add_node(Dropout(nb_dropout[0]),
				name='drop1', input='pool1')

#layer 2
model.add_node(Convolution2D(nb_filters[1], nb_conv[1], nb_conv[1],
				activation='relu', border_mode='valid'),
				name='conv4', input='pool1')
model.add_node(Convolution2D(nb_filters[1], nb_conv[1], nb_conv[1],
				activation='relu', border_mode='valid'),
				name='conv5', input='conv4')
model.add_node(MaxPooling2D(pool_size=(nb_pool[1], nb_pool[1])),
				name='pool2', input='conv5')
model.add_node(Dropout(nb_dropout[1]),
				name='drop2', input='pool2')

#layer 3
model.add_node(Flatten(),
				name='flatten3', input='pool2')
# unknown = nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) * (shapey / nb_pool[0] / nb_pool[1])
model.add_node(Dense(nb_dense[0],
				activation='relu', init='glorot_uniform'),
				name='dense3', input='flatten3')
model.add_node(Dropout(nb_dropout[2]),
				name='drop3', input='dense3')
model.add_node(Dense(nb_dense[1],
				activation='softmax', init='glorot_uniform'),
				name='dense4', input='drop3')

#output
model.add_output(name='output1', input='dense4', merge_mode='sum')


#optimize, compile, and print
sgd1 = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(sgd1, {'output1':'categorical_crossentropy'})
# model.get_config(verbose=1)

#train
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
history=model.fit({'input1':X_train[:train_datapoints], 'output1':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True,
            validation_data={'input1':X_test[:test_datapoints], 'output1':Y_test[:test_datapoints]})
print (history.history)

#model.predict({'input1':X_test})