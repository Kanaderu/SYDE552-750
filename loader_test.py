# Peter Duggins, psipeter@gmail.com
# SYDE 552/750
# Final Project
# Winter 2016
# CNN Adapted from https://github.com/fchollet/keras/issues/762

# from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import csv

# Keras imports
from keras.datasets import mnist
from keras.models import Graph
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json

model = model_from_json(open('mnist_CNN_v2_test_model.json').read())
model.load_weights('mnist_CNN_v2_test_weights.h5')

for i in model.node_config():
	print (i)
