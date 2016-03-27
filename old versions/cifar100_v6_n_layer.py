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
import csv

# Theano configuration
# theano.config.floatX = 'float32'
# theano.config.device = 'gpu'
# import theano.sandbox.cuda
# theano.sandbox.cuda.use("gpu")
# ssh ctnuser@ctngpu2.uwaterloo.ca -p 3656
# pswd neuro...with replacements eio

# Keras imports
from keras.datasets import cifar100 
from keras.models import Graph
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

# CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
shapex, shapey, shapez = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
image_dim=(shapez,shapex,shapey)

# Parameters
filename='cifar100_v6_n_layer'
batch_size = 32
classes = 100
epochs = 20
learning_rate=0.01
decay=1e-6
momentum=0.9
nesterov=True
train_datapoints=5000
test_datapoints=3000
n_conv_layers=5
n_dense_layers=3
n_filters = [8,16,32,16,8] # number of feature maps at layer_i, len(n_conv_layers)
pool_size = [4,4,4,4,4] # square size of pooling window at layer_i, len(n_conv_layers)
kernel_size = [7,7,5,5,3] # square size of kernel at layer_i, len(n_conv_layers)
dropout_frac = [.5,.5,.5,.5,.5,.5,.5,.5]# dropout fraction at layer_i, len(n_conv_layers+n_dense_layers)
dense_size = [512,512,512] # output dimension for dense layers, len(n_dense_layers)
params={ #dictionary
	'batch_size' : batch_size,
	'classes' : classes,
	'epochs' : epochs,
	'learning_rate' : learning_rate,
	'decay' : decay,
	'momentum' : momentum,
	'nesterov' : nesterov,
	'n_filters' : n_filters,
	'pool_size' : pool_size,
	'kernel_size' : kernel_size,
	'dropout_frac' : dropout_frac,
	'dense_size' : dense_size,
	'train_datapoints' : train_datapoints, #samples_train
	'test_datapoints' : train_datapoints, #samples_test
	'filename' : filename,
	'n_conv_layers' : n_conv_layers,
	'n_dense_layers' : n_dense_layers,
}

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255






'''Network
# In this version, average and max pooling are dead ends at each layer;
# they only calculate statistics'''
model = Graph()

#input and 1st conv layer
model.add_input(name='input', input_shape=image_dim)
model.add_node(Convolution2D(n_filters[0], kernel_size[0], kernel_size[0], 
				activation='relu', input_shape=image_dim),
				name='conv%s' %0, input='input')
model.add_node(AveragePooling2D(pool_size=(pool_size[0], pool_size[0])), 
				name='avgpool%s' %0, input='conv%s' %0)
model.add_node(MaxPooling2D(pool_size=(pool_size[0], pool_size[0])),
				name='maxpool%s' %0, input='conv%s' %0)
model.add_node(Dropout(dropout_frac[0]),
				name='drop%s' %0, input='conv%s' %0)

#conv layers
for i in np.arange(1,n_conv_layers):

	model.add_node(Convolution2D(n_filters[i], kernel_size[i], kernel_size[i], 
					activation='relu', input_shape=image_dim),
					name='conv%s' %i, input='conv%s' %(i-1))
	model.add_node(AveragePooling2D(pool_size=(pool_size[i], pool_size[i])), 
					name='avgpool%s' %i, input='conv%s' %i)
	model.add_node(MaxPooling2D(pool_size=(pool_size[i], pool_size[i])),
					name='maxpool%s' %i, input='conv%s' %i)
	model.add_node(Dropout(dropout_frac[i]),
					name='drop%s' %i, input='conv%s' %i)

#1st dense layer
model.add_node(Flatten(),
				name='flatten', input='conv%s' %(n_conv_layers-1))
model.add_node(Dense(dense_size[0],
				activation='relu', init='glorot_uniform'),
				name='dense%s' %0, input='flatten')

#dense layers
for j in np.arange(1,n_dense_layers):
	model.add_node(Dense(dense_size[j],
					activation='relu', init='glorot_uniform'),
					name='dense%s' %j, input='dense%s' %(j-1))
	model.add_node(Dropout(dropout_frac[j]),
					name='drop%s' %(i+j), input='dense%s' %j)

#classifier layer and output
model.add_node(Dense(classes,
				activation='softmax', init='glorot_uniform'),
				name='dense_output', input='drop%s' %(n_dense_layers+n_conv_layers-2))
model.add_output(name='output', input='dense_output', merge_mode='sum')







'''optimize, compile, and print'''
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(sgd, {'output':'categorical_crossentropy'})

#train
history=model.fit({'input':X_train[:train_datapoints], 'output':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input':X_test[:test_datapoints], 'output':Y_test[:test_datapoints]})
predictions = model.predict({'input':X_test[:test_datapoints]})







'''output'''
#save layer statistics
def get_outputs(model, input_name, layer_name, X_batch):
    get_outputs = theano.function([model.inputs[input_name].input],
    				model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    my_outputs = get_outputs(X_batch)
    return my_outputs

def get_activities(model):
	conv_nodes=[]
	avg_pool_nodes=[]
	max_pool_nodes=[]
	for the_node in model.nodes:
		if (model.nodes[the_node].get_config()['name']) == 'Convolution2D':
			conv_nodes.append(the_node)
		if (model.nodes[the_node].get_config()['name']) == 'AveragePooling2D':
			avg_pool_nodes.append(the_node)
		if (model.nodes[the_node].get_config()['name']) == 'MaxPooling2D':
			max_pool_nodes.append(the_node)
	return conv_nodes, avg_pool_nodes, max_pool_nodes

def output_stuff(params, model, history):

	#parameters
	with open(filename+'_params.json', 'w') as fp:
	    json.dump(params, fp)

	#architecture
	json_string = model.to_json()
	open(filename+'_model.json', 'w').write(json_string)

	#weights
	model.save_weights(filename+'_weights.h5')

	#history
	history_file = open(filename+"_history.txt", "w")
	history_file.write(str(history.history))
	history_file.close()

def output_stats(filename,conv_nodes, avg_pool_nodes, max_pool_nodes):

	stats_file = open(filename+"_stats.csv", 'w')
	writer = csv.writer(stats_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
	row1=['Statistics for %s' %filename]
	writer.writerow(row1)

	for i in range(len(conv_nodes)):
		# Activations are 4-D tensors with (test_datapoints, n_filters_i,shapex,shapey)
		A_conv_raw = get_outputs(model,'input',conv_nodes[i],X_test[:test_datapoints])
		A_avg_raw = get_outputs(model,'input',avg_pool_nodes[i],X_test[:test_datapoints])
		A_max_raw = get_outputs(model,'input',max_pool_nodes[i],X_test[:test_datapoints])
		# Average over test_datapoints to make a 3-D tensor (n_filters_i,shapex,shapey)
		A_conv = np.average(A_conv_raw,axis=0)
		A_avg = np.average(A_avg_raw,axis=0)
		A_max = np.average(A_max_raw,axis=0)
		#stats of all conv node activations in layer_i
		mean_A_conv=np.average(A_conv, axis=(0,1,2))
		std_A_conv=np.std(A_conv, axis=(0,1,2))
		max_A_conv=np.max(A_conv, axis=(0,1,2))
		#mean and std for each feature_maps in that layer
		features_mean_A_conv=np.average(A_conv,axis=(1,2))
		features_max_A_conv=np.max(A_conv,axis=(1,2))
		#std of feature map 0 means/maxs across inputs
		trials_mean_A_conv=np.average(A_conv_raw,axis=(2,3))
		trials_max_A_conv=np.max(A_conv_raw,axis=(2,3))
		std_mean_across_trials=np.std(trials_mean_A_conv,axis=0)
		std_max_across_trials=np.std(trials_max_A_conv,axis=0)

		# Write to CSV file
		writer.writerow([])
		writer.writerow(['layer',i])
		writer.writerow(['mean(A) across layer ',mean_A_conv])
		writer.writerow(['std(A) across layer ',std_A_conv])
		writer.writerow(['max(A) across layer ',max_A_conv])

		writer.writerow(['mean(A across trials) for each feature ',[m for m in features_mean_A_conv]])
		writer.writerow(['max(A across trials) for each feature ',[m for m in features_max_A_conv]])

		writer.writerow(['mean(feature map 0) for each trial ',[m[0] for m in trials_mean_A_conv]])
		writer.writerow(['max(feature map 0) for each trial ',[m[0] for m in trials_max_A_conv]])
		
		writer.writerow(['std (mean(feature map)) across trials ',std_mean_across_trials])
		writer.writerow(['std (max(feature map)) across trials ',std_max_across_trials])

		writer.writerow(['mean(avg pool unit across trials) within feature map 0 ',[m[0] for m in A_avg[0]]])
		writer.writerow(['mean(max pool unit across trials) within feature map 0 ',[m[0] for m in A_max[0]]])

	stats_file.close()

output_stuff(params, model, history)
conv_nodes, avg_pool_nodes, max_pool_nodes = get_activities(model)
output_stats(filename,conv_nodes, avg_pool_nodes, max_pool_nodes)