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
filename='cifar100_v4_krizhevsky'
batch_size = 128
classes = 100
epochs = 50
learning_rate=0.01
decay=5e-4
momentum=0.9
nesterov=True
train_datapoints=50000 #samples_train
test_datapoints=30000 #samples_test
n_filters = [16,32,64,32] #[96,256,384,384,256] # number of feature maps at layer_i, len(n_conv_layers)
pool_size = [2,2] # square size of pooling window at layer_i, len(n_conv_layers)
kernel_size = [7,5,3,3] #[11,5,3,3,3] # square size of kernel at layer_i, len(n_conv_layers)
stride_size = [1] #[4]
dropout_frac = [0.5,0.5] # dropout fraction at layer_i, len(n_conv_layers+n_dense_layers)
dense_size = [512,512,classes] # output dimension for dense layers, len(n_dense_layers)
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
}

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255






'''
Network
Model off of Krizhevsky 2012
http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
'''
model = Graph()

# Input and 1st conv layer
model.add_input(name='input', input_shape=image_dim)
model.add_node(Convolution2D(n_filters[0], kernel_size[0], kernel_size[0], 
				activation='relu', input_shape=image_dim, subsample=(stride_size[0],stride_size[0])),
				name='conv%s' %1, input='input')

# 2nd conv and maxpool layer
model.add_node(Convolution2D(n_filters[1], kernel_size[1], kernel_size[1], 
				activation='relu'),
				name='conv%s' %2, input='conv%s' %1)
model.add_node(MaxPooling2D(pool_size=(pool_size[0], pool_size[0])),
				name='maxpool%s' %1, input='conv%s' %2)

# 2nd conv and maxpool layer
model.add_node(Convolution2D(n_filters[2], kernel_size[2], kernel_size[2], 
				activation='relu'),
				name='conv%s' %3, input='maxpool%s' %1)
model.add_node(MaxPooling2D(pool_size=(pool_size[1], pool_size[1])),
				name='maxpool%s' %2, input='conv%s' %3)

# 4th conv layers
model.add_node(Convolution2D(n_filters[3], kernel_size[3], kernel_size[3], 
				activation='relu'),
				name='conv%s' %4, input='maxpool%s' %2)

# 1st, 2nd, 3rd dense layers
model.add_node(Flatten(),
				name='flatten%s' %1, input='conv%s' %4)
model.add_node(Dense(dense_size[0],
				activation='relu', init='glorot_uniform'),
				name='dense%s' %1, input='flatten%s' %1)
model.add_node(Dropout(dropout_frac[0]),
				name='drop%s' %1, input='dense%s' %1)
model.add_node(Dense(dense_size[1],
				activation='relu', init='glorot_uniform'),
				name='dense%s' %2, input='drop%s' %1)
model.add_node(Dropout(dropout_frac[1]),
				name='drop%s' %2, input='dense%s' %2)
model.add_node(Dense(dense_size[2],
				activation='relu', init='glorot_uniform'),
				name='dense_output', input='drop%s' %2)

# Output
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
	open(filename+'_model', 'w').write(json_string)

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
		# Average over test_datapoints to make a 3-D tensor (n_filters_i,shapex,shapey)
		A_conv = np.average(A_conv_raw,axis=0)
		#stats of all conv node activations in layer_i
		mean_A_conv=np.average(A_conv, axis=(0,1,2))
		std_A_conv=np.std(A_conv, axis=(0,1,2))
		max_A_conv=np.max(A_conv, axis=(0,1,2))
		#stats across feature_maps in that layer
		std_across_features_mean_A_conv=np.std(np.average(A_conv,axis=(1,2)))
		std_across_features_max_A_conv=np.std(np.max(A_conv,axis=(1,2)))
		# Write to CSV file
		writer.writerow([])
		writer.writerow(['layer',i])
		writer.writerow(['mean_A_conv',mean_A_conv])
		writer.writerow(['std_A_conv',std_A_conv])
		writer.writerow(['max_A_conv absolute, relative',max_A_conv, max_A_conv/mean_A_conv])
		writer.writerow(['std_across_features_mean_A_conv  absolute, relative',
							std_across_features_mean_A_conv,
							std_across_features_mean_A_conv/mean_A_conv])
		writer.writerow(['std_across_features_max_A_conv absolute, relative',
							std_across_features_max_A_conv,
							std_across_features_max_A_conv/mean_A_conv])

	for i in range(len(avg_pool_nodes)):
		A_avg_raw = get_outputs(model,'input',avg_pool_nodes[i],X_test[:test_datapoints])
		A_avg = np.average(A_avg_raw,axis=0)
		mean_A_avg=np.average(A_avg,axis=(0,1,2))
		#stats of all avgpooled node activations (dxd subset, like superpixels)
		#in feature_map_0 of layer_i
		mean_acoss_map0_A_avg=np.average(A_avg[0], axis=(0,1))
		std_acoss_map0_A_avg=np.std(A_avg[0], axis=(0,1))
		max_acoss_map0_A_avg=np.max(A_avg[0], axis=(0,1))
		#stats of avgpooled center node (shapex/2, shapey/2) across feature_maps in layer_i
		#center is arbitrarily chosen, but choosing unit_0,0 might create edge effects
		center_x, center_y = A_avg.shape[1]/2, A_avg.shape[2]/2
		std_across_features_A_avg_centerunit=np.std(A_avg[:,center_x,center_y])
		writer.writerow([])
		writer.writerow(['layer',i])
		writer.writerow(['mean_acoss_map0_A_avg absolute, relative',mean_acoss_map0_A_avg,
							mean_acoss_map0_A_avg/mean_A_avg])
		writer.writerow(['std_acoss_map0_A_avg absolute, relative',std_acoss_map0_A_avg,
							std_acoss_map0_A_avg/mean_A_avg])
		writer.writerow(['max_acoss_map0_A_avg absolute, relative',max_acoss_map0_A_avg,
							max_acoss_map0_A_avg/mean_A_avg])
		writer.writerow(['std_across_features_A_avg_centerunit absolute, relative',
							std_across_features_A_avg_centerunit, 
							std_across_features_A_avg_centerunit/mean_A_avg])

	for i in range(len(max_pool_nodes)):
		A_max_raw = get_outputs(model,'input',max_pool_nodes[i],X_test[:test_datapoints])
		A_max = np.average(A_max_raw,axis=0)
		mean_A_max=np.average(A_max,axis=(0,1,2))
		#stats of all maxpooled node activations in feature_map_0 of layer_i
		mean_acoss_map0_A_max=np.average(A_max[0], axis=(0,1))
		std_acoss_map0_A_max=np.std(A_max[0], axis=(0,1))
		max_acoss_map0_A_max=np.max(A_max[0], axis=(0,1))
		#stats of avgpooled center node across feature_maps in layer_i
		center_x, center_y = A_max.shape[1]/2, A_max.shape[2]/2
		std_across_features_A_max_centerunit=np.std(A_max[:,center_x,center_y])
		writer.writerow([])
		writer.writerow(['layer',i])
		writer.writerow(['mean_acoss_map0_A_max absolute, relative',mean_acoss_map0_A_max,
							mean_acoss_map0_A_max/mean_A_max])
		writer.writerow(['std_acoss_map0_A_max absolute, relative',std_acoss_map0_A_max,
							std_acoss_map0_A_max/mean_A_max])
		writer.writerow(['max_acoss_map0_A_max absolute, relative',max_acoss_map0_A_max,
							max_acoss_map0_A_max/mean_A_max])
		writer.writerow(['std_across_features_A_max_centerunit absolute, relative',
							std_across_features_A_max_centerunit, 
							std_across_features_A_max_centerunit/mean_A_max])

	stats_file.close()

output_stuff(params, model, history)
conv_nodes, avg_pool_nodes, max_pool_nodes = get_activities(model)
output_stats(filename,conv_nodes, avg_pool_nodes, max_pool_nodes)