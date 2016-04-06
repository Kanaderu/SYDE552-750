# Peter Duggins, psipeter@gmail.com
# SYDE 552/750
# Final Project
# Winter 2016
# CNN Adapted from https://github.com/fchollet/keras/issues/762

# from __future__ import absolute_import
import numpy as np
import theano
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import csv

# Theano configuration
# theano.config.floatX = 'float32'
# theano.config.device = 'gpu'
# import theano.sandbox.cuda
# theano.sandbox.cuda.use("gpu")
# ssh ctnuser@ctngpu2.uwaterloo.ca -p 3656
# pswd neuro...with replacements eio

# Keras imports
from keras.datasets import mnist
from keras.models import Graph
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

# Hyperparameters
filename='keras_CNN_v5_line_3'
batch_size = 128
learning_rate=0.01
decay=1e-6
momentum=0.9
nesterov=True
split=0.8
frac=0.1
epochs=10
dataset='lines'


if dataset=='mnist':

	print 'Loading MNIST data...'
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	img_x, img_y, img_z = 28,28,1
	image_dim=(img_z,img_x,img_y)
	X_train = X_train.reshape(X_train.shape[0], img_z, img_x, img_y)
	X_test = X_test.reshape(X_test.shape[0], img_z, img_x, img_y)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	classes=10

if dataset=='lines':

	datafile='mix_data.npz'
	labelfile='mix_labels.npz'
	data=np.load(datafile)['arr_0']
	labels=np.load(labelfile)['arr_0']
	samples_train=int(split*data.shape[0])
	samples_test=len(data)-samples_train
	image_dim=(1,data[0].shape[0],data[0].shape[1])
	classes=3

	X_train=data[:samples_train]
	y_train=labels[:samples_train]
	X_test=data[samples_train:]
	y_test=labels[samples_train:]

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = X_train.reshape(samples_train,image_dim[0],image_dim[1],image_dim[2])
	X_test = X_test.reshape(samples_test,image_dim[0],image_dim[1],image_dim[2])


if dataset=='crosses':
	datafile='+_data.npz'
	labelfile='+_labels.npz'
	data=np.load(crossesfile)['arr_0']
	samples_train=data.shape[0]
	image_dim=(1,data[0].shape[0],data[0].shape[1])
	X_train=data[:samples_train]
	X_train = X_train.reshape(samples_train,image_dim[0],image_dim[1],image_dim[2])

# MNIST data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# labels = np.concatenate((y_train,y_test))
# img_x, img_y = 28,28
# X_train = X_train.reshape(X_train.shape[0], 1, img_x, img_y)
# X_test = X_test.reshape(X_test.shape[0], 1, img_x, img_y)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# samples_train = frac*X_train.shape[0]
# samples_test = frac*X_test.shape[0]
# image_dim=(1,img_x,img_y)

# #lines data (no shuffling)
# datafile='lines_data_50000.npy'
# labelfile='lines_labels_50000.npy'
# data=np.load(datafile)
# labels=np.load(labelfile)
# split=0.8
# datapoints_train=int(split*len(data))
# datapoints_test=len(data)-datapoints_train
# samples_train=datapoints_train*frac
# samples_test=datapoints_test*frac
# image_dim=(1,data[0].shape[0],data[0].shape[1])
# X_train=data[:samples_train] 
# y_train=labels[:samples_train]
# X_test=data[-samples_test:]
# y_test=labels[-samples_test:]
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train.reshape(samples_train,image_dim[0],image_dim[1],image_dim[2])
# X_test = X_test.reshape(samples_test,image_dim[0],image_dim[1],image_dim[2])

Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)

'''
Network
'''
def get_outputs(model, input_name, layer_name, X_batch):
    get_outputs = theano.function([model.inputs[input_name].input],
    				model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    my_outputs = get_outputs(X_batch)
    return my_outputs

model = Graph()

#input
model.add_input(name='input', input_shape=image_dim)

#1st conv layer
FM1=7
ker1=5
pool1=2
pad1=0 #no option for padding in convolution2d anyway
stride1=1
model.add_node(Convolution2D(FM1, ker1, ker1, subsample=(stride1, stride1),activation='relu'),
							name='conv0', input='input')
model.add_node(MaxPooling2D(pool_size=(pool1, pool1)),name='maxpool0', input='conv0')
model.add_node(Dropout(0.5),name='drop0',input='maxpool0')
#1st salience layer
size1=get_outputs(model,'input','maxpool0',X_train[:samples_train]).shape[-1]
model.add_node(AveragePooling2D(pool_size=(size1,size1)),name='sal_f0', input='drop0')
model.add_node(Flatten(),name='flat_sal_f0', input='sal_f0')
model.add_node(Flatten(),name='final_sal_f0', input='flat_sal_f0') #for symmetry

#2nd conv layer
FM2=5
ker2=5
pool2=2
pad2=0
stride2=1
model.add_node(Convolution2D(FM2, ker2, ker2, subsample=(stride2, stride2), activation='relu'),
							name='conv1', input='drop0')
model.add_node(MaxPooling2D(pool_size=(pool2, pool2)), name='maxpool1', input='conv1')
model.add_node(Dropout(0.5), name='drop1', input='maxpool1')
#2nd salience layer
size2=get_outputs(model,'input','maxpool1',X_train[:samples_train]).shape[-1]
model.add_node(AveragePooling2D(pool_size=(size2, size2)), name='sal_f1', input='drop1')
model.add_node(Flatten(), name='flat_sal_f1', input='sal_f1')
model.add_node(Dense(FM2, activation='relu'), name='dense_sal_12', input='final_sal_f0')
model.add_node(Flatten(), name='final_sal_f1', inputs=['flat_sal_f1','dense_sal_12'], merge_mode='sum') #must be this order of inputs

#3nd conv layer
FM3=3
ker3=5
pool3=2
pad3=0
stride3=1
model.add_node(Convolution2D(FM3, ker3, ker3, subsample=(stride3, stride3), activation='relu'),
							name='conv2', input='drop1')
model.add_node(MaxPooling2D(pool_size=(pool3, pool3)), name='maxpool2', input='conv2')
model.add_node(Dropout(0.5), name='drop2', input='maxpool2')
#3nd salience layer
size3=get_outputs(model,'input','maxpool2',X_train[:samples_train]).shape[-1]
model.add_node(AveragePooling2D(pool_size=(size3, size3)), name='sal_f2', input='drop2')
model.add_node(Flatten(), name='flat_sal_f2', input='sal_f2')
model.add_node(Dense(FM2, activation='relu'), name='dense_sal_23', input='final_sal_f1')
model.add_node(Flatten(), name='final_sal_f2', inputs=['flat_sal_f2','dense_sal_23'], merge_mode='sum') #must be this order of inputs

#flatten layer
model.add_node(Flatten(), name='flat_conv', input='drop2')
model.add_node(Flatten(), name='flat_sal', input='final_sal_f2') #for symmetry

#1st dense layer (fully connected)
model.add_node(Dense(128, activation='relu'), name='dense_C0', input='flat_conv')
model.add_node(Dense(128, activation='relu'), name='dense_S0', input='flat_sal')
model.add_node(Dropout(0.5), name='drop_C0', input='dense_C0')
model.add_node(Dropout(0.5), name='drop_S0', input='dense_S0')

#2d dense layer (classification)
model.add_node(Dense(classes, activation='softmax'), name='dense_C1', input='drop_C0')
model.add_node(Dense(classes, activation='softmax'), name='dense_S1', input='drop_S0')

# output
model.add_output(name='output', inputs=['dense_C1','dense_S1'], merge_mode='sum')





'''optimize, compile, and train'''
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(sgd, {'output':'categorical_crossentropy'})
history=model.fit({'input':X_train[:samples_train], 'output':Y_train[:samples_train]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input':X_test[:samples_test], 'output':Y_test[:samples_test]})

#TODO automate this with model, weight information (gave up after many hrs and no Keras.usergroups response)
arch_dict=OrderedDict((
	('input',			{'type':'input', 			'input_name':'image', 							'weights':None, 										'biases':None, 											'activation':None 																}),

	('conv0', 			{'type':'Convolution2D', 	'input_name':'input', 							'weights':model.nodes['conv0'].get_weights()[0],		'biases':model.nodes['conv0'].get_weights()[1],			'activation':'relu', 	'stride':stride1, 										}),
	('maxpool0', 		{'type':'MaxPooling2D', 	'input_name':'conv0', 							'weights':None, 										'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'max', 	'pool_size': pool1	}),
	('drop0',			{'type':'Dropout', 			'input_name':'maxpool0', 						'weights':None,											'biases':None,											'activation':None,  				 											}),
	('sal_f0', 			{'type':'AveragePooling2D', 'input_name':'drop0', 							'weights':None, 										'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'avg', 	'pool_size': size1	}),
	('flat_sal_f0',		{'type':'Flatten', 			'input_name':'sal_f0', 							'weights':None,											'biases':None,											'activation':None,  															}),
	('final_sal_f0',	{'type':'Flatten', 			'input_name':'flat_sal_f0', 					'weights':None,											'biases':None,											'activation':None,  	 														}),

	('conv1', 			{'type':'Convolution2D', 	'input_name':'drop0', 							'weights':model.nodes['conv1'].get_weights()[0], 		'biases':model.nodes['conv1'].get_weights()[1], 		'activation':'relu', 	'stride':stride2, 										}),
	('maxpool1',		{'type':'MaxPooling2D', 	'input_name':'conv1', 							'weights':None,											'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'max', 	'pool_size': pool2	}),
	('drop1',			{'type':'Dropout', 			'input_name':'maxpool1', 						'weights':None,											'biases':None,											'activation':None,  				 											}),
	('sal_f1', 			{'type':'AveragePooling2D', 'input_name':'drop1', 							'weights':None, 										'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'avg', 	'pool_size': size2	}),
	('flat_sal_f1', 	{'type':'Flatten', 			'input_name':'sal_f1', 							'weights':None,											'biases':None,											'activation':None,  				 											}),
	('dense_sal_12', 	{'type':'Dense', 			'input_name':'final_sal_f0', 					'weights':model.nodes['dense_sal_12'].get_weights()[0], 'biases':model.nodes['dense_sal_12'].get_weights()[1],	'activation':'relu',															}),
	('final_sal_f1', 	{'type':'Flatten_Merge', 	'input_name':['flat_sal_f1','dense_sal_12'], 	'weights':None,											'biases':None,											'activation':'relu',	 														}),

	('conv2', 			{'type':'Convolution2D', 	'input_name':'drop1', 							'weights':model.nodes['conv2'].get_weights()[0], 		'biases':model.nodes['conv2'].get_weights()[1], 		'activation':'relu', 	'stride':stride3, 										}),
	('maxpool2',		{'type':'MaxPooling2D', 	'input_name':'conv2', 							'weights':None,											'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'max', 	'pool_size': pool3	}),
	('drop2',			{'type':'Dropout', 			'input_name':'maxpool2', 						'weights':None,											'biases':None,											'activation':None,  				 											}),
	('sal_f2', 			{'type':'AveragePooling2D', 'input_name':'drop2', 							'weights':None, 										'biases':None,											'activation':None, 		'stride':1, 	'pool_type':'avg', 	'pool_size': size3	}),
	('flat_sal_f2', 	{'type':'Flatten', 			'input_name':'sal_f2', 							'weights':None,											'biases':None,											'activation':None,  				 											}),
	('dense_sal_23', 	{'type':'Dense', 			'input_name':'final_sal_f1', 					'weights':model.nodes['dense_sal_23'].get_weights()[0], 'biases':model.nodes['dense_sal_23'].get_weights()[1],	'activation':'relu',															}),
	('final_sal_f2', 	{'type':'Flatten_Merge', 	'input_name':['flat_sal_f2','dense_sal_23'], 	'weights':None,											'biases':None,											'activation':'relu',	 														}),

	('flat_conv', 		{'type':'Flatten', 			'input_name':'drop2', 							'weights':None,											'biases':None,											'activation':None, 		 														}),
	('flat_sal', 		{'type':'Flatten', 			'input_name':'final_sal_f2',					'weights':None,											'biases':None,											'activation':None,  				 											}),
	('dense_C0', 		{'type':'Dense', 			'input_name':'flat_conv', 						'weights':model.nodes['dense_C0'].get_weights()[0], 	'biases':model.nodes['dense_C0'].get_weights()[1],		'activation':'relu',															}),
	('dense_S0', 		{'type':'Dense', 			'input_name':'flat_sal', 						'weights':model.nodes['dense_S0'].get_weights()[0], 	'biases':model.nodes['dense_S0'].get_weights()[1],		'activation':'relu',															}),
	('dense_C1', 		{'type':'Dense', 			'input_name':'dense_C0', 						'weights':model.nodes['dense_C1'].get_weights()[0],		'biases':model.nodes['dense_C1'].get_weights()[1], 		'activation':'softmax', 														}),
	('dense_S1', 		{'type':'Dense', 			'input_name':'dense_S0', 						'weights':model.nodes['dense_S1'].get_weights()[0],		'biases':model.nodes['dense_S1'].get_weights()[1], 		'activation':'softmax', 														}),

	('output', 			{'type':'output', 			'input_name':['dense_C1','dense_S1'], 			'weights':None, 										'biases':None, 											'activation':None 																}),
))


'''output'''
#save layer statistics

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

def output_stuff(model, history):

	#parameters
	params={ #dictionary
		'filename' : filename,
		'batch_size' : batch_size,
		'classes' : classes,
		'epochs' : epochs,
		'learning_rate' : learning_rate,
		'decay' : decay,
		'momentum' : momentum,
		'nesterov' : nesterov,
		'samples_train' : samples_train,
		'samples_test' : samples_train,
		}
	with open(filename+'_params.json', 'w') as fp:
	    json.dump(params, fp)

	#architecture, for keras and nengo loading respectively
	json_string = model.to_json()
	open(filename+'_model.json', 'w').write(json_string)


	with open(filename+"_arch.json","w") as datafile:
		dump(arch_dict, datafile)

	# weights
	# model.save_weights(filename+'_weights.h5')

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
		# Activations are 4-D tensors with (samples_test, n_filters_i,shapex,shapey)
		A_conv_raw = get_outputs(model,'input',conv_nodes[i],X_test[:samples_test])
		# A_avg_raw = get_outputs(model,'input',avg_pool_nodes[i],X_test[:samples_test])
		A_max_raw = get_outputs(model,'input',max_pool_nodes[i],X_test[:samples_test])
		# Average over samples_test to make a 3-D tensor (n_filters_i,shapex,shapey)
		A_conv = np.average(A_conv_raw,axis=0)
		# A_avg = np.average(A_avg_raw,axis=0)
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
		# writer.writerow(['mean(avg pool unit across trials) within feature map 0 ',[m[0] for m in A_avg[0]]])
		writer.writerow(['mean(max pool unit across trials) within feature map 0 ',[m[0] for m in A_max[0]]])

	stats_file.close()

output_stuff(model, history)
conv_nodes, avg_pool_nodes, max_pool_nodes = get_activities(model)
output_stats(filename,conv_nodes, avg_pool_nodes, max_pool_nodes)

output1 = get_outputs(model,'input','dense_C1',X_test[:samples_test])
output2 = get_outputs(model,'input','dense_S1',X_test[:samples_test])
guesses = np.argmax(output1+output2, axis=1)
answers = y_test[:samples_test]
error_rate=np.count_nonzero(guesses != answers)/float(len(answers))
print ('total test error rate', filename, error_rate)
# print (model.nodes['conv0'].get_weights()[0].shape)
# print (model.nodes['dense_C0'].get_weights()[0].shape)
# print (get_outputs(model,'input','sal_f0',X_test[:samples_test]).shape)
# print (get_outputs(model,'input','drop0',X_test[:samples_test]).shape)
# print (get_outputs(model,'input','sal_f1',X_test[:samples_test]).shape)
# print (get_outputs(model,'input','drop0',X_test[:samples_test]).shape)
# print (model.nodes['drop0'].get_config())
# print ([get_outputs(model,'input',the_node,X_test[:samples_test]).shape for the_node in model.nodes])