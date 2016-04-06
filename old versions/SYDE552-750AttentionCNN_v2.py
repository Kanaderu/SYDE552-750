# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import nengo
from nengo_deeplearning import Conv2d, Pool2d
from keras.datasets import mnist
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 20

#import images from MNIST data
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
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]

#import architecture and weights from keras network
print 'Loading Model Architecture from Keras .json output...'
in_filename='mnist_CNN_v2'
with open(in_filename+"_arch.json") as datafile:    
	arch_dict=load(datafile,preserve_order=True)


print 'Building the Network...'
model_dict={}
conn_dict={}
presentation_time=0.05
# image=X_train[0]
# model_dict['image']=image
model = nengo.Network()
with model:

	#rebuild the keras model
	for key, value in arch_dict.items():

		if key == 'input':
			def get_image_at_time(t):
				# print int(t/presentation_time)
				return X_train[int(t/presentation_time)].ravel()
			# my_input=model_dict['image'].ravel()
			# model_dict[key] = nengo.Node(my_input)
			model_dict[key] = nengo.Node(output=get_image_at_time)
			img_probe=nengo.Probe(model_dict[key])

		elif key == 'conv0':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(Conv2d(image_dim, value['weights'],
											biases=value['biases'],stride=value['stride']))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif value['type']=='Convolution2D':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(Conv2d(my_input.output.shape_out, value['weights'],
											biases=value['biases'],stride=value['stride']))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif value['type']=='MaxPooling2D':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(Pool2d(my_input.output.shape_out, value['pool_size'],
											kind='max',stride=value['stride']))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			# print 'Pool2D input shape',my_input.output.shape_out
			# print 'Pool2D output shape',model_dict[key].output.shape_out
			# print 'Keras activity matrix shape',arch_dict[key]['activities'].shape

		elif value['type']=='AveragePooling2D':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(Pool2d(my_input.output.shape_out, value['pool_size'],
											kind='avg',stride=value['stride']))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif value['type']=='Flatten':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=lambda t,x: x,
											size_in=np.prod(my_input.output.shape_out))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif value['type']=='Dense':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=lambda t,x: x,
											size_in=value['weights'].shape[1])
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None,
											transform=value['weights'].T)

		elif value['type']=='output':
			def summation(t,x):
				return 10*np.sum(x)
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=summation,
											size_in=my_input.size_out)
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			output_probe=nengo.Probe(model_dict[key])


	# #add salience layers
	# for key, value in arch_dict.items():

	# 	if value['type']=='Convolution2D':
	# 		def FM_sum(t,x):
	# 			print x.shape, np.sum(x,axis=(1,2))
	# 			return np.sum(x,axis=(1,2))
	# 		my_input=model_dict[key]
	# 		my_name='salience_'+my_input
	# 		model_dict[my_name] = nengo.Node(output=summation,
	# 										size_in=my_input.output.shape_out)
	# 		conn_dict[my_input+'_to_'+key]=nengo.Connection(
	# 										my_input,model_dict[my_name],synapse=None)

dt=0.001
T=presentation_time-dt
sim = nengo.Simulator(model)
for i in range(100):
	sim.run(T)
	p_out = sim.data[output_probe]
	print 'correct classification is',y_train[int(i*T/presentation_time)]
	print 'model classification is',np.average(p_out[i*T/dt:(i+1)*T/dt])