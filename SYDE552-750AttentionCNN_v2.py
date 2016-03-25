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
in_filename='mnist_CNN_v1'
with open(in_filename+"_arch.json") as datafile:    
	arch_dict=load(datafile,preserve_order=True)

# building my own network
model_dict={}
conn_dict={}
image=X_train[0]
model_dict['image']=image

model = nengo.Network()
with model:

	#rebuild the keras model
	for key, value in arch_dict.items():
		print key
		if key == 'input':
			my_input=model_dict['image'].ravel()
			model_dict[key] = nengo.Node(my_input)
			input_probe=nengo.Probe(model_dict[key])
		elif key == 'conv0':
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(Conv2d(image_dim, value['weights'],
											biases=value['biases'],stride=value['stride']))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			conv0_probe=nengo.Probe(model_dict[key])
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
			def func_flatten(t,x):
				output=x
				print x.shape,x.sum(),output.shape
				return output
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=func_flatten,#lambda t,x: x,
											size_in=np.prod(my_input.output.shape_out))
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			flatten_probe=nengo.Probe(model_dict[key])
		elif value['type']=='Dense':
			def func_dense(t,x):
				output=np.dot(x,value['weights'])
				print x.shape,x.sum(),output.shape
				return output
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=func_dense,#lambda t,x: np.dot(x,value['weights']),
											size_in=my_input.size_out)
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
		elif value['type']=='output':
			def func_output(t,x):
				output=x #softmax or sum
				print x.shape,x.sum(),output.shape
				return output
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=func_output,#lambda t,x: np.dot(x,value['weights']),
											size_in=my_input.size_out)
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			output_probe=nengo.Probe(model_dict[key])


sim = nengo.Simulator(model)
T=0.01
sim.run(T)
t=np.arange(0,T,0.001)
p1 = sim.data[input_probe]
p2 = sim.data[conv0_probe]
p3 = sim.data[flatten_probe]
p4 = sim.data[output_probe]
print p3.shape,p4.shape,p4.sum()
