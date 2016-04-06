# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import nengo
from nengo.processes import Process
from nengo.params import EnumParam, IntParam, NdarrayParam, TupleParam
from nengo_deeplearning import Conv2d, Pool2d
from keras.datasets import mnist
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 20
	

def load_mnist():
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
	return X_train, y_train, X_test, y_test

def import_keras_json(filename):

	#import architecture and weights from keras network
	print 'Loading Model Architecture from Keras .json output...'
	with open(filename+"_arch.json") as datafile:    
		arch_dict=load(datafile,preserve_order=True)
	return arch_dict

def build_from_keras(arch_dict,model_dict,conn_dict,probe_dict,images,pt):

	for key, value in arch_dict.items():

		if key == 'input': #want to avoid passing images to build_from_keras
			model_dict[key] = nengo.Node(output=nengo.processes.PresentInput(images,pt))
			probe_dict['img_probe']=nengo.Probe(model_dict[key],sample_every=pt)

		elif value['type']=='Convolution2D':
			my_input=model_dict[value['input_name']]
			if value['input_name'] == 'input':
				model_dict[key] = nengo.Node(Conv2d(my_input.output.inputs.shape[1:],
											value['weights'],activation=value['activation'],
											biases=value['biases'],stride=value['stride']))
			else:
				model_dict[key] = nengo.Node(Conv2d(my_input.output.shape_out, value['weights'],
											activation=value['activation'], biases=value['biases'],
											stride=value['stride']))
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

		# Todo
		# elif value['type']=='Dropout':
		# 	my_input=model_dict[value['input_name']]
		# 	if my_input == ()
		# 	model_dict[key] = nengo.Node(output=lambda t,x: x,
		# 									size_in=np.prod(my_input.output.shape_out))
		# 	conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
		# 									my_input,model_dict[key],synapse=None)

		elif value['type']=='Dense':
			my_input=model_dict[value['input_name']]
			activation=value['activation']
			if activation == 'softmax':
				model_dict[key] = nengo.Node(output=softmax,
											size_in=value['weights'].shape[1])
			elif activation == 'relu':
				model_dict[key] = nengo.Node(output=lambda t,x: np.maximum(x,np.zeros((x.shape))),
											size_in=value['weights'].shape[1])
			else:
				model_dict[key] = nengo.Node(output=lambda t,x: x,
											size_in=value['weights'].shape[1])

			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None,
											transform=value['weights'].T)

			b = nengo.Node(output=value['biases'].ravel())
			conn_dict['biases'+'_to_'+key]=nengo.Connection(
											b,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif value['type']=='output':		
			my_input=model_dict[value['input_name']]
			model_dict[key] = nengo.Node(output=lambda t,x: np.argmax(x),
											size_in=my_input.size_out)
			conn_dict[value['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict['output_probe']=nengo.Probe(model_dict[key],sample_every=pt)

	return

def build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict):

	for key, value in arch_dict.items():

		if value['type']=='Convolution2D':
			my_input=model_dict[key]
			my_name='salience_'+key
			shape_in=my_input.output.shape_out
			shape_out=shape_in[1:]
			model_dict[my_name] = nengo.Node(Sal2d(shape_in,shape_out))
			conn_dict[key+'_to_'+my_name]=nengo.Connection(
											my_input,model_dict[my_name],synapse=None)
			probe_dict['my_name']=nengo.Probe(model_dict[my_name])

class Sal2d(Process):

	shape_in = TupleParam('shape_in', length=3)
	shape_out = TupleParam('shape_out', length=2)

	def __init__(self, shape_in, shape_out):  # noqa: C901
		self.shape_in = tuple(shape_in)
		self.shape_out = tuple(shape_out)
		super(Sal2d, self).__init__(
			default_size_in=np.prod(self.shape_in),
			default_size_out=np.prod(self.shape_out))

	def make_step(self, size_in, size_out, dt, rng):
		assert size_in == np.prod(self.shape_in)
		assert size_out == np.prod(self.shape_out)
		shape_in = self.shape_in
		shape_out = self.shape_out

		def step_conv2d(t, x):
			x = x.reshape(shape_in)
			y = np.sum(x,axis=0)
			print x.sum(), y.sum()
			return y.ravel()

		return step_conv2d


def softmax(t,x):
	normalized = np.array([np.exp(xi)/np.sum((np.exp(x))) for xi in x])
	return normalized
def summation(t,x): #returns an array of the correct values
	summation = np.sum(x)
	return summation
def summation_bug(t,x): #returns an array of 1.
	return np.sum(x)
def maximization(t,x):
	maximumizizium = np.argmax(x)
	return maximumizizium
def FM_sum(t,x):
	print x.shape
	FM_sum=np.sum(x,axis=(1,2))
	print x.shape, FM_sum.shape, FM_sum.sum()
	return FM_sum

def build_model(arch_dict,X_train,frac,pt):
	
	print 'Building the Network...'
	model_dict={}
	conn_dict={}
	probe_dict={}
	model = nengo.Network()
	n_images=int(frac*len(X_train))
	model_dict = {}
	conn_dict = {}
	probe_dict = {}

	with model:
		build_from_keras(arch_dict,model_dict,conn_dict,probe_dict,X_train[:n_images],pt)
		build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict)
	return model, model_dict, conn_dict, probe_dict

def simulate_error_rate(model,data,labels,probe_dict,frac,pt):
	
	print 'Running the error-rate simulation...'
	sim = nengo.Simulator(model)
	n_images=int(frac*len(data))
	sim.run(pt*n_images)
	guesses = sim.data[probe_dict['output_probe']].ravel()
	answers = labels[:n_images]
	error=np.count_nonzero(guesses != answers)/float(len(answers))
	return error

def main():

	X_train, y_train, X_test, y_test = load_mnist()
	data=(X_train,y_train)
	filename='mnist_CNN_v1_test'
	arch_dict = import_keras_json(filename)
	pt=0.002 #image presentation time
	frac=0.0001 #fraction of dataset to simulate
	model, model_dict, conn_dict, probe_dict = build_model(arch_dict,data[0],frac,pt)
	error=simulate_error_rate(model,data[0],data[1],probe_dict,frac,pt)
	print 'error rate', error

main()