# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
import json
from json_tricks.np import dump, dumps, load, loads, strip_comments
import nengo
from keras_nengo_layers import * 
# from nengo_deeplearning import Conv2d, Pool2d #moved into keras_nengo_layers
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

	for key, info in arch_dict.items():

		if key == 'input': #want to avoid passing images to build_from_keras
			model_dict[key] = nengo.Node(output=nengo.processes.PresentInput(images,pt))
			probe_dict['img_probe']=nengo.Probe(model_dict[key],sample_every=pt)

		elif info['type']=='Convolution2D':
			#sublayer that receives input and performs convolution
			my_input=model_dict[info['input_name']]
			if info['input_name'] == 'input':
				shape_in=my_input.output.inputs.shape[1:]
			else:
				shape_in=my_input.output.shape_out
			model_dict[key+'_pre'] = nengo.Node(Conv2d(shape_in, info['weights'],
											activation=info['activation'], biases=info['biases'],
											stride=info['stride']))
			conn_dict[info['input_name']+'_to_'+key+'_pre']=nengo.Connection(
											my_input,model_dict[key+'_pre'],synapse=None)
			#sublayer that receives feedforward from above and feedback from sal
			my_input=model_dict[key+'_pre']
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(FeatureMap2d(shape_in,
											activation='linear', recurrent='none'))
			conn_dict[key+'_pre'+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			#FB connections built in method build_salience_layers

		elif info['type']=='MaxPooling2D':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(Pool2d(shape_in, info['pool_size'],
											kind='max',stride=info['stride']))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif info['type']=='AveragePooling2D':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(Pool2d(shape_in, info['pool_size'],
											kind='avg',stride=info['stride']))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif info['type']=='Flatten':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			shape_out=[np.prod(my_input.output.shape_out)]
			model_dict[key] = nengo.Node(Flatten(shape_in,shape_out))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)

		elif info['type']=='Dense':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(Dense_1d(shape_in, info['weights'].shape[1],
											weights=info['weights'],activation=info['activation'],
											biases=info['biases']))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif info['type']=='output':		
			my_input=model_dict[info['input_name']]
			model_dict[key] = nengo.Node(output=lambda t,x: np.argmax(x),
											size_in=my_input.size_out)
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict['output_probe']=nengo.Probe(model_dict[key],sample_every=pt)

		#TODO dropout

	return

def build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,tau_FB):

	for key, info in arch_dict.items():

		if info['type']=='Convolution2D':
			#F unit
			my_input=model_dict[key]
			my_name='sal_F_'+key
			shape_in=my_input.output.shape_out #(filters,convN_x, convN_y)
			shape_out=shape_in[0] #dimension = number of filters in sal layer
			model_dict[my_name] = nengo.Node(Sal_F(shape_in,shape_out))
			conn_dict[key+'_to_'+my_name]=nengo.Connection(
											my_input,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])

			#C unit
			last_name=my_name
			my_F=model_dict[last_name] #input F unit
			my_name='sal_C_'+key
			shape_in=my_F.output.shape_out
			shape_out=shape_in
			model_dict[my_name] = nengo.Node(Sal_C(shape_in,shape_out,
											competition='none'))
			conn_dict[last_name+'_to_'+my_name]=nengo.Connection(
											my_F,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])

			#B_near unit
			last_name=my_name
			my_C=model_dict[last_name] #input C unit
			my_name='sal_B_near_'+key
			shape_in=my_C.output.shape_out
			shape_out=model_dict[key].output.shape_out #back to (filters,convN_x, convN_y)
			model_dict[my_name] = nengo.Node(Sal_B_near(shape_in,shape_out,
											feedback='constant'))
			conn_dict[last_name+'_to_'+my_name]=nengo.Connection(
											my_C,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])

			#connect B back to conv
			conn_dict[my_name+'_to_'+key]=nengo.Connection(
											model_dict[my_name],model_dict[key],synapse=tau_FB)

			#B_far unit
			# my_F=model_dict[my_name] #input C unit
			# my_name='sal_B_far_'+key
			# shape_in=my_input.output.shape_out
			# shape_out=model_dict[key].output.shape_out #TODO back to ??? C units
			# model_dict[my_name] = nengo.Node(Sal_B_far(shape_in,shape_out,
			# 								feedback='constant'))
			# conn_dict[key+'_to_'+my_name]=nengo.Connection(
			# 								my_input,model_dict[my_name],synapse=None)
			# probe_dict[my_name]=nengo.Probe(model_dict[my_name])

def build_model(arch_dict,X_train,frac,pt,tau_FB):
	
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
		build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,tau_FB)
	return model, model_dict, conn_dict, probe_dict

def simulate_error_rate(model,data,labels,probe_dict,frac,pt):
	
	print 'Running the error-rate simulation...'
	sim = nengo.Simulator(model)
	n_images=int(frac*len(data))
	sim.run(pt*n_images)
	guesses = sim.data[probe_dict['output_probe']].ravel()
	answers = labels[:n_images]
	error=np.count_nonzero(guesses != answers)/float(len(answers))
	return sim, error

def main():

	X_train, y_train, X_test, y_test = load_mnist()
	data=(X_train,y_train)
	filename='mnist_CNN_v1_relu_pool1'
	arch_dict = import_keras_json(filename)
	pt=0.002 #image presentation time
	frac=0.001 #fraction of dataset to simulate
	tau_FB=0.001
	model, model_dict, conn_dict, probe_dict = build_model(arch_dict,data[0],frac,pt,tau_FB)
	sim, error=simulate_error_rate(model,data[0],data[1],probe_dict,frac,pt)
	print 'sal B near probe',sim.data[probe_dict['sal_B_near_conv0']].sum()
	print 'dense1 probe',sim.data[probe_dict['dense1']].sum()
	print 'error rate', error

main()