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
			probe_dict[key+'_pre']=nengo.Probe(model_dict[key+'_pre'])
			#sublayer that receives feedforward from above and feedback from sal
			my_input=model_dict[key+'_pre']
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(FeatureMap2d(shape_in,
											activation='linear', recurrent='none'))
			conn_dict[key+'_pre'+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			#FB connections built in method build_salience_layers
			probe_dict[key]=nengo.Probe(model_dict[key])

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
			shape_out=np.prod(my_input.output.shape_out)
			model_dict[key] = nengo.Node(Flatten(shape_in,shape_out))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])			

		elif info['type']=='Dense':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(Dense_1d(shape_in, info['weights'].shape[1],
											weights=info['weights'],activation=info['activation'],
											biases=info['biases']))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif info['type']=='DenseFB':
			my_input1=model_dict[info['input_name'][0]] 
			my_input2=model_dict[info['input_name'][1]]
			shape_in=my_input1.output.shape_out
			model_dict[key] = nengo.Node(Dense_1d(shape_in, info['weights'].shape[1],
											weights=info['weights'],activation=info['activation'],
											biases=info['biases']))
			# conn_dict[info['input_name'][0]+'_to_'+key]=nengo.Connection( #FF from last sal layer
			# 								my_input1,model_dict[key],synapse=None)
			conn_dict[info['input_name'][1]+'_to_'+key]=nengo.Connection( #FF from conv layer
											my_input2,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif info['type']=='output':		
			my_input1=model_dict[info['input_name'][0]]
			my_input2=model_dict[info['input_name'][1]]
			model_dict[key] = nengo.Node(output=lambda t,x: np.argmax(x),
											size_in=my_input1.size_out)
			conn_dict[info['input_name'][0]+'_to_'+key]=nengo.Connection( #FF from conv
											my_input1,model_dict[key],synapse=None)
			# conn_dict[info['input_name'][1]+'_to_'+key]=nengo.Connection( #FF from sal
			# 								my_input2,model_dict[key],synapse=None)
			probe_dict['output_probe']=nengo.Probe(model_dict[key],sample_every=pt)
		#TODO dropout
		#TODO bypass unnecessary dense, flattens in keras arch_dict

	return

def build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,FB_dict):

	for key, info in arch_dict.items():

		if info['type']=='Convolution2D':
			#F unit
			#assumption: sum of all nodes in a feature map represents salience/presence of that feature
			my_input=model_dict[key]
			my_name='sal_F_'+key
			shape_in=my_input.output.shape_out #(filters,convN_x, convN_y)
			shape_out=shape_in[0] #dimension = number of filters in sal layer
			model_dict[my_name] = nengo.Node(Sal_F(shape_in,shape_out))
			conn_dict[key+'_to_'+my_name]=nengo.Connection(
											my_input,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])

			#C unit
			#assumption: C units compete only within layer, but receive FB from higher salience layers
			last_name='sal_F_'+key
			my_name='sal_C_'+key
			my_input=model_dict[last_name] #input F unit
			shape_in=my_input.output.shape_out
			shape_out=shape_in
			comp=FB_dict[key]['competition']
			model_dict[my_name] = nengo.Node(Sal_C(shape_in,shape_out,
											competition=comp)) #softmax
			conn_dict[last_name+'_to_'+my_name]=nengo.Connection(
											my_input,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])

			#B_near unit
			#assumption: salience of FM_n feeds back additively/multiplicatively to FM_n-1, not farther
			last_name='sal_C_'+key
			my_name='sal_B_near_'+key
			my_input=model_dict[last_name] #input C unit
			shape_in=my_input.output.shape_out
			shape_out=model_dict[key].output.shape_out #back to (filters,convN_x, convN_y)
			FB_near_type=FB_dict[key]['FB_near_type']
			k_FB_near=FB_dict[key]['k_FB_near']
			model_dict[my_name] = nengo.Node(Sal_B_near(shape_in,shape_out,
											feedback_near=FB_near_type,k_FB_near=k_FB_near))
			conn_dict[last_name+'_to_'+my_name]=nengo.Connection( #connect C to B_near
											my_input,model_dict[my_name],synapse=None)
			probe_dict[my_name]=nengo.Probe(model_dict[my_name])
			tau_FB_near=FB_dict[key]['tau_FB_near']
			conn_dict[my_name+'_to_'+key]=nengo.Connection( #connect B_near back to conv
											model_dict[my_name],model_dict[key],synapse=tau_FB_near)

			#B_far unit
			#assumption: salience of FM_n feedsback to salience of FM_{n-1}, not to FM_n-1
			#assumption: contribution of sal_F_n-1 to sal_F_n is learned in Keras and inverted 
			if int(key[-1]) > 0:
				last_name='sal_C_'+key
				my_name='sal_B_far_'+key
				my_input=model_dict[last_name] #input C unit
				shape_in=my_input.output.shape_out
				layer_n=int(key[-1])
				layer_n_minus_1=layer_n-1
				output_name='sal_C_'+key[:-1]+str(layer_n_minus_1) #conv1 => conv0
				my_output=model_dict[output_name] 
				shape_out=my_output.output.shape_in #back to FM_n-1 C units
				W=arch_dict['sal_FB_'+str(layer_n)+str(layer_n+1)]['weights']
				FB_far_type=FB_dict[key]['FB_far_type']
				k_FB_far=FB_dict[key]['k_FB_far']
				model_dict[my_name] = nengo.Node(Sal_B_far(shape_in,shape_out,W,
												feedback_far=FB_far_type,k_FB_far=k_FB_far))
				conn_dict[key+'_to_'+my_name]=nengo.Connection(
												my_input,model_dict[my_name],synapse=None)
				probe_dict[my_name]=nengo.Probe(model_dict[my_name])
				tau_FB_far=FB_dict[key]['tau_FB_far']
				conn_dict[my_name+'_to_'+key]=nengo.Connection( #connect B_near back to conv
												model_dict[my_name],my_output,synapse=tau_FB_far)


def build_model(arch_dict,X_train,frac,pt,FB_dict):
	
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
		build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,FB_dict)
	return model, model_dict, conn_dict, probe_dict

def simulate_error_rate(model,data,labels,probe_dict,frac,pt):
	
	print 'Running the error-rate simulation...'
	sim = nengo.Simulator(model)
	n_images=int(frac*len(data))
	sim.run(pt*n_images)
	y_predicted = sim.data[probe_dict['output_probe']].ravel()
	y_true = labels[:n_images]

	results={}
	results['error rate']=round(np.count_nonzero(y_predicted != y_true)/(1.0*len(y_true))*100)/100
	for n in np.arange(0.0,10.0,1.0):
		results[n]={}
		true_index=np.where(y_true==int(n))[0]
		predicted_index=np.where(y_predicted==n)[0]
		total_index=np.union1d(true_index,predicted_index)
		correct_index=np.intersect1d(true_index,predicted_index)
		false_pos=np.setdiff1d(total_index,true_index)
		false_neg=np.setdiff1d(total_index,predicted_index)
		correct_ratio=1.0*len(correct_index)/len(total_index)
		FP_ratio=1.0*len(false_pos)/len(total_index)
		FN_ratio=1.0*len(false_neg)/len(total_index)
		results[n]['false negative']=round(FN_ratio*100)/100
		results[n]['false positive']=round(FP_ratio*100)/100
		results[n]['correct']=round(correct_ratio*100)/100
	# print 'results',results
	results['y_predicted']=y_predicted
	results['y_true']=y_true
	return sim, results

def plot_features(layer,image,sim,model_dict,probe_dict,pt,frac):

	sal_vs_time=[] #(FM_activities,timesteps)
	data=sim.data[probe_dict[layer]] #(images*pt/dt,FMs)
	FMs=data.shape[1]
	timesteps=pt/0.001
	images=data.shape[0]/timesteps
	indices=np.arange(0,FMs,1)
	activations=data.reshape((images, timesteps, FMs))
	normed_activations=np.zeros((activations.shape))
	for im in range(normed_activations.shape[0]):
		for t in range(normed_activations.shape[1]):
			norm=np.sum(activations[im][t])
			for fm in range(normed_activations.shape[2]):
				normed_activations[im][t][fm] = activations[im][t][fm]/norm

	#image #, before and after feedback
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.bar(indices,normed_activations[image][0],width=1,label='t=0.0',alpha=0.5,color='b')
	ax.bar(indices,normed_activations[image][-1],width=1,label='t=%s' %timesteps,alpha=0.5, color='g')
	ax.set_xlabel('Feature #')
	ax.set_ylabel('Activation')
	ax.set_xlim(0,FMs)
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def main():

	X_train, y_train, X_test, y_test = load_mnist()
	data=(X_train,y_train)
	filename='mnist_CNN_v2_all_epochs=20'
	# filename='mnist_CNN_v1_relu_pool1'
	arch_dict = import_keras_json(filename)
	pt=0.005 #image presentation time, larger = more time for feedback
	frac=0.001 #fraction of dataset to simulate

	FB_dict={}
	for key, info in arch_dict.items():
		if key == 'conv0':
			FB_dict[key] = {
				'competition': 'none',
				'FB_near_type': 'none',
				'tau_FB_near': 0.001,
				'k_FB_near': 100,
				'FB_far_type': 'none',
				'tau_FB_far': 0.001,
				'k_FB_far': 1, 
				}
		if key == 'conv1':
			FB_dict[key] = {
				'competition': 'softmax',
				'FB_near_type': 'constant',
				'tau_FB_near': 0.001,
				'k_FB_near': 30,
				'FB_far_type': 'none',
				'tau_FB_far': 0.001,
				'k_FB_far': 1,
				}

	model, model_dict, conn_dict, probe_dict = build_model(arch_dict,data[0],frac,pt,FB_dict)
	sim, results=simulate_error_rate(model,data[0],data[1],probe_dict,frac,pt)
	plot_features('sal_F_conv1',0,sim,model_dict,probe_dict,pt,frac)

	print 'results'
	for key, item in results.items():
		print key, '...', item

main()