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

def load_lines(datafile,labelfile,split=0.8):

	data=np.load(datafile)['arr_0']
	labels=np.load(labelfile)['arr_0']
	samples_train=int(split*data.shape[0])
	samples_test=len(data)-samples_train
	image_dim=(1,data[0].shape[0],data[0].shape[1])

	X_train=data[:samples_train]
	y_train=labels[:samples_train]
	X_test=data[samples_train:]
	y_test=labels[samples_train:]

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = X_train.reshape(samples_train,image_dim[0],image_dim[1],image_dim[2])
	X_test = X_test.reshape(samples_test,image_dim[0],image_dim[1],image_dim[2])

	return X_train, y_train, X_test, y_test

def load_crosses(crossesfile):
	
	data=np.load(crossesfile)['arr_0']
	samples_train=data.shape[0]
	image_dim=(1,data[0].shape[0],data[0].shape[1])
	X_train=data[:samples_train]
	X_train = X_train.reshape(samples_train,image_dim[0],image_dim[1],image_dim[2])
	return X_train

def import_keras_json(archfile):

	#import architecture and weights from keras network
	print 'Loading Model Architecture from Keras .json output...'
	with open(archfile+"_arch.json") as datafile:    
		arch_dict=load(datafile,preserve_order=True)
	return arch_dict

def build_from_keras(arch_dict,model_dict,conn_dict,probe_dict,images,pt):

	for key, info in arch_dict.items():

		if key == 'input': #want to avoid passing images to build_from_keras
			model_dict[key] = nengo.Node(output=nengo.processes.PresentInput(images,pt))
			probe_dict['input']=nengo.Probe(model_dict[key]) #,sample_every=pt

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

		elif info['type']=='Dense':
			# print key, shape_in, info['weights'].shape
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			model_dict[key] = nengo.Node(Dense_1d(shape_in, info['weights'].shape[1],
											weights=info['weights'],activation=info['activation'],
											biases=info['biases']))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
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

		elif info['type']=='Flatten_Merge':
			my_input1=model_dict[info['input_name'][0]] 
			my_input2=model_dict[info['input_name'][1]]
			shape_in=my_input1.output.shape_out
			model_dict[key] = nengo.Node(Flatten(shape_in,shape_out))
			conn_dict[info['input_name'][0]+'_to_'+key]=nengo.Connection( #from current sal layer
											my_input1,model_dict[key],synapse=None)
			conn_dict[info['input_name'][1]+'_to_'+key]=nengo.Connection( #from previous sal layer
											my_input2,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif info['type']=='Dropout':
			my_input=model_dict[info['input_name']]
			shape_in=my_input.output.shape_out
			shape_out=shape_in
			model_dict[key] = nengo.Node(Dropout(shape_in,shape_out))
			conn_dict[info['input_name']+'_to_'+key]=nengo.Connection(
											my_input,model_dict[key],synapse=None)
			probe_dict[key]=nengo.Probe(model_dict[key])

		elif info['type']=='output':		
			my_input1=model_dict[info['input_name'][0]]
			my_input2=model_dict[info['input_name'][1]]
			#pre_output is the input to the output node, before argmax(x) is applied; for plotting 
			model_dict['pre_'+key] = nengo.Node(output=lambda t,x: x,
											size_in=my_input1.size_out)
			conn_dict[info['input_name'][0]+'_to_'+key]=nengo.Connection( #FF from conv
											my_input1,model_dict['pre_'+key],synapse=None)
			# conn_dict[info['input_name'][1]+'_to_'+key]=nengo.Connection( #FF from sal
			# 								my_input2,model_dict['pre_'+key],synapse=None)
			#actual output node
			model_dict[key] = nengo.Node(output=lambda t,x: np.argmax(x),
											size_in=my_input1.size_out)
			conn_dict[info['input_name'][0]+'_to_'+key]=nengo.Connection( #FF from conv
											my_input1,model_dict[key],synapse=None)
			# conn_dict[info['input_name'][1]+'_to_'+key]=nengo.Connection( #FF from sal
			# 								my_input2,model_dict[key],synapse=None)
			probe_dict['pre_output']=nengo.Probe(model_dict['pre_'+key])
			probe_dict['output']=nengo.Probe(model_dict[key]) #,sample_every=pt

	return

def build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,FB_dict,stim_dict):

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
			#stimulate specific features externally
			model_dict['stim_'+my_name]=nengo.Node(stim_dict[key],size_out=shape_in)
			conn_dict['stim_'+my_name+'_to_'+my_name]=nengo.Connection(
											model_dict['stim_'+my_name],model_dict[my_name],synapse=None)			

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
			#assumption: contribution of sal_F_n-1 to sal_F_n is learned in Keras (dense) and inverted 
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
				W=arch_dict['dense_sal_'+str(layer_n)+str(layer_n+1)]['weights']
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


def build_model(arch_dict,X_train,frac,pt,FB_dict,stim_dict):
	
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
		build_salience_layers(arch_dict,model_dict,conn_dict,probe_dict,FB_dict,stim_dict)
	return model, model_dict, conn_dict, probe_dict

def get_error(sim,model,data,labels,probe_dict,n_images,pt):
	
	print 'Calculating error...'
	y_predicted = sim.data[probe_dict['output']][::pt/0.001].ravel().astype(int) #probe at end of pt
	y_true = labels[:n_images]

	results={}
	results['error rate']=round(np.count_nonzero(y_predicted != y_true)/(1.0*len(y_true))*100)/100

	#for each unique class, find the number correct, false positive, false negative
	# for n in np.arange(0.0,10.0,1.0): #MNIST
	for n in np.array([0,1,2]): #lines
		results[n]={}
		true_index=np.where(y_true==int(n))[0]
		predicted_index=np.where(y_predicted==n)[0]
		total_index=np.union1d(true_index,predicted_index)
		correct_index=np.intersect1d(true_index,predicted_index)
		false_pos=np.setdiff1d(total_index,true_index)
		false_neg=np.setdiff1d(total_index,predicted_index)
		if len(total_index)!=0: #if there were any occurances of class n
			correct_ratio=1.0*len(correct_index)/len(total_index)
			FP_ratio=1.0*len(false_pos)/len(total_index)
			FN_ratio=1.0*len(false_neg)/len(total_index)
			results[n]['false negative']=round(FN_ratio*100)/100
			results[n]['false positive']=round(FP_ratio*100)/100
			results[n]['correct']=round(correct_ratio*100)/100
		else:
			results[n]['false negative']=None
			results[n]['false positive']=None
			results[n]['correct']=None

	# print 'results',results
	results['y_predicted']=y_predicted
	results['y_true']=y_true
	return results

def plot_saliences(layer,image,sim,model_dict,probe_dict,pt):

	print 'Plotting Feature Activations...'
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
	ax.set_title('Layer %s' %layer)
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def plot_avg_salience(layer,data,sim,model_dict,probe_dict,pt):

	print 'Plotting Average Feature Activations...'
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
	A_avg_images=np.average(normed_activations,axis=0)
	A_std_images=np.std(normed_activations,axis=0)

	#image #, before and after feedback
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.bar(indices, A_avg_images[0], yerr=A_std_images[0], width=1, label='t=0.0', alpha=0.5, color='b')
	ax.bar(indices, A_avg_images[-1], yerr=A_std_images[-1], width=1, label='t=%s' %timesteps, alpha=0.5, color='g')
	ax.set_xlabel('Feature #')
	ax.set_ylabel('Mean Activation')
	ax.set_xlim(0,FMs)
	ax.set_title('Layer %s' %layer)
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def plot_outputs(image,sim,model_dict,probe_dict,pt):

	print 'Plotting Class Activations...'
	sal_vs_time=[] #(FM_activities,timesteps)
	data=sim.data[probe_dict['pre_output']]
	timesteps=pt/0.001
	images=data.shape[0]/timesteps
	classes=data.shape[1]
	# indices=np.arange(0,classes,1) #MNIST
	indices=np.array([0,1,2]) #lines
	activations=data.reshape((images, timesteps, classes))
	normed_activations=np.zeros((activations.shape))
	for im in range(normed_activations.shape[0]):
		for t in range(normed_activations.shape[1]):
			norm=np.sum(activations[im][t])
			for fm in range(normed_activations.shape[2]):
				normed_activations[im][t][fm] = activations[im][t][fm]/norm
	#image number, before and after feedback
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.bar(indices,normed_activations[image][0],width=1,label='t=0.0',alpha=0.5,color='b')
	ax.bar(indices,normed_activations[image][-1],width=1,label='t=%s' %timesteps,alpha=0.5, color='g')
	ax.set_xlabel('Class #')
	ax.set_ylabel('Activation')
	plt.xticks(indices)
	ax.set_title('Output')
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def plot_image(image,layer,FM_number,data,sim,arch_dict,model_dict,probe_dict,pt,n_images):

	print 'Displaying image...'
	img=data[image][0] #expects training data as 4D tuple (images, channels, img_x, img_y)
	weights=arch_dict['conv%s' %layer]['weights'] #expects 4D tuple (FMs, channels, ker_x, ker_y)
	layer_size=model_dict['conv%s' %layer].output.shape_in
	#expects (pt*n_images, np.prod(FMs, channels, ker_x, ker_y))
	#retrieve FM values at beginning and end of presentation time
	probe_start=sim.data[probe_dict['conv%s' %layer]][image*pt/0.001].reshape(layer_size)
	probe_end=sim.data[probe_dict['conv%s' %layer]][(image+1)*pt/0.001-1].reshape(layer_size)
	probe_sal_end=sim.data[probe_dict['sal_F_conv%s' %layer]][(image+1)*pt/0.001-1]
	# sum the FMs, which are already weighed through feedback, to reconstruct the image
	redraw_start=np.average(probe_start,axis=0)
	redraw_end=np.average(probe_end,axis=0)
	# print probe_start[10], probe_end[10], redraw_start, redraw_end

	fig=plt.figure(figsize=(18,12))
	ax=fig.add_subplot(231)
	ax.imshow(weights[FM_number][0], cmap='gray', interpolation='none')
	ax.set_title('Kernel %s' %FM_number)
	ax=fig.add_subplot(232)
	ax.imshow(probe_start[FM_number], cmap='gray', vmin=0, interpolation='none')
	ax.set_title('FM %s Act Begin' %FM_number)
	ax=fig.add_subplot(233)
	ax.imshow(probe_end[FM_number], cmap='gray', vmin=0, interpolation='none')
	ax.set_title('FM %s Act End' %FM_number)

	ax=fig.add_subplot(234)
	ax.imshow(img, cmap='gray', interpolation='none')
	ax.set_title('Image')
	ax=fig.add_subplot(235)
	ax.imshow(redraw_start, cmap='gray', vmin=0, interpolation='none')
	ax.set_title('Sum of FM Acts Begin')
	ax=fig.add_subplot(236)
	ax.imshow(redraw_end, cmap='gray', vmin=0, interpolation='none')
	ax.set_title('Sum of FM Acts End')
	plt.show()

def make_FB_stim_dict(arch_dict):
	FB_dict={}
	stim_dict={}
	for key, info in arch_dict.items():
		if key == 'conv0':
			FB_dict[key] = {
				'competition': 'softmax', #'none', 'softmax'
				'FB_near_type': 'constant', #'none', 'constant'
				'tau_FB_near': 0.001,
				'k_FB_near': 3,
				'FB_far_type': 'none', #'none', 'dense_inverse'
				'tau_FB_far': 0.001,
				'k_FB_far': 0, 
				}
			stim_dict[key] = np.zeros((info['weights'].shape[0]))
			stim_dict[key][0] = 0
		if key == 'conv1':
			FB_dict[key] = {
				'competition': 'softmax',
				'FB_near_type': 'constant',
				'tau_FB_near': 0.001,
				'k_FB_near': 3,
				'FB_far_type': 'dense_inverse',
				'tau_FB_far': 0.001,
				'k_FB_far': 5,
				}
			stim_dict[key] = np.zeros((info['weights'].shape[0]))
			stim_dict[key][0] = 0
		if key == 'conv2':
			FB_dict[key] = {
				'competition': 'softmax',
				'FB_near_type': 'none',
				'tau_FB_near': 0.001,
				'k_FB_near': 10,
				'FB_far_type': 'dense_inverse',
				'tau_FB_far': 0.001,
				'k_FB_far': 5,
				}
			stim_dict[key] = np.zeros((info['weights'].shape[0]))
			stim_dict[key][0] = 0
	return FB_dict, stim_dict

def main():

	# X_train, y_train, X_test, y_test = load_mnist()
	datafile='lines_data_50000.npz'
	labelfile='lines_labels_50000.npz'
	datafile='+_data.npz'
	labelfile='+_labels.npz'

	X_train, y_train, X_test, y_test = load_lines(datafile,labelfile,split=0.8)
	data=(X_test,y_test)

	archfile='mnist_CNN_v5_50000'
	arch_dict = import_keras_json(archfile)
	pt=0.003 #image presentation time, larger = more time for feedback
	frac=0.1 #fraction of dataset to simulate

	FB_dict, stim_dict = make_FB_stim_dict(arch_dict)
	model, model_dict, conn_dict, probe_dict = build_model(arch_dict,data[0],frac,pt,FB_dict,stim_dict)
	
	print 'Running the simulation...'
	sim = nengo.Simulator(model)
	n_images=int(frac*len(data[0]))
	sim.run(pt*n_images)

	results=get_error(sim,model,data[0],data[1],probe_dict,n_images,pt)
	print 'results'
	for key, item in results.items():
		print key, '...', item

	image_num=0
	layer=0
	FM_number=0
	plot_saliences('sal_F_conv1',image_num,sim,model_dict,probe_dict,pt)
	# plot_saliences('sal_F_conv1',image_num,sim,model_dict,probe_dict,pt)
	# plot_saliences('sal_F_conv2',image_num,sim,model_dict,probe_dict,pt)
	# plot_avg_salience('sal_F_conv0',data[0],sim,model_dict,probe_dict,pt)
	# plot_avg_salience('sal_F_conv1',data[0],sim,model_dict,probe_dict,pt)
	# plot_avg_salience('sal_F_conv2',data[0],sim,model_dict,probe_dict,pt)
	# plot_outputs(image_num,sim,model_dict,probe_dict,pt)
	plot_image(image_num,FM_number,layer,data[0],sim,arch_dict,model_dict,probe_dict,pt,n_images)


main()