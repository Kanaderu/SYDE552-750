# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
from scipy import signal
import json
import nengo
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 20

from keras.models import model_from_json
from keras.datasets import cifar100 

filename='cifar100_v5_n_layer_test'
arch = model_from_json(open(filename+'_model.json').read())
arch.load_weights(filename+'_weights.h5')

arch_dict={}
# print dir(arch)
# print dir(arch.nodes)
# print dir(arch.node_config)
s=0
for n in range(len(arch.nodes)):
	name=str(arch.nodes.items()[n][0])
	arch_dict[name] = {} #one dictionary per layer
	layer_type=str(arch.nodes[name].get_config()['name'])
	arch_dict[name]['type'] = layer_type
	arch_dict[name]['input'] = arch.node_config[n]['input']
	# arch_dict[name]['input_shape'] = arch.nodes[name].get_config()['input_shape']
	if (layer_type == 'Convolution2D' or layer_type == 'Dropout' or layer_type == 'Dense'):
		arch_dict[name]['weights'] = arch.get_weights()[s]
		s+=1

#import an image
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
img_x, img_y, img_z = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
image=X_train[0]

# img_x, img_y, img_z = 32,32,3
# image_test=np.random.rand(img_z, img_x, img_y)
# weight_test=np.random.rand(3,16,7,7)

ens_dict={}
conn_dict={}
image_stim_dict={}
size_array=[img_x] #for this particular network
for i in np.arange(0,5):
	kernel_size=arch_dict['conv%s'%i]['weights'].shape[-1]
	size_array.append(size_array[i]-kernel_size+1)
for i in np.arange(0,3):
	dense_size=arch_dict['dense%s'%i]['weights'].shape[-1]
	size_array.append(dense_size)
print 'size array ', size_array

model = nengo.Network('CNN with Attention')

with model:

	for c in range(img_z):
		image_stim_dict['color_%s' %c] = nengo.Node(output=lambda t: image[c].ravel())

	#represent image pixels in 3 ensemble arrays
	for i in range(img_z):
		input_name='color_%s' %i
		ens_name='ens_img_ft_%s' %i
		conn_name='conn_img_ft_%s' %i
		img_stim_ft = image_stim_dict[input_name]
		ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
									n_ensembles=img_x*img_y,
									ens_dimensions=1,
									neuron_type=nengo.Direct())
		conn_dict[conn_name] = nengo.Connection(img_stim_ft,
									ens_dict[ens_name].input)

	#define convolution function between layers
	def conv(t,A):
		A_square=A.reshape((np.sqrt(len(A))),np.sqrt(len(A)))
		conv=signal.convolve2d(A_square,kernel,mode='valid').ravel()
		return conv

	#connect the first convolutional layer to the image ensemble
	W=arch_dict['conv%s'%0]['weights']
	FMs_here, FMs_in, ker_x, ker_y = W.shape
	for i in range(FMs_here): #feature map i of layer_n
		ens_name='ens_lyr_%s_ft_%s' %(0,i)
		ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
									n_ensembles=size_array[1]**2,
									ens_dimensions=1,
									neuron_type=nengo.Direct())
		for j in range(FMs_in): #connection from feature map j of layer_(n-1)
			input_name='ens_img_ft_%s' %j
			conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(-1,j,0,i)
			kernel=W[i][j]
			inter=nengo.Node(output=conv,size_in=size_array[0]**2)
			conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,inter)
			conn_dict[conn_name+'pass_node'] = nengo.Connection(inter,ens_dict[ens_name].input)

	W=arch_dict['conv%s'%1]['weights']
	FMs_here, FMs_in, ker_x, ker_y = W.shape
	for i in range(FMs_here):
		ens_name='ens_lyr_%s_ft_%s' %(1,i)
		ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
									n_ensembles=size_array[1+1]**2,
									ens_dimensions=1,
									neuron_type=nengo.Direct())
		for j in range(FMs_in):
			input_name='ens_lyr_%s_ft_%s' %(1-1,j)
			conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(1-1,j,1,i)
			kernel=W[i][j]
			inter=nengo.Node(output=conv,size_in=size_array[1]**2)
			conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,inter)
			conn_dict[conn_name+'pass_node'] = nengo.Connection(inter,ens_dict[ens_name].input)

	# W=arch_dict['conv%s'%2]['weights']
	# FMs_here, FMs_in, ker_x, ker_y = W.shape
	# for i in range(FMs_here):
	# 	ens_name='ens_lyr_%s_ft_%s' %(2,i)
	# 	ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
	# 								n_ensembles=size_array[1+2]**2,
	# 								ens_dimensions=1,
	# 								neuron_type=nengo.Direct())
	# 	for j in range(FMs_in):
	# 		input_name='ens_lyr_%s_ft_%s' %(2-1,j)
	# 		conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(2-1,j,2,i)
	# 		kernel=W[i][j]
	# 		inter=nengo.Node(output=conv,size_in=size_array[2]**2)
	# 		conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,inter)
	# 		conn_dict[conn_name+'pass_node'] = nengo.Connection(inter,ens_dict[ens_name].input)		
	
	# W=arch_dict['conv%s'%3]['weights']
	# FMs_here, FMs_in, ker_x, ker_y = W.shape
	# for i in range(FMs_here):
	# 	ens_name='ens_lyr_%s_ft_%s' %(3,i)
	# 	ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
	# 								n_ensembles=size_array[3+1]**2,
	# 								ens_dimensions=1,
	# 								neuron_type=nengo.Direct())
	# 	for j in range(FMs_in):
	# 		input_name='ens_lyr_%s_ft_%s' %(3-1,j)
	# 		conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(3-1,j,1,i)
	# 		kernel=W[i][j]
	# 		inter=nengo.Node(output=conv,size_in=size_array[3]**2)
	# 		conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,inter)
	# 		conn_dict[conn_name+'pass_node'] = nengo.Connection(inter,ens_dict[ens_name].input)

	# # build the rest of the convolutional stack
	# for n in np.arange(1,5):
	# 	print n
	# 	W=arch_dict['conv%s'%n]['weights']
	# 	FMs_here, FMs_in, ker_x, ker_y = W.shape
	# 	for i in range(FMs_here): #feature map i of layer_n
	# 		ens_name='ens_lyr_%s_ft_%s' %(n,i)
	# 		ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
	# 									n_ensembles=size_array[1+n]**2,
	# 									ens_dimensions=1,
	# 									neuron_type=nengo.Direct())
	# 		for j in range(FMs_in): #connection from feature map j of layer_(n-1)
	# 			input_name='ens_lyr_%s_ft_%s' %(n-1,j)
	# 			conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(n-1,j,n,i)
	# 			kernel=W[i][j]
	# 			inter=nengo.Node(output=conv,size_in=size_array[n]**2)
	# 			conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,inter)
	# 			conn_dict[conn_name+'pass_node'] = nengo.Connection(inter,ens_dict[ens_name].input)

	#flatten the last convolutional layer
	# ens_name='ens_lyr_flatten'
	# FMs_to_flatten=FMs_here
	# nodes_per_FM=size_array[5]**2
	# flatten_nodes=FMs_to_flatten*nodes_per_FM #feature maps in last conv layer * units in each FM
	# ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
	# 								n_ensembles=flatten_nodes,
	# 								ens_dimensions=1,
	# 								neuron_type=nengo.Direct())
	# for i in range(FMs_to_flatten):
	# 	input_name='ens_lyr_%s_ft_%s' %(4,i)
	# 	conn_name='conn_lyr_%s_ft_%s_to_flatten' %(4,i)
	# 	conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,
	# 											ens_dict[ens_name].input
	# 											[i*nodes_per_FM:(i+1)*nodes_per_FM])

 # 	#connect the flattened layer to the first dense layer
	# input_name='ens_lyr_flatten'
	# ens_name='ens_lyr_dense_%s' %0
	# conn_name='conn_lyr_flatten_to_dense_%s' %0
	# weights=arch_dict['dense%s'%0]['weights'].T
	# print weights.shape
	# ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
	# 								n_ensembles=size_array[6],
	# 								ens_dimensions=1,
	# 								neuron_type=nengo.Direct())
	# conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,
	# 								ens_dict[ens_name].input,
	# 								transform=weights)

	probe_0 = nengo.Probe(ens_dict['ens_img_ft_0'].output)
	probe_1 = nengo.Probe(ens_dict['ens_lyr_1_ft_0'].output)
	# probe_2 = nengo.Probe(ens_dict['ens_lyr_dense_0'].output)

sim = nengo.Simulator(model)
sim.run(0.05)
t=sim.trange()
print t

fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(211)
ax.plot(t,sim.data[probe_0])
ax=fig.add_subplot(212)
ax.plot(t,sim.data[probe_1])
ax.set_xlabel('time')
plt.show()