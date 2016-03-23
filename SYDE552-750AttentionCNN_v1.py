# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
from scipy import signal
import json
import nengo
import matplotlib.pyplot as plt
# from keras.models import model_from_json
# from keras.datasets import cifar100 

# filename='cifar100_v5_n_layer_pool=4'
# arch = model_from_json(open(filename+'_model.json').read())
# arch.load_weights(filename+'_weights.h5')
# arch_dict={}

# for node in arch.nodes:
# 	name=str(arch.nodes[node].get_config()['custom_name'])
# 	config = arch.nodes[node].get_config()
# 	# print arch.nodes[node].get_input(train=False).name
# 	arch_dict[name] = config

# #import an image
# (X_train, y_train), (X_test, y_test) = cifar100.load_data()

# img_x, img_y, img_z = X_train.shape[2], X_train.shape[3], X_train.shape[1]
# samples_train = X_train.shape[0]
# samples_test = X_test.shape[0]
# image=X_train[0]
# image_test=np.random.rand(img_x, img_y)

img_x, img_y, img_z = 32,32,3

image_test=np.random.rand(img_z, img_x, img_y)
weight_test=np.random.rand(3,16,7,7)


model = nengo.Network('CNN with Attention')

with model:

	image_stim_dict={}
	for c in range(img_z):
		image_stim_dict['color_%s' %c] = nengo.Node(output=lambda t: image_test[c].ravel())

	#represent image pixels in 3 ensemble arrays
	ens_dict={}
	conn_dict={}
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
	def conv(A):
		#make 1D input into 2D array
		input_2d=np.zeros((img_x-1,img_y-1))
		for x in range(img_x-1):
			for y in range(img_y-1):
				input_2d[x][y]=A[x*(img_x-1)+y]
		#convolve, then flatten for input to next ensemble array
		output=signal.convolve2d(input_2d,kernel,mode='valid').ravel()
		return output

	#connect the first convolutional layer to the image ensemble
	FMs_in=weight_test.shape[0]
	FMs_here=weight_test.shape[1]
	ker_x=weight_test.shape[2]
	ker_y=weight_test.shape[3]
	for i in range(FMs_here): #feature map i of layer_n
		ens_name='ens_lyr_%s_ft_%s' %(0,i)
		ens_dict[ens_name] = nengo.networks.EnsembleArray(n_neurons=3,
									n_ensembles=(img_x-ker_x)*(img_y-ker_y),
									ens_dimensions=1,
									neuron_type=nengo.Direct())
		for j in range(FMs_in): #connection from feature map j of layer_(n-1)
			input_name='ens_img_ft_%s' %j
			conn_name='conn_lyr_%s_ft_%s_to_lyr%s_ft_%s' %(-1,j,0,i)
			kernel=weight_test[j][i]
			T_test=np.ones((625,1024))
			conn_dict[conn_name] = nengo.Connection(ens_dict[input_name].output,
												ens_dict[ens_name].input, 
												transform=T_test)
												# function=conv) #passthrough node error
												#, synapse=pstc)

	probe_0 = nengo.Probe(ens_dict['ens_img_ft_0'].output)
	probe_1 = nengo.Probe(ens_dict['ens_lyr_0_ft_1'].output)

sim = nengo.Simulator(model)
sim.run(0.1)
t=sim.trange()

fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(211)
ax.plot(t,sim.data[probe_0])
ax=fig.add_subplot(212)
ax.plot(t,sim.data[probe_1])
ax.set_xlabel('time')
ax.set_ylabel('ens_image')
plt.show()
