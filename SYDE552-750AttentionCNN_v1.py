# Peter Duggins
# SYDE 552/750
# Final Project
# Nengo Attention CNN

import numpy as np
import json
import nengo
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.datasets import cifar100 

filename='cifar100_v5_n_layer_pool=4'
arch = model_from_json(open(filename+'_model.json').read())
arch.load_weights(filename+'_weights.h5')
arch_dict={}

for node in arch.nodes:
	name=str(arch.nodes[node].get_config()['custom_name'])
	config = arch.nodes[node].get_config()
	# print arch.nodes[node].get_input(train=False).name
	arch_dict[name] = config

#import an image
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

shapex, shapey, shapez = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
image=X_train[0]
image_test=np.random.rand(shapex, shapey)

model = nengo.Network('CNN with Attention')

with model:

	image_stim = nengo.Node(output=lambda t: np.ravel(image))

	ens_dict={}
	conn_dict={}
	for i in range(shapez):
		img_stim_ft = image_stim[i*shapex*shapey:(i+1)*i*shapex*shapey]
		print img_stim_ft
		ens_dict['ens_img_ft_%s' %i] = nengo.networks.EnsembleArray(n_neurons=3,
											n_ensembles=shapex*shapey,
											ens_dimensions=1,
											neuron_type=nengo.Direct())
		conn_dict['conn_img_ft_%s' %i] = nengo.Connection(img_stim_ft,
											ens_dict['ens_img_ft_%s' %i].input)

	probe_ens_img = nengo.Probe(ens_dict['ens_img_ft_0'].output)

sim = nengo.Simulator(model)
sim.run(0.1)
t=sim.trange()

print sim.data[probe_ens_img].shape
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
ax.plot(t,sim.data[probe_ens_img])
ax.set_xlabel('time')
ax.set_ylabel('ens_image')
plt.show()
