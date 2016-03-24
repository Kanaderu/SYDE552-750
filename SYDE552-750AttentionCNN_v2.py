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

#import images
# MNIST data
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

in_filename='mnist_CNN_v1'
with open(in_filename+"_arch.json") as datafile:    
    arch_dict=load(datafile)

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


# building my own network
image=X_train[0]
img_x, img_y, img_z = X_train.shape[2], X_train.shape[3], X_train.shape[1]
W=arch_dict['conv%s'%0]['weights']
FMs_here, FMs_in, ker_x, ker_y = W.shape
pool_size=2

model = nengo.Network()
with model:
    input_img = nengo.Node(image.ravel())
    img_probe=nengo.Probe(input_img)

    conv0 = nengo.Node(Conv2d(image.shape, W, biases=None, padding=0))
    conv_probe = nengo.Probe(conv0)
    nengo.Connection(input_img, conv0, synapse=None)

    pool0 = nengo.Node(Pool2d(conv0.output.shape_out, pool_size,stride=None, kind='avg'))
    pool_probe = nengo.Probe(pool0)
    nengo.Connection(conv0, pool0, synapse=None)

    test0 = nengo.Node(Pool2d(pool0.output.shape_out, 1, stride=None, kind='avg'))
    test_probe = nengo.Probe(test0)
    nengo.Connection(pool0, test0, synapse=None)

    stim0=nengo.Node(output=lambda t: np.ones(test0.output.shape_in).ravel())
    nengo.Connection(stim0,test0)


sim = nengo.Simulator(model)
sim.run(0.01)
p1 = sim.data[img_probe][-1]
p2 = sim.data[conv_probe][-1]
p3 = sim.data[pool_probe].sum()
p4 = sim.data[test_probe].sum()
print p1.shape, p2.shape, p3, p4