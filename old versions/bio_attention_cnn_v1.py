# Peter Duggins, psipeter@gmail.com
# SYDE 552/750
# Final Project
# Winter 2016

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Graph
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np
import theano

# Learning parameters
batch_size = 40
classes = 10
epochs = 2
train_datapoints=50
test_datapoints=25
learning_rate=0.01
decay=1e-6
momentum=0.9
nesterov=True

# Network Parameters
n_filters = [16, 24] # number of convolutional filters to use at layer_i
pool_size = [2, 2] # square size of pooling window at layer_i
kernel_size = [3, 3] # square size of kernel at layer_i
dropout_frac = [0.25, 0.25, 0.5] # dropout fraction at layer_i
dense_size = [512, classes] # output dimension for dense layers

# CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
shapex, shapey, shapez = X_train.shape[2], X_train.shape[3], X_train.shape[1]
samples_train = X_train.shape[0]
samples_test = X_test.shape[0]
image_dim=(shapez,shapex,shapey)

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Feedforward weights workarounds
# for inhibitory interneurons u and v
def get_u_weights(n_filters):
	weights=[]
	for i in range(n_filters):
		weights_i=[]
		for j in range(n_filters):
			if i==j:
				weights_i.append(-1.0*np.ones((1,1)))
			else:
				weights_i.append(np.zeros((1,1)))
		weights.append(weights_i)
	return np.array(weights)

# for spreading units
def get_s_weights(n_filters):
	weights=[]
	for i in range(n_filters):
		weights_i=[]
		#first half of concatenated inputs is from v population
		for j in range(n_filters):
			if i==j:
				weights_i.append(1.0*np.ones((1,1))) #positive weights
			else:
				weights_i.append(np.zeros((1,1)))
		#second half of concatenated inputs is from u population
		for j in range(n_filters):
			if i==j:
				weights_i.append(-0.5*np.ones((1,1))) #negative weights
			else:
				weights_i.append(np.zeros((1,1)))
		weights.append(weights_i)
	return np.array(weights)

# for pooling units
def get_p_weights(n_filters,x_dim,y_dim,ISS): #ISS=inhibitory surround size
	weights=[]
	for i in range(n_filters):
		weights_i=[]
		#first half of concatenated inputs is from s population
		for j in range(n_filters):
			if i==j:
				#make a ISSxISS kernel that implements supressive surround
				#such that s_ij's neighbors have negative weight.
				#default: UDLF neighbor weights=-1, no mutual excitation
				weights_i.append(np.array(	[[0,-1,0],
											[-1,0,-1],
											[0,-1,0]]  ))
			else:
				#make an empty 3x3 kernel
				weights_i.append(np.zeros((ISS,ISS)))
		#second half of concatenated inputs is from b population
		weights_i.append(-1.0*np.ones((1,1))) #salience inhibition 
		weights.append(weights_i)
	return np.array(weights)

# for feedforward salience units
def get_f_weights(n_filters,x_dim,y_dim):
	weights=[np.ones((x_dim,y_dim)) for i in range(n_filters)]
	return np.array(weights)

# TEMP for feedback salience units
def get_b_weights(x_dim,y_dim):
	weights=np.ones((x_dim,y_dim))
	return np.array(weights)

#for all units, used in interneurons and pooling
def get_biases(bias_value,n_filters):
	biases=np.full(shape=n_filters, fill_value=bias_value)
	return biases


# Network
model = Graph()

#input and convolution
model.add_input(name='input', input_shape=image_dim)

model.add_node(Convolution2D(n_filters[0],kernel_size[0],kernel_size[0],
				activation='relu',input_shape=image_dim,trainable=False),
				name='in_1',input='input')


# spreading and pooling layer
#input shape = (16,16,1,1), output_shape = (16,16,1,1)
u_1_matrix=[get_u_weights(n_filters[0]),get_biases(0.1,n_filters[0])]
model.add_node(Convolution2D(n_filters[0],1,1,
				activation='linear',weights=u_1_matrix,trainable=False),
				name='u_1',input='in_1')

#test merge with fake v_1 layer
# model.add_node(Convolution2D(n_filters[0],1,1,
# 				activation='linear',weights=v_1_matrix,trainable=False),
# 				name='v_1',input='f_1')

#input shape = (16,32,1,1), output_shape = (16,16,1,1)
s_1_matrix=[get_s_weights(n_filters[0]),get_biases(0.0,n_filters[0])]
model.add_node(Convolution2D(n_filters[0],1,1,
				activation='linear', weights=s_1_matrix, trainable=False),
				name='s_1',inputs=['v_1','u_1'],
				merge_mode='concat',concat_axis=1)

#input shape = (16,16,1,1), output_shape = (16,16,1,1)
v_1_matrix=[get_u_weights(n_filters[0]),get_biases(1,n_filters[0])]
model.add_node(Convolution2D(n_filters[0],1,1,
				activation='hard_sigmoid',weights=v_1_matrix,trainable=False),
				name='v_1',input='p_1')

#input shape = (16,17,1,1), output_shape = (16,16,1,1)
x_dim, y_dim, ISS = (n_filters[0]-kernel_size[0]+1),(n_filters[0]-kernel_size[0]+1), 3
p_1_matrix=[get_p_weights(n_filters[0],x_dim,y_dim,ISS),get_biases(1.0,n_filters[0])]
print (p_1_matrix[0].shape, p_1_matrix[1].shape)
model.add_node(Convolution2D(n_filters[0],1,1,
				activation='linear', weights=p_1_matrix, trainable=False),
				name='p_1',inputs=['s_1','b_1'],
				merge_mode='concat', concat_axis=1)


# Salience layer
# each f unit sums activation of all s_ij from each previous feature map
#input shape = (1,16,30,30), output_shape = (1,1,16,16)
f_1_matrix=[get_f_weights(n_filters[0],x_dim,y_dim),get_biases(0.0,n_filters[0])]
model.add_node(Convolution2D(1,n_filters[0],n_filters[0],
				activation='linear',weights=f_1_matrix,trainable=False),
				name='f_1',input='s_1')

# temporary b unit copies f input
#input shape = (1,1,16,16), output_shape = (16,1,1,1)
b_1_matrix=[get_b_weights(x_dim,y_dim),get_biases(0.0,x_dim)]
model.add_node(Convolution2D(1,1,1,
				activation='linear',weights=b_1_matrix,trainable=False),
				name='b_1',input='f_1')



model.add_node(Flatten(),
				name='flat_1', input='s_1')
model.add_node(Dense(dense_size[0],
				activation='relu'),
				name='dense_1', input='flat_1')
model.add_node(Dropout(dropout_frac[2]),
				name='drop_1', input='dense_1')
model.add_node(Dense(dense_size[1],
				activation='softmax'),
				name='dense_out', input='drop_1')

model.add_output(name='output',input='dense_out')

# Optimize and compile
my_sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
model.compile(my_sgd, {'output':'categorical_crossentropy'})

# Train
history=model.fit({'input':X_train[:train_datapoints], 'output':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input':X_test[:test_datapoints], 'output':Y_test[:test_datapoints]})

# Print results and network configuration
# print (history.history)
# model.get_config(verbose=1)
def get_inputs(model, input_name, layer_name, X_batch):
    get_inputs = theano.function([model.inputs[input_name].input], model.nodes[layer_name].get_input(train=False), allow_input_downcast=True)
    my_inputs = get_inputs(X_batch)
    return my_inputs

def get_outputs(model, input_name, layer_name, X_batch):
    get_outputs = theano.function([model.inputs[input_name].input], model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    my_outputs = get_outputs(X_batch)
    return my_outputs


f_1_input=get_inputs(model,'input','f_1',X_test[:test_datapoints])
v_1_input=get_inputs(model,'input','v_1',X_test[:test_datapoints])
u_1_input=get_inputs(model,'input','u_1',X_test[:test_datapoints])
s_1_input=get_inputs(model,'input','s_1',X_test[:test_datapoints])
print ('f_1 input shape, sum',f_1_input.shape,f_1_input.sum())
print ('v_1 input shape, sum',v_1_input.shape,v_1_input.sum())
print ('u_1 input shape, sum',u_1_input.shape,u_1_input.sum())
print ('s_1 input shape, sum',s_1_input.shape,s_1_input.sum())


f_1_output=get_outputs(model,'input','f_1',X_test[:test_datapoints])
v_1_output=get_outputs(model,'input','v_1',X_test[:test_datapoints])
u_1_output=get_outputs(model,'input','u_1',X_test[:test_datapoints])
s_1_output=get_outputs(model,'input','s_1',X_test[:test_datapoints])
print ('f_1 output shape, sum',f_1_output.shape,f_1_output.sum())
print ('v_1 output shape, sum',v_1_output.shape,v_1_output.sum())
print ('u_1 output shape, sum',u_1_output.shape,u_1_output.sum())
print ('s_1 output shape, sum',s_1_output.shape,s_1_output.sum())

# print ('s_1 weights',np.array(model.nodes['s_1'].get_weights()).shape,
# 	np.array(model.nodes['s_1'].get_weights()).sum())


