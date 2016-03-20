n_filters = np.arange(16,2*n_conv_layers+16,2,dtype=np.float32) # number of convolutional filters to use at layer_i
pool_size = 3 * np.ones(n_conv_layers,dtype=np.float32) # square size of pooling window at layer_i
kernel_size = np.arange(1,2*n_conv_layers,2,dtype=np.float32)[::-1] # square size of kernel at layer_i
dropout_frac = 0.5 * np.ones(shape=n_conv_layers+n_dense_layers,dtype=np.float32)# dropout fraction at layer_i
dense_size = 512 * np.ones(shape=n_dense_layers,dtype=np.float32) # output dimension for dense layers


#test merge with fake u1 layer
model.add_node(Convolution2D(n_filters[0],1,1,
				activation='linear',weights=v_1_matrix,trainable=False),
				name='u_1',input='f_1')
s_1_matrix=[get_s_weights(n_filters[0]),get_s_biases(n_filters[0])]
v_1_matrix=[get_v_weights(n_filters[0]),get_v_biases(0.1,n_filters[0])]


# Network
model = Graph()
model.add_input(name='input', input_shape=image_dim)

model.add_node(Convolution2D(n_filters[0],kernel_size[0],kernel_size[0],
				activation='relu',input_shape=image_dim,trainable=False),
				name='f_1',input='input')
# feedforward weight matrix has 2 elements, of shape 
# (in_dim,out_dim) and (out_dim,) for weights and biases respectively
# layer_1_size=[np.zeros((n_filters[0],n_filters[0],1,1)),np.zeros((n_filters[0]))]
layer_1_size=[np.ones((n_filters[0],n_filters[0],1,1)),np.zeros((n_filters[0]))]
model.add_node(Convolution2D(n_filters[0],1,1,  #1x1 kernels for preserving identity
				activation='relu',weights=layer_1_size,trainable=False), #init='identity',
				name='v_1',input='f_1')

model.add_node(Flatten(),
				name='flat_1', input='v_1')
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
print (history.history)
# model.get_config(verbose=1)
def get_activations(model, input_name, layer_name, X_batch):
    get_activations = theano.function([model.inputs[input_name].input], model.nodes[layer_name].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

model2 = Graph()
model2.add_input(name='input2', input_shape=image_dim)
model2.add_node(Convolution2D(n_filters[0],kernel_size[0],kernel_size[0],
				activation='relu',weights=model.nodes['f_1'].get_weights(),trainable=False),
				name='f_12',input='input2')
model2.add_node(Convolution2D(n_filters[0],1,1,  #1x1 kernels for preserving identity
				activation='relu',weights=layer_1_size,trainable=False), #init='identity',
				name='v_12',input='f_12')
model2.add_node(Flatten(),
				name='flat_12', input='v_12')
model2.add_node(Dense(dense_size[0],
				activation='relu'),
				name='dense_12', input='flat_12')
model2.add_node(Dropout(dropout_frac[2]),
				name='drop_12', input='dense_12')
model2.add_node(Dense(dense_size[1],
				activation='softmax'),
				name='dense_out2', input='drop_12')
model2.add_output(name='output2',input='dense_out2')

my_sgd2 = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
model2.compile(my_sgd2, {'output2':'categorical_crossentropy'})

history2=model2.fit({'input2':X_train[:train_datapoints], 'output2':Y_train[:train_datapoints]},
			batch_size=batch_size, nb_epoch=epochs, shuffle=True,
            validation_data={'input2':X_test[:test_datapoints], 'output2':Y_test[:test_datapoints]})