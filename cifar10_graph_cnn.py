#courtesy https://github.com/fchollet/keras/issues/762

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

'''#####################################################################
 Defining varibales 
#####################################################################'''

batch_size = 32
nb_classes = 10
nb_epoch = 2
data_augmentation = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 64]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the CIFAR10 images are RGB
image_depth = 3
image_dimensions=(image_depth,shapex,shapey)



'''#####################################################################
 Loading the data and labels
#####################################################################
dirname = "cifar-10-batches-py"
origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
path = get_file(dirname, origin=origin, untar=True)

nb_test_samples = 10000
nb_train_samples = 50000

X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
y_train = np.zeros((nb_train_samples,), dtype="uint8")

for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    data, labels = load_batch(fpath)
    X_train[(i-1)*10000:i*10000, :, :, :] = data
    y_train[(i-1)*10000:i*10000] = labels

fpath = os.path.join(path, 'test_batch')
X_test, y_test = load_batch(fpath)

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)'''

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

'''#####################################################################
 Graph model
#####################################################################'''
model =  Graph()
# Load the input
model.add_input(name='input1', input_shape=image_dimensions)
# Convolution Neural Network architecture (5 convolution layers, 3 pooling layers)

model.add_node(Convolution2D(nb_filters[0], nb_conv[0], nb_conv[0], activation='relu', border_mode='valid'), name='conv2', input='input1')
model.add_node(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0], activation='relu', border_mode='valid'), name='conv3', input='conv2')
model.add_node(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])), name='pool1', input='conv3')

model.add_node(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], activation='relu', border_mode='valid'), name='conv4', input='pool1')
model.add_node(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1], activation='relu', border_mode='valid'), name='conv5', input='conv4')
model.add_node(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])), name='pool2', input='conv5')

model.add_node(Flatten(), name='flatten', input='pool2')

model.add_node(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) * (shapey / nb_pool[0] / nb_pool[1]), 512, activation='relu', init='uniform'),  name='dense1', input='flatten')
model.add_node(Dense(512, nb_classes, activation='softmax', init='uniform'), name='dense2', input='dense1')

model.add_output(name='output1', input='dense2', merge_mode='sum')
model.compile('sgd', {'output1':'categorical_crossentropy'})
model.get_config(verbose=1)

model.fit({'input1':X_train, 'output1':Y_train},batch_size=batch_size, nb_epoch=nb_epoch)

#model.predict({'input1':X_test})