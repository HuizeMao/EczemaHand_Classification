#import all necessary libraries
import numpy as np
import keras
from keras.losses import binary_crossentropy
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import keras.backend as K
from time import sleep
import cv2
#load dataset
X_train_orig, Y_train, X_CV_orig, Y_CV, X_test_orig, Y_test = load_dataset()

#select a training data image and show
show_img = (X_train_orig[358]).astype(np.uint8)
print(show_img)
plt.imshow(show_img)
plt.show()

# Normalize image vectors
X_train = X_train_orig/255.
X_CV = X_CV_orig/255.
X_test = X_test_orig/255.

m = X_train[0].shape
print("Number of training examples: {}".format(m))
#change the shape of Y outputs
Y_train = np.expand_dims(Y_train,axis=1)
Y_CV = np.expand_dims(Y_CV,axis=1)
Y_test = np.expand_dims(Y_test,axis=1)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of cross validation examples = " + str(X_CV.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("X_CV shape: " + str(X_CV.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_CV shape: " + str(Y_CV.shape))
print ("Y_test shape: " + str(Y_test.shape))

input = input('pause')
#implement identity function that is used as one residual block
def identity_block(X, f, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

#implement convolution block which is also a risidual block but this one change the size of input data
def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    X = Dropout(0.1)(X)

    return X

#create resnet with 50 layers using the risidual blocks implemented above
def ResNet50(input_shape,classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((7, 7))(X_input)

    # Stage 1
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1', padding ='same')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2,2))(X)
    X = Dropout(0.1)(X)

    # Stage 2
    X = Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Dropout(0.1)(X)

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=4, block='b')
    X = identity_block(X, 3, [128,128,512], stage=4, block='c')
    X = identity_block(X, 3, [128,128,512], stage=4, block='d')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='f')

    # Stage 6
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=6, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=6, block='c')

    # Maxpool
    X = MaxPooling2D(pool_size=(2,2), name = 'max_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='fc0' + str(classes))(X)
    X = Dropout(0.4)(X)
    X = Dense(500, activation='relu', name='fc1' + str(classes))(X)
    X = Dropout(0.5)(X)
    X = Dense(100, activation='relu', name='fc2' + str(classes))(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='sigmoid', name='fc3' + str(classes))(X)
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#create model
model = load_model("ResNet50_7.h5")
#model = load_model('ResNet50_6.h5')
#compile
model.compile(loss='binary_crossentropy',
            optimizer = keras.optimizers.SGD(lr=0.03, decay=0, momentum=0, nesterov=False),
            metrics=['acc'])

#fit
history = model.fit(X_train, Y_train, batch_size = 2315,epochs = 100,verbose = 1, validation_data = (X_CV,Y_CV),shuffle=True)
#save model
model.save('ResNet50_8.h5')

#evaluate
preds = model.evaluate(X_CV, Y_CV)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#summary
model.summary()

#model history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
