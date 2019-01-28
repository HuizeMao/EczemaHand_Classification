import cv2
from PIL import Image, ImageFile
import os, sys
from PIL import Image
import numpy as np
from time import time
from time import sleep
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

"""X_train = np.load('dataset/X_train.npy')
Y_train = np.load('dataset/Y_train.npy')
X_dev = np.load('dataset/X_dev.npy')
Y_dev = np.load('dataset/Y_dev.npy')
X_test = np.load('dataset/X_test.npy')
Y_test = np.load('dataset/Y_test.npy')"""

#Eczema hand Train:915 CV:233 test:16
normalhand = np.load('dataset/normalhand.npy')
m = ImageArray.shape[0]
print('normal hand m: ' + str(m))
print(ImageArray.shape)

#Normal Hand Train:1342, CV:447, test:447
eczemahand = np.load('dataset/EczemaHandDataset.npy')
m2 = SecondImageArray.shape[0]
print('EczemaHand dataset shape: ' + str(m2))
print(SecondImageArray.shape)

#function that splits datasets into three sets
def split_into_three_sets_eczema(dataset,TrainSetNum,CVSetNum,TestSetNum,classes):
    X_train = dataset[0:TrainSetNum,:,:,:] #input of training set m*n
    Y_train = np.ones((TrainSetNum))
    X_dev = dataset[TrainSetNum:TrainSetNum+CVSetNum,:,:,:] #
    Y_dev = np.ones((CVSetNum)) #
    X_test = dataset[TrainSetNum+CVSetNum:,:,:,:]
    Y_test = np.ones((TestSetNum))
    return(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

X_train,Y_train,X_dev,Y_dev,X_test,Y_test = split_into_three_sets_normal(eczemahand,915,233,16,2)

def split_into_three_sets_normal(dataset,TrainSetNum,CVSetNum,TestSetNum,classes):
    X_train = dataset[0:TrainSetNum,:,:,:] #input of training set m*n
    Y_train = np.zeros((TrainSetNum))
    X_dev = dataset[TrainSetNum:TrainSetNum+CVSetNum,:,:,:] #
    Y_dev = np.zeros((CVSetNum)) #
    X_test = dataset[TrainSetNum+CVSetNum:,:,:,:]
    Y_test = np.zeros((TestSetNum))
    return(X_train,Y_train,X_dev,Y_dev,X_test,Y_test)

X_train2,Y_train2,X_dev2,Y_dev2,X_test2,Y_test2 = split_into_three_sets_normal(normalhand,1342,447,447,2)

X_train = np.append(X_train,X_train2,axis = 0)
X_dev  = np.append(X_dev,X_dev2,axis = 0)
X_test = np.append(X_test,X_test2,axis = 0)
Y_train = np.append(Y_train,Y_train2,axis = 0)
Y_dev = np.append(Y_dev,Y_dev2,axis = 0)
Y_test = np.append(Y_test,Y_test2,axis = 0)


print("X_train shape:" + str(X_train.shape))
print("X_dev shape: " + str(X_dev.shape))
print("X_test shape:" + str(X_test.shape))
print("Y_train shape:" + str(Y_train.shape))
print("Y_dev shape" + str(Y_dev.shape))
print("Y_test shape:" + str(Y_test.shape))

np.save('dataset/X_train.npy',X_train)
np.save('dataset/Y_train.npy',Y_train)
np.save('dataset/X_dev.npy',X_dev)
np.save('dataset/Y_dev.npy',Y_dev)
np.save('dataset/X_test.npy',X_test)
np.save('dataset/Y_test.npy',Y_test)
