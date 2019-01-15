#import all necessary libraries
import numpy as np
import h5py
import cv2
import keras
from keras.models import Model, load_model

#load test set labeled pairs
X_test = np.load('dataset/X_test.npy')
Y_test = np.load('dataset/Y_test.npy')
Y_test = np.array([Y_test]).T
print("Dataset loaded")
print("Input data shape: " + str(X_test.shape))
print("labeled data(Y) shape: " + str(Y_test.shape))

#load the trained model
model = load_model('ResNet50.h5')
print("Trained model loaded.")

#evaluate the accuracy of the model
TestAccuracy = model.evaluate(X_test, Y_test)
print ("Loss = " + str(TestAccuracy[0]))
print ("Test Accuracy = " + str(TestAccuracy[1]))
model.summary()
