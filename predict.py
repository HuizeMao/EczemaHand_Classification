from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2

"""
cv2
#read image and reshape
img = cv2.imread('InputImage.png')
img = cv2.resize(img,(128,128))
img = np.expand_dims(img,axis=0)"""


img = load_img('InputImage.png')
img = img_to_array(img)
img = img.reshape(1,128,128,3)
#load model and predict
model = load_model('ResNet50.h5')
classes = model.predict(img)

print (classes)
