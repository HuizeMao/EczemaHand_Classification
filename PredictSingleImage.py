import numpy as np
import cv2
from keras.models import Model,load_model
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#define the input file
file = "InputImage.jpg"
#load trained model
model = load_model('ResNet50.h5')

input_image = load_img(file)  # this is a PIL image

# Convert to Numpy Array
input_image = img_to_array(input_image)
input_image = input_image.reshape(1,128,128,3)

def predict(img):
    pred = model.predict(img)
    print("Prediction of the input image: " + str(pred))
    return pred

prediction = predict(input_image)
