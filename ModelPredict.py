import numpy as np
import cv2
from keras.models import Model,load_model

model = load_model('ResNet50.h5')
#model.get_weigts()
input_image = cv2.imread('InputImage.jpg')

def predict(img):
    pred = model.predict(img)
    print("Prediction of the input image: " + str(pred))
    return pred

prediction = predict(input_image)
