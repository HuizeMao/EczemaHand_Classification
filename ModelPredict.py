import numpy as np
import cv2
from keras.models import Model,load_model

input_image = cv2.imread('Input.png')
input_image = np.expand_dims(input_image,axis=0)
input_image = input_image/255.
model = load_model('ResNet50_8.h5')
#model.get_weigts()

def predict(img):
    pred = model.predict(img)
    if pred <= 0.5:
        pred = 0
    else:
        pred = 1
    print("Prediction of the input image: " + str(pred))
    return pred

prediction = predict(input_image)
