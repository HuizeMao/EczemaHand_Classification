from keras.models import Model,load_model
import numpy as np
import cv2

img = cv2.imread('image.png')
img = np.expand_dims(img,axis=0)
img = img/255.

#ResNet = load_model('ResNet50_8.h5')
InceptionModel = load_model('Inception_2.h5')

def predict(img):
    #pred_res = ResNet.predict(img)
    pred_inc = InceptionModel.predict(img)
    print(pred_inc)
    #pred = "mix"
    if pred_inc <= 0.5:
        pred_inc = 0
    else:
        pred_inc = 1
    print("Prediction of the input image by ResNet: " + str(pred_inc))

    return pred_inc

prediction = predict(img)
