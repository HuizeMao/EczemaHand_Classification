import numpy as np
import cv2
import os
from keras.models import Model,load_model
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#define the input data
folder = "TestImages"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
train_files = []
for _file in onlyfiles:
    train_files.append(_file)
i = 0
#dimension of data
image_height, image_width, channels = (128,128,3)

#load trained model
model = load_model('ResNet50.h5')
"""input_image = load_img(file)  # this is a PIL image"""
dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
# Convert to Numpy Array
"""input_image = img_to_array(input_image)
input_image = input_image.reshape(1,128,128,3)"""

for _file in train_files:
    try:
        img = load_img(folder + "/" + _file)  # this is a PIL image
        # Convert to Numpy Array
        x = img_to_array(img)
        x = x.reshape(128,128,3)
        # Normalize
        dataset[i] = x
        i+=1
    except(OSError):
        pass
def predict(img):
    pred = model.predict(img)
    print("Prediction of the input image: " + str(pred))
    return pred

prediction = predict(dataset)
