import cv2
from PIL import Image, ImageFile
import os, sys
from PIL import Image
import numpy as np
from time import time
from time import sleep
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#define folder directory and file locations

folder = "Images"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#define train files
train_files = []
i=0
for _file in onlyfiles:
    train_files.append(_file)

print("Files in train_files: %d" % len(train_files))

channels = 3
image_height = 128
image_width = 128

dataset_diseased_hand = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
for _file in train_files:
    try:
        img = load_img(folder + "/" + _file)  # this is a PIL image
        # Convert to Numpy Array
        x = img_to_array(img)
        x = x.reshape(128,128,3)
        # Normalize
        dataset_diseased_hand[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    except(ValueError,OSError):
        pass

np.save("images.npy", dataset_diseased_hand)
print("All images to array!")
