from keras.models import Model,load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
#499 layers(counting dropout individually)
model = load_model("Inception_2.h5")
img = cv2.imread('image.png')

img = np.expand_dims(img,axis=0)
img = img/255.

layer_outputs = [layer.output for layer in model.layers[1:]] #exclude input layer
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img)

def display_activation(activations, col_size, row_size, act_index): # act_index is the number of layer in the model
    activation = activations[act_index]
    activation_index= 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            try:
                ax[row][col].imshow(activation[0, :, :, activation_index])
                activation_index += 1
            except(IndexError):
                pass
display_activation(activations, 10, 10, -7)
plt.show()
# 0 3 9
