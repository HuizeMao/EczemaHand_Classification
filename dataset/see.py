from matplotlib import pyplot as plt
import numpy as np
import sys
X_train = np.load("X_train.npy")
X_dev = np.load("X_dev.npy")
X_test = np.load("X_test.npy")

Y_dev = np.load('Y_dev.npy')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')
label_t,label_d,label_te = (914,199,39)

print(Y_train)
print("se")
print(Y_dev)
print("se")
print(Y_test)
print("se")

X_train_img = X_train[label_t]
X_dev_img = X_dev[label_d]
X_test_img = X_test[label_te]


print(Y_train[label_t])
print(Y_dev[label_d])
print(Y_test[label_te])


show_img_t = (X_train_img).astype(np.uint8)
show_img_d = (X_dev_img).astype(np.uint8)
show_img_te = (X_test_img).astype(np.uint8)
plt.imshow(show_img_t)
plt.show()

plt.imshow(show_img_d)
plt.show()

plt.imshow(show_img_te)
plt.show()
