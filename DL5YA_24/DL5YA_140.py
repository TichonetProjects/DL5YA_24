# 140 All Exercises
# Gad Lidror
import numpy as np
import h5py
import matplotlib.pyplot as plt

from unit10 import c1w5_utils as u10

from DL3_Comp import *



# additions for stage 2
import matplotlib
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from PIL import Image, ImageOps

print (" --- Stage 1 ---")
print (" ---------------")

# set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

print (" --- Targil 140.1.1 ---")

np.random.seed(1)
softmax_layer = DLLayer ("Softmax 1", 3,(4,),"softmax","random")
A_prev = np.random.randn(4, 5)
A = softmax_layer.forward_propagation(A_prev, False)
dA = A
dA_prev = softmax_layer.backward_propagation(dA)
print("A:\n",A)
print("dA_prev:\n",dA_prev)


print (" --- Targil 140.1.2 ---")

np.random.seed(2)
softmax_layer = DLLayer("Softmax 2", 3,(4,),"softmax","random")
print("W before:\n",softmax_layer.W)
print("b before:\n",softmax_layer.b)
model = DLModel()
model.add(softmax_layer)
model.compile("categorical_cross_entropy")
X = np.random.randn(4, 5)
Y = np.random.rand(3, 5)
Y = np.where(Y==Y.max(axis=0),1,0)
cost = model.train(X,Y,1)
print("cost:",cost[0])
print("W after:\n",softmax_layer.W)
print("b after:\n",softmax_layer.b)


print (" --- Targil 140.1.3 ---")
np.random.seed(3)
softmax_layer = DLLayer("Softmax 3", 3,(4,),"softmax")
model = DLModel()
model.add(softmax_layer)
model.compile("categorical_cross_entropy")
X = np.random.randn(4, 50000)*10
Y = np.zeros((3, 50000))
sumX = np.sum(X,axis=0)
for i in range (len(Y[0])):
    if sumX[i] > 5:
        Y[0][i] = 1
    elif sumX[i] < -5:
        Y[2][i] = 1
    else:
        Y[1][i] = 1
costs = model.train(X,Y,1000)
plt.plot(costs)
plt.show()
predictions = model.predict(X)
print("right",np.sum(Y.argmax(axis=0) == predictions.argmax(axis=0)))
print("wrong",np.sum(Y.argmax(axis=0) != predictions.argmax(axis=0)))


print (" --- Stage 2 ---")
print (" ---------------")


print (" --- Targil 140.2.1 ---")


mnist = fetch_openml('mnist_784')
X, Y = mnist["data"], mnist["target"]
X = X / 255.0 -0.5
i = 12

img = X[i:i+1].to_numpy().reshape(28,28)
plt.imshow(img, cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print("Label is: '"+Y[i]+"'")


print (" --- Targil 140.2.2 ---")
digits = 10
examples = Y.shape[0]

Y = Y.to_numpy().reshape(1, examples)

Y_new = np.eye(digits)[Y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

print(Y_new[:,12])

print (" --- Targil 140.2.3 ---")
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

np.random.seed(111)
shuffle_index = np.random.permutation(m)

X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
i = 12
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(Y_train[:,i])

print (" --- Targil 140.2.4 ---")
np.random.seed(1)
hidden_layer = DLLayer ("Softmax 1", 64,(784,),"sigmoid","He", 1)
softmax_layer = DLLayer ("Softmax 1", 10,(64,),"softmax","He", 1)
model = DLModel()
model.add(hidden_layer )
model.add(softmax_layer)
model.compile("categorical_cross_entropy")
costs = model.train(X_train,Y_train,2000)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(1))
plt.show()

print (" --- Targil 140.2.5 ---")

print('Deep train accuracy')
model.confusion_matrix(X_train, Y_train)
print('Deep test accuracy')
model.confusion_matrix(X_test, Y_test)


print (" --- Targil 140.2.6 ---")

from PIL import Image, ImageOps
#Test your image
num_px = 28
img_path = r'data\imag\three.jpg' # full path of the rgb image
my_label_y = [0,0,0,1,0,0,0,0,0,0] # change the 1’s position to fit image
image = Image.open(img_path)
image28 = image.resize((num_px, num_px), Image.LANCZOS) # resize to 28X28 
plt.imshow(image)	   # Before scale 
plt.show();
plt.imshow(image28)  # After scale
plt.show();
gray_image = ImageOps.grayscale(image28)# grayscale – to fit to training data   
my_image = np.reshape(gray_image,(num_px*num_px,1))
my_label_y = np.reshape(my_label_y,(10,1))
my_image = my_image / 255.0 -0.5  # normelize
p = model.predict(my_image)
print (p)
