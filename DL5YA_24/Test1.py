import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sys
sys.path.append('C:/Users/User/Desktop/')
from unit10 import c1w3_utils as u10
from SDL1 import *


np.random.seed(1)
X, Y = u10.load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))

input_shape = (10,)
hidden_layer_shape = (5,)
output_shape = (1,)
hidden_layer = DLLayer(input_shape, 5, activation="tanh")
output_layer = DLLayer(hidden_layer.output_shape, 1, activation="sigmoid")

hidden_layer.learning_rate = 0.1
output_layer.learning_rate = 0.1
hidden_layer.random_initialize_weights(0.01)
output_layer.random_initialize_weights(0.01)

output_layer.W = hidden_layer.output_weights()
output_layer.b = hidden_layer.output_bias()

model = DLModel(hidden_layer, output_layer)

model.compile(loss='cross_entropy')

model.decision_boundary = 0.5


def predict(x):
    y_pred = model.predict(x)
    y_pred = np.where(y_pred > model.decision_boundary, 1, 0)
    return y_pred

np.random.seed(1)
print(model)

model.train(X, Y, epochs=1500, print_ind=10)


costs = model.train(X,Y,10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) +
np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

print (" ==============================================================================================")
print (" ==============================================================================================")

#EX 3.3

import sys


np.random.seed(1)
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = u10.load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,"noisy_moons": noisy_moons,"blobs": blobs,"gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y%2
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))

input_shape = (10,)
hidden_layer_shape = (10,) # Changed the number of neurons in the hidden layer
output_shape = (1,)
hidden_layer = DLLayer(input_shape, 10, activation="relu", optimization="adaptive") # Changed the activation function of the hidden layer
output_layer = DLLayer(hidden_layer.output_shape, 1, activation="sigmoid")

hidden_layer.learning_rate = 0.01 # Changed the learning rate of the hidden layer
output_layer.learning_rate = 0.01 # Changed the learning rate of the output layer
hidden_layer.random_initialize_weights(0.01)
output_layer.random_initialize_weights(0.01)

output_layer.W = hidden_layer.output_weights()
output_layer.b = hidden_layer.output_bias()

model = DLModel(hidden_layer, output_layer)

model.compile(loss='cross_entropy')

model.decision_boundary = 0.5


def predict(x):
    y_pred = model.predict(x)
    y_pred = np.where(y_pred > model.decision_boundary, 1, 0)
    return y_pred

np.random.seed(1)
print(model)

model.train(X, Y, epochs=1500, print_ind=10)

hidden_layer2_shape = (5,)
hidden_layer2 = DLLayer(hidden_layer.output_shape, 5, activation="relu", optimization="adaptive")
output_layer2 = DLLayer(hidden_layer2.output_shape, 1, activation="sigmoid")

hidden_layer2.learning_rate = 0.01
output_layer2.learning_rate = 0.01
hidden_layer2.random_initialize_weights(0.01)
output_layer2.random_initialize_weights(0.01)

output_layer2.W = hidden_layer2.output_weights()
output_layer2.b = hidden_layer2.output_bias()

model = DLModel(hidden_layer2, output_layer2)
model.compile(loss='cross_entropy')
model.decision_boundary = 0.5

costs = model.train(X, Y, 10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(10))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %f' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
