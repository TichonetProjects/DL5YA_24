from tkinter import W
import numpy
import random
import matplotlib.pyplot as plt
import math
from unit10 import c2w1_init_utils as u10
from SDL1 import *

'''
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()

numpy.random.seed(1)
hidden1 = DLLayer("Perseptrons 1",
30,(12288,),"relu",W_initialization = "Xaviar",learning_rate
= 0.0075, optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2",
15,(30,),"trim_sigmoid",W_initialization = "He",learning_rate
= 0.1)
print(hidden1)
print(hidden2)
hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075)
hidden1.b = numpy.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir","Hidden1")
hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization = "SaveDir/Hidden1.h5",learning_rate = 0.1)
print(hidden1)
print(hidden2)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
dir = "model"
model.save_weights(dir)
print(os.listdir(dir))
'''
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()


model = DLModel("Model", True)
model._add(DLLayer("Perspetron 1", 10, (2,), "relu", W_initialization="random", learning_rate=0.01, random_scale=1))
model._add(DLLayer("Perspetron 2", 5, (10,), "relu", W_initialization="random", learning_rate=0.01, random_scale=0.01))
model._add(DLLayer("Output", 1, (5,), "trim_sigmoid", W_initialization="random", learning_rate=0.1, random_scale=10))

model._compile("cross_entropy")


# costs = model.train(train_X, train_Y, 15000)
'''
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
axes = plt.gca()
axes.set_ylim([0.65,0.75])
plt.title("Model with -zeros- initialization")
plt.show()
'''
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.show()
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)
predictions = model.predict(train_X)
print ('Train accuracy: %d' % float((numpy.dot(train_Y,predictions.T) + numpy.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model.predict(test_X)
print ('Test accuracy: %d' % float((numpy.dot(test_Y,predictions.T) + numpy.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')



"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c2w1_init_utils as u10
from SDL1 import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# load image dataset: blue/red dots in circles

train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()


model13 = DLModel()
layer1 = DLLayer("Perseptrons 1", 10,(2,),"relu",W_initialization = "random",learning_rate = 0.01, optimization = None, random_scale = 1)
layer2 = DLLayer("Perseptrons 2", 5,(10,),"relu",W_initialization = "random",learning_rate = 0.01, optimization = None, random_scale = 0.01)
layer3 = DLLayer("Output layer", 1,(5,),"trim_sigmoid",W_initialization = "zeros",learning_rate = 0.1, optimization = None, random_scale = 10)

model13._add(layer1)
model13._add(layer2)
model13._add(layer3)

model13._compile("cross_entropy", 0.5)


costs = model13.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(" random initialization")
plt.show()
plt.title("Model with random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model13.predict(x.T), test_X, test_Y)
predictions = model13.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model13.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')

#1.5
#Same code but He
model13 = DLModel()
layer1 = DLLayer("Perseptrons 1", 10,(2,),"relu",W_initialization = "He",learning_rate = 0.01, optimization = None, random_scale = 1)
layer2 = DLLayer("Perseptrons 2", 5,(10,),"relu",W_initialization = "He",learning_rate = 0.01, optimization = None, random_scale = 0.01)
layer3 = DLLayer("Output layer", 1,(5,),"trim_sigmoid",W_initialization = "He",learning_rate = 0.1, optimization = None, random_scale = 10)

model13._add(layer1)
model13._add(layer2)
model13._add(layer3)

model13._compile("cross_entropy", 0.5)


costs = model13.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(" He initialization")
plt.show()
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model13.predict(x.T), test_X, test_Y)
predictions = model13.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model13.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')

"""