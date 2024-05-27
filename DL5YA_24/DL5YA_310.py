import numpy as np
import matplotlib.pyplot as plt
from unit10 import c2w1_utils as u10
import sklearn
import sklearn.datasets
import scipy.io
from DL4_Comp import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


print ("\n --- Targil 310.1 ---\n")

l1 = DLLayer("Hidden",6 , (5,) ,"relu", learning_rate = 0.1)
l2 = DLLayer("Output",1 , (6,) ,"sigmoid",learning_rate = 0.1)
print("Default:")
print(l1.is_train)  #_is_compiled
print(l2.is_train)
n = DLModel("Example")
n.add(l1)
n.add(l2)
n.set_train(True)
print("After set to True:")
print(l1.is_train)
print(l2.is_train)


print ("\n --- Targil 310.2 ---\n")
np.random.seed(1)
ll1 = DLLayer ("no regularization",8,(7,),"tanh",learning_rate = 0.1)
ll2 = DLLayer("L2 regularization", 18, (7,) , "sigmoid", learning_rate = 0.1, regularization="L2")
ll2.L2_lambda = 0.5
ll3 = DLLayer("dropout", 8, (17,), "leaky_relu", learning_rate = 0.1, regularization="dropout")
print(f"{ll1}\n{ll2}\n{ll3}")

print ("\n --- Targil 310.3 ---\n")
np.random.seed(2)
l4 = DLLayer("forward dropout", 2, (4,), activation = "NoActivation",  learning_rate = 0.1, regularization = "dropout")
prev_A = np.random.randn(4,5) * 10
A_no_dropout = l4.forward_propagation(prev_A)
print("Input with no dropout:")
print(l4._A_prev)
print("Output with no dropout:")
print(A_no_dropout)
np.random.seed(2)
l4.set_train(True)
A_with_dropout = l4.forward_propagation(prev_A)
print("Input with dropout: (same input as without dropout).")
print(l4._A_prev)
print("Output with dropout:")
print(A_with_dropout)


print ("\n --- Targil 310.4 ---\n")
np.random.seed(2)
A4 = np.random.randn(15,4) * 5
l5 = DLLayer("backward dropout", 3, (15,), activation = "NoActivation", W_initialization="Xaviar", learning_rate = 0.1, regularization = "dropout")
l5.set_train(True)
A5_with_dropout = l5.forward_propagation(A4)
dA5 = np.random.randn(3,4) * 7
dA4 = l5.backward_propagation(dA5)
print ("A4 with dropout:")
print (l5._A_prev)
print ("dA4:")
print (dA4)



print ("\n --- Targil 310.5 ---\n")
np.random.seed(2)
l1 = DLLayer("Hidden1", 6, (5,), activation = "NoActivation", learning_rate = 0.1, regularization = "L2", W_initialization="random")
l2 = DLLayer("Hidden2", 12, (6,), activation="tanh", learning_rate = 0.1, regularization = "L2", W_initialization="random")
l3 = DLLayer("Output", 1, (12,), activation="sigmoid", learning_rate = 0.1, W_initialization="random")
n = DLModel("L2 model")
n.add(l1)
n.add(l2)
n.add(l3)
n.compile( "cross_entropy", threshold = 0.6)
Y_hat = np.random.rand(1,17)
Y = np.random.rand(1,17)
Y = np.where(Y>0.4,1,0)
print(f"Cost with L2 regularization: {n.compute_cost(Y_hat, Y)}")
l1.L2_lambda = 0
l2.L2_lambda = 0
print(f"Cost without L2 regularization: {n.compute_cost(Y_hat, Y)}")



print ("\n --- Targil 310.6 ---\n")
np.random.seed(2)
l = DLLayer("backward L2", 7, (4,), learning_rate = 0.1, activation = "NoActivation", regularization = "L2", W_initialization="random")
l.W *= 100
prev_A = np.random.randn(4,11) * 5
Z = l.forward_propagation(prev_A)
dZ = np.random.randn(*Z.shape)
dA_prev = l.backward_propagation(dZ)
print(f"dW with regularization:\n{l.dW}")
l.L2_lambda = 0
dA_prev = l.backward_propagation(dZ)
print(f"dW with no regularization:\n{l.dW}")


print ("\n --- Targil 310.7 No regularization---\n")
np.random.seed(2)
train_X, train_Y, test_X, test_Y = u10.load_2D_dataset()

l1 = DLLayer("Layer1", 64, (2,), learning_rate = 0.05, activation = "relu", W_initialization="He")
l2 = DLLayer("Layer2", 32, (64,), learning_rate = 0.05, activation = "relu", W_initialization="He")
l3 = DLLayer("Layer3", 5, (32,), learning_rate = 0.05, activation = "relu", W_initialization="He")
l4 = DLLayer("Layer4", 1, (5,), learning_rate = 0.05, activation = "sigmoid", W_initialization="He")

model = DLModel("Example")
model.add(l1)
model.add(l2)
model.add(l3)
model.add(l4)
model.set_train(True)
model.compile("categorical_cross_entropy")

costs = model.train(train_X, train_Y, 70000)
print("train accuracy:", np.mean((model.predict(train_X) > 0.7) == train_Y))
print("test accuracy:", np.mean((model.predict(test_X) > 0.7) == test_Y))
plt.title(f"Model no regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
u10.plot_decision_boundary(lambda x: model.predict(x.T), train_X, train_Y)

plt.plot(costs)
plt.show()



print ("\n --- Targil 310.7 dropout ---\n")
np.random.seed(2)
train_X, train_Y, test_X, test_Y = u10.load_2D_dataset()

l1 = DLLayer("Layer1", 64, (2,), learning_rate = 0.05, activation = "relu", W_initialization="He")
l2 = DLLayer("Layer2", 32, (64,), learning_rate = 0.05, activation = "relu", W_initialization="He", regularization="dropout")
l3 = DLLayer("Layer3", 5, (32,), learning_rate = 0.05, activation = "relu", W_initialization="He", regularization="dropout")
l4 = DLLayer("Layer4", 1, (5,), learning_rate = 0.05, activation = "sigmoid", W_initialization="He")

model = DLModel("Example")
model.add(l1)
model.add(l2)
model.add(l3)
model.add(l4)
model.set_train(True)
model.compile("categorical_cross_entropy")


costs = model.train(train_X, train_Y, 20000)
print("train accuracy:", np.mean((model.predict(train_X) > 0.7) == train_Y))
print("test accuracy:", np.mean((model.predict(test_X) > 0.7) == test_Y))
plt.title(f"Model deopout regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
u10.plot_decision_boundary(lambda x: model.predict(x.T), train_X, train_Y)

plt.plot(costs)
plt.show()



print ("\n --- Targil 310.7 L2 ---\n")
np.random.seed(2)
train_X, train_Y, test_X, test_Y = u10.load_2D_dataset()

l1 = DLLayer("Layer1", 64, (2,), learning_rate = 0.05, activation = "relu", W_initialization="He", regularization="L2")
l2 = DLLayer("Layer2", 32, (64,), learning_rate = 0.05, activation = "relu", W_initialization="He", regularization="L2")
l3 = DLLayer("Layer3", 5, (32,), learning_rate = 0.05, activation = "relu", W_initialization="He", regularization="L2")
l4 = DLLayer("Layer4", 1, (5,), learning_rate = 0.05, activation = "sigmoid", W_initialization="He")

model = DLModel("Example")
model.add(l1)
model.add(l2)
model.add(l3)
model.add(l4)
model.set_train(True)
model.compile("categorical_cross_entropy")


costs = model.train(train_X, train_Y, 20000)
print("train accuracy:", np.mean((model.predict(train_X) > 0.7) == train_Y))
print("test accuracy:", np.mean((model.predict(test_X) > 0.7) == test_Y))
plt.title(f"Model L2 regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
u10.plot_decision_boundary(lambda x: model.predict(x.T), train_X, train_Y)

plt.plot(costs)
plt.show()