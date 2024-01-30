from StudantDL import *

# Dan Kazaz
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import unit10.c1w2_utils as u10
from DL1 import *

'''
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()


index = 25 # change index to get a different picture
plt.imshow(train_set_x_orig[index])
plt.show()
train_set_y=train_set_y.T
print ("y = " + str(train_set_y[index]) + ", it's a '" +
classes[np.squeeze(train_set_y[index])].decode("utf-8") + "' picture.")


m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]
num_P=num_px*num_px*3

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("Number of parameters to tune: num_P = " + str(num_P+1))
print()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print()

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def dSigmoid_dZ(Z): 
    A = sigmoid(Z)
    dZ = A*(1-A)
    return dZ

Z = np.array([[1,2],[0,100],[-2,-100]])
print("Z = " + str(Z))
print ("sigmoid(Z) = " + str(sigmoid(Z)))
print ("sigmoid'(Z) = " + str(dSigmoid_dZ(Z)))
print()

def initialize_with_zeros(dim):
    W = np.zeros((dim, 1))
    b = 0

    return W, b

W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print("b = " + str(b))
print()

def forward_propagation(X, Y, W, b):
    A = sigmoid(np.dot(W.T,X)+b)
    m = X.shape[1]
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return A, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]),np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
print ("cost = " + str(cost))
print()

def backward_propagation(X, Y, A):
    m = X.shape[1]
    dZ = A-Y
    dw = (1/m)*np.dot(X,dZ.T)
    db = (1/m)*np.sum(dZ)
    return dw, db

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]),np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
dw, db = backward_propagation(X, Y, A)
print ("dW = " + str(dw))
print ("db = " + str(db))
print()

def train(X, Y, num_iterations, learning_rate):
    W, b = initialize_with_zeros(X.shape[0])
    costs = []
    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)
        W = W-learning_rate*dw
        b = b-learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    return W, b, costs

X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1])
W, b, costs = train(X, Y, num_iterations= 100, learning_rate = 0.009)
print ("W = " + str(W))
print ("b = " + str(b))
    
def predict(X, W, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    W = W.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(W.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction

W = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(X, W, b)))

def train(X, Y, epochs, alpha):
    W, b = initialize_with_zeros(X.shape[0])
    costs = []
    for i in range(epochs):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)
        W = W-alpha*dw
        b = b-alpha*db
        if i % 100 == 0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    return W, b, costs

W, b, costs = train(train_set_x, train_set_y, epochs=4000, alpha=0.005)
Y_prediction_train = predict(train_set_x, W, b)
Y_prediction_test = predict(test_set_x, W, b)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train -
train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test -
test_set_y)) * 100))
'''

#DL1.py code:
np.random.seed(1)
l = [None]
l.append(DLLayer("Hidden 1", 6, (4000,)))
print(l[1])
l.append(DLLayer("Hidden 2", 12,
(6,),"leaky_relu", "random", 0.5,"adaptive"))
l[2].adaptive_cont = 1.2
print(l[2])
l.append(DLLayer("Neurons 3",16, (12,),"tanh"))
print(l[3])
l.append(DLLayer("Neurons 4",3, (16,),"sigmoid",
"random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
print(l[4])

Z = np.array([[1,-2,3,-4],
              [-10,20,30,-40]])
l[2].leaky_relu_d = 0.1
for i in range(1, len(l)):
    print(l[i].activation_forward(Z))
    
np.random.seed(2)
m = 3
X = np.random.randn(4000,m)
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    print('layer',i," A", str(Al.shape), ":\n", Al)
    
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    dZ = l[i].activation_backward(Al)
    print('layer',i," dZ", str(dZ.shape), ":\n", dZ)
    
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)
    
np.random.seed(3)
fig, axes = plt.subplots(1, 4, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
dAl = np.random.randn(Al.shape[0],m) * np.random.random_integers(-100, 100, Al.shape)
for i in reversed(range(1,len(l))):
    axes[i-1].hist(dAl.reshape(-1), align='left')
    axes[i-1].set_title('dAl['+str(i)+']')
    dAl = l[i].backward_propagation(dAl)
plt.show()

'''
np.random.seed(4)
l1 = DLLayer("Hidden1", 3, (4,),"trim_sigmoid", "zeros", 0.2, "adaptive")
l2 = DLLayer("Hidden2", 2, (3,),"relu", "random", 1.5)
print("before update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))
l1.dW = np.random.randn(3,4) * np.random.randint(-100,100)
l1.db = np.random.randn(3,1) * np.random.randint(-100,100)
l2.dW = np.random.randn(2,3) * np.random.randint(-100,100)
l2.db = np.random.randn(2,1) * np.random.randint(-100,100)
l1.update_parameters()
l2.update_parameters()
print("after update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))
'''

