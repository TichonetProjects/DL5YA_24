import numpy as np
import random 
# import unit10.b_utils as u10
import matplotlib.pyplot as plot
import unit10.c1w2_utils as u10


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()
'''
index = 25 # change index to get a different picture
plot.imshow(train_set_x_orig[index])
plot.show()
print ("y = " + str(train_set_y[index]) + ", it's a '" + classes[np.squeeze(train_set_y[index])].decode("utf-8") + "' picture.")
'''
# תרגיל 1
m_train = len(train_set_y[0])
m_test = len(test_set_y[0])
num_px = len(train_set_x_orig[0])
num_P = num_px**2*3 + 1

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("Number of parameters to tune: num_P = " + str(num_P))

# תרגיל 2 

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

train_set_x_flatten = train_set_x_flatten.T
test_set_x_flatten = test_set_x_flatten.T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0


# תרגיל 3

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid_dZ(x):
    return sigmoid(x) * (1-sigmoid(x))


Z = np.array([[1,2],[0,100],[-2,-100]])
print("Z = " + str(Z))
print ("sigmoid(Z) = " + str(sigmoid(Z)))
print ("sigmoid'(Z) = " + str(dSigmoid_dZ(Z)))

# תרגיל 4 

def initialize_with_zeros(dim):
    W = np.zeros((dim, 1))
    W = np.array(W)
    b = 0
    return W, b

W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print ("b = " + str(b))


# תרגיל 5

def forward_propagation(X, Y, W, b):
    predict = np.dot(W.T, X)
    A = sigmoid(predict + b)
    m = X.shape[1]
    J = np.sum(-(Y*np.log(A)+(1-Y)*np.log(1-A))/m)
    return A, J

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
print("Activation shape = " + str(A.shape))
print ("cost = " + str(cost))

def backward_propagation(X, Y, A):
    dz = (A-Y)/X.shape[0]
    dw = np.dot(X, dz.T)
    db = np.sum(dz)
    return dw, db

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
dw, db = backward_propagation(X, Y, A)
print ("dW = " + str(dw))
print ("db = " + str(db))
# print("dZ = " + str(dz))

# תרגיל 7

def train(X, Y, epoches, alpha):
    W, b = initialize_with_zeros(X.shape[0])
    activation_new, J_compare = forward_propagation(X,Y,W,b)
    alpha_W, alpha_b = backward_propagation(X, Y, activation_new)
    for i in range(epoches):
        for i in range(len(W)):
            A, J = forward_propagation(X, Y, W, b)
            dW, db = backward_propagation(X, Y, A)
        
        alpha_W = np.where(dW * alpha_W > 0, alpha_W * 1.1, alpha_W * -0.5)
        alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5

        W -= alpha_W*dW
        b -= alpha_b*db

    
    return W, b, J

X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1])
W, b, costs = train(X, Y,100,0.009)
print ("W = " + str(W))
print ("b = " + str(b))



