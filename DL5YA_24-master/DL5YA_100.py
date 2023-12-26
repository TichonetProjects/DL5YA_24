# 100
# Gad Lidror
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import unit10.c1w2_utils as u10

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

# Example of a picture
index = 25 # change index to get a different picture
plt.imshow(train_set_x_orig[index])
#plt.show()
print ("y = " + str(train_set_y[0][index]) + ", it's a '" + 
       classes[np.squeeze(train_set_y[0][index])].decode("utf-8") +  "' picture.")

print ("--- Ex 100.1 Finding basic information about the database ---")

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
num_P = num_px*num_px*3+1       # We know each picture is in the shape (num_px, num_px, 3)

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("Number of parameters to tune: num_P = " + str(num_P))


print ("--- Ex 100.2 Flatten arrays ---")

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten/255.0 - 0.5
test_set_x = test_set_x_flatten/255.0 - 0.5

print ("--- Ex 100.3 Sigmoid ---")

# Sigmoid on a numpy matrix. 
def sigmoid(Z):
     s = 1/(1+np.exp(-Z))
     return s

# Derivative of sigmoid on a numpy matrix. 
def dSigmoid_dZ(Z):
    s = sigmoid(Z)
    return s*(1-s)

Z = np.array([[1,2],[0,100],[-2,-100]])
print("Z = " + str(Z))
print ("sigmoid(Z) = " + str(sigmoid(Z)))
print ("sigmoid'(Z) = " + str(dSigmoid_dZ(Z)))

print ("--- Ex 100.4 Initialize parameters to zero ---")

def initialize_with_zeros(dim):
    W = np.zeros((dim,1), dtype = float)
    b = 0.0
    return W, b

W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print ("b = " + str(b))

print ("--- Ex 100.5 Forward propegation ---")

# This function calculates the linear portion of the perspetron (Z)
#     for all of the samples in paralel (using numpy)
# then, it will do the non-linear part of the perspetron (Sigmoid) that result in A
# then it will calculate the cost (using cross antropy)
def forward_propagation(X, Y, W, b):
    m = X.shape[1]
    Z = np.dot(W.T,X) + b 
    A = sigmoid(Z)
    
    # compute the cost
    J = np.sum(cross_entropy(A,Y))/m  
    J = np.squeeze(J)
    return A, J

# Caculate cross entropy cost function. 
# input paramters:
# Y_hat - the activation result (A)
# Y - the real value of the samples
# return the error (cross entropy) of each sample
def cross_entropy(Y_hat, Y):
    L = -((1 - Y)*np.log(1 - Y_hat)+Y*np.log(Y_hat)) # the error for each sample
    return L 

# Calculate the derivative of the cross entropy function
# return the derivative for each A (activation) value of sample that was part of the cross entropy
def dCross_entropy(Y_hat, Y):
    m = Y_hat.shape[1]
    dY_hat =(1-Y)/(1-Y_hat)-Y/Y_hat
    return dY_hat/m

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1]) 
A, cost = forward_propagation(X, Y, w, b)
print ("cost = " + str(cost))


print ("--- Ex 100.6 Backword propegation ---")

# Backword propegation function. Will use the derivatives calculations
def backward_propagation(X, Y, A):
    m = X.shape[1]
    dZ = (A-Y)/m
    dW = np.dot(X, dZ.T)
    db = np.sum(dZ)
    return dW, db

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1]) 
A, cost = forward_propagation(X, Y, w, b)
dw, db = backward_propagation(X, Y, A)
print ("dW = " + str(dw))
print ("db = " + str(db))

print ("--- Ex 100.7 train ---")

# This function do the full training session. 
# It initialize the parameters 
# It do forward, backword and parameters setting , epoch times.
# return the trained W and b
def train(X, Y, epochs, alpha):
    n = X.shape[0]
    delta = max(int(epochs / 100),1)
    costs = []
    W,b = initialize_with_zeros(n)
    for i in range (epochs):
        A, cost = forward_propagation(X, Y, W, b)   
        dW, db = backward_propagation(X, Y, A)
        W -= alpha * dW
        b -= alpha * db
        if (i !=0 and i % delta == 0):
            print ("Iteration " , i , "  cost " , cost)
            costs.append(cost)    
    return W, b, costs


X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1]) 
W, b, costs = train(X, Y, epochs= 100, alpha = 0.009)
print ("W = " + str(W))
print ("b = " + str(b))

print ("--- Ex 100.8 predict ---")

# This function get data (X) and for each, try to predict 
# if it is a cat or not, using a threshold to decide it True or False
def predict(X, W, b):
    threshold = 0.5

    Z = np.dot(W.T,X) + b 
    A = sigmoid(Z)
    Predictions = np.where(A>threshold, 1, 0)
    return Predictions
    
W = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(X, W, b)))


print ("--- Ex 100.9 Cat or not-cat ---")
# Train the perseptron, the use the predict function
# to see how many of the training set was correctly identified
# and how many from the test (accuracy)
W, b, costs = train(train_set_x, train_set_y, epochs=4000, alpha=0.005)
Y_prediction_train = predict(train_set_x, W, b)
Y_prediction_test = predict(test_set_x, W, b)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))


"""
fname = r'black_cat.jpg'  # <=== change image full path
img = Image.open(fname)
# Set the image to be with the same parameters as the training data
img = img.resize((num_px, num_px), Image.ANTIALIAS)
plt.imshow(img)
plt.show()

my_image_flatten = np.array(img).reshape(1, -1).T	 # flatten the image
my_image = my_image_flatten/255.0		 # normalize the image

my_predicted_image = predict(my_image, W, b)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""