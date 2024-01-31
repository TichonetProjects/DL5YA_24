
import numpy as np
import matplotlib.pyplot as plt
import random
import math
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def leaky_relu(x):
    return np.maximum(0.1*x,x)
def tanh(x):
    return np.tanh(x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
def cross_entropy_loss(Y, A):
    return -np.sum(Y*np.log(A))/Y.shape[1]
def dSigmoid_dZ(Z):
    return sigmoid(Z)*(1-sigmoid(Z))
def dRelu_dZ(Z):
    return np.where(Z>0,1,0)
def dLeaky_relu_dZ(Z):
    return np.where(Z>0,1,0.1)
def dTanh_dZ(Z):
    return 1-np.tanh(Z)**2
def dSoftmax_dZ(Z):
    return Z*(1-Z)
def dCross_entropy_loss_dZ(Y, A):
    return A-Y

class DLLayer:
    def __init__(self, name, num_units, input_shape: tuple, activation="relu", W_initialization ="random", Learning_Rate=0.001, optimization = "none"):
        def relu_backward(self, dA):
            dZ = np.where(self._Z <= 0, 0, dA)
            return dZ
        def leaky_relu_backward(self, dA):
            dZ = np.where(self._Z <= 0, self.leaky_relu_d*dA, dA)
            return dZ
        def tanh_backward(self, dA):
            dZ = dA*(1-np.tanh(self._Z)**2)
            return dZ
        def sigmoid_backward(self, dA):
            A = self._sigmoid(self._Z)
            dZ = dA * A * (1-A)
            return dZ
        def softmax_backward(self, dA):
            dZ = dA * self._A * (1-self._A)
            return dZ
        def backward_propagation(self, dA):
            if self.activation == "sigmoid":
                dZ = sigmoid_backward(self, dA)
            elif self.activation == "relu":
                dZ = relu_backward(self, dA)
            elif self.activation == "leaky_relu":
                dZ = leaky_relu_backward(self, dA)
            elif self.activation == "tanh":
                dZ = tanh_backward(self, dA)
            elif self.activation == "softmax":
                dZ = softmax_backward(self, dA)
            self.dW = np.dot(dZ, self._A.T)/self._A.shape[1]
            self.db = np.sum(dZ, axis=1, keepdims=True)/self._A.shape[1]
            dA_prev = np.dot(self.W.T, dZ)
            return dA_prev
        def update_parameters(self):
            if self.optimization == "none":
                self.W = self.W-self.Learning_Rate*self.dW
                self.b = self.b-self.Learning_Rate*self.db
            elif self.optimization == "adaptive":
                self.W = self.W-self.Learning_Rate*self.dW
                self.b = self.b-self.Learning_Rate*self.db
                self.Learning_Rate = self.Learning_Rate*1.2
            elif self.optimization == "momentum":
                self.W = self.W-self.Learning_Rate*self.dW
                self.b = self.b-self.Learning_Rate*self.db
                self.Learning_Rate = self.Learning_Rate*1.2
            elif self.optimization == "rmsprop":
                self.W = self.W-self.Learning_Rate*self.dW
                self.b = self.b-self.Learning_Rate*self.db
                self.Learning_Rate = self.Learning_Rate*1.2
            elif self.optimization == "adam":
                self.W = self.W-self.Learning_Rate*self.dW
                self.b = self.b-self.Learning_Rate*self.db
                self.Learning_Rate = self.Learning_Rate*1.2
        def init_weights(self, W_initialization):
            if W_initialization == "random":
                self.W = np.random.randn(self.num_units, self.input_shape[0])*self.random_scale
                self.b = np.zeros((self.num_units,1))
            elif W_initialization == "xavier":
                self.W = np.random.randn(self.num_units, self.input_shape[0])*np.sqrt(1/self.input_shape[0])
                self.b = np.zeros((self.num_units,1))
            elif W_initialization == "he":
                self.W = np.random.randn(self.num_units, self.input_shape[0])*np.sqrt(2/self.input_shape[0])
                self.b = np.zeros((self.num_units,1))
        self.name = name
        self.num_units = num_units
        self.input_shape = input_shape
        self.activation = activation
        self.W_initialization = W_initialization
        self.Learning_Rate = Learning_Rate
        self.optimization = optimization
        self.W = init_weights(self, self.W_initialization)
        self.b = np.zeros((self.num_units,1))
        self.dW = np.zeros((self.num_units, self.input_shape[0]))
        self.db = np.zeros((self.num_units,1))
        update_parameters(self)
        self.adaptive_cont = 1.2
        self.leaky_relu_d = 0.01
        self.adaptive_switch = 100
        self.random_scale = 1.0
        
    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self.num_units) + "\n"
        s += "\tactivation: " + self.activation + "\n"
        if self.activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self.input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.Learning_Rate) + "\n"
        if self.optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s;
