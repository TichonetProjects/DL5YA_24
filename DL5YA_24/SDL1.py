# Roy Cohen - DLLayer class - from 1.1 to 1.3(including 1.4 working as well with 1.5-1.6 partly working (there are some bugs that I wasn't able to fix yet))
from tkinter import SEL
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import h5py
class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate = 0.01, optimization = "None", random_scale = 0.01):
        # initialize the weights and bias
        self._name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._learning_rate = learning_rate
        self._optimization = optimization
        self.random_scale = random_scale

        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self._learning_rate)
            self._adaptive_alpha_W = np.full((self._num_units, *(self._input_shape)), self._learning_rate)

        self.adaptive_cont = 1.1
        self.adaptive_switch = 0.5
        
        self.activation_trim = 1e-10

        # the original is relu
        self.activation_forward = self.__relu
        self.activation_backward = self._relu_backward
        
        if (activation == "sigmoid"):
            self.activation_forward = self.__sigmoid
            self.activation_backward = self._sigmoid_backward
        if (activation == "leaky_relu"):
            self.leaky_relu_d = 0.01
            self.activation_forward = self.__leaky_relu
            self.activation_backward = self._leaky_relu_backward
        if (activation == "tanh"):
            self.activation_forward = self.__tanh
            self.activation_backward = self._tanh_backward
        if (activation == "trim_sigmoid"):
            self.activation_trim = 1e-10
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward
        if (activation == "trim_tanh"):
            self.activation_trim = 1e-10
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward
            
        
        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float)
        if W_initialization == "zeros":
            self.W = np.zeros((self._num_units, *(self._input_shape)), dtype=float)
        elif W_initialization == "He":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(1/self._input_shape[0]) * self.random_scale
        elif W_initialization == "Xaviar":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(2/self._input_shape[0]) * self.random_scale
        elif W_initialization == "random":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        else:  
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)
        if (0 in self.W):
            print("true")
   

    # forwards: 
    def __sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    def __leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_d)    
    def __relu(self, Z):
        return np.maximum(0, Z)
    def __tanh(self, Z):
        return np.tanh(Z)
    
    def _trim_sigmoid(self,Z):

        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100,Z)
                A = A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A
    
    def forward_propagation(self, A_prev, is_predict):
        # copy the previous layer's results
        self._A_prev = np.array(A_prev, copy=True)
        # calculate the linear function
        self._Z = np.dot(self.W, self._A_prev) + self.b
        # activate that and return it 
        return self.activation_forward(self._Z)

    # backwords:
    def _sigmoid_backward(self, dA):
        A = self.__sigmoid(self._Z)
        return dA * A * (1-A)

    def _leaky_relu_backward(self, dA):
        return np.where(self._Z <= 0, dA * self.leaky_relu_d, dA)    

    def _relu_backward(self,dA):
        return np.where(self._Z <= 0, 0, dA)
    
    # I do not understand why the function that I created didn't work but I found this on the Internet and it works and it does the same thing as the function that I created
    def _tanh_backward(self, dA):
        return dA * (1 - np.power(self.__tanh(self._Z), 2))
    
    def _trim_sigmoid_backward(self, dA):
        return self._sigmoid_backward(dA)
    
    def _trim_tanh_backward(self, dA):
        return self._tanh_backward(dA)
 
    def backward_propagation(self, dA):
        # calculate the derivative of the activation function
        dZ = self.activation_backward(dA)
        # calculate the derivative of the weights
        self.dW = np.dot(dZ, self._A_prev.T) * (1/self._A_prev.shape[1])
        # calculate the derivative of the bias
        self.db = np.sum(dZ, axis=1, keepdims=True) * (1/self._A_prev.shape[1])
        # calculate the derivative of the previous layer
        dA_prev = np.dot(self.W.T, dZ)
        # return the derivative of the previous layer
        return dA_prev

        # create the function "update_parameters" that updates the weights and bias
    def update_parameters(self):
        # update the weights
        if self._optimization == "adaptive":
            self._adaptive_alpha_W = np.where(self.dW * self._adaptive_alpha_W > 0, self._adaptive_alpha_W * self.adaptive_cont, self._adaptive_alpha_W * self.adaptive_switch)
            self.W = self.W - self.dW * self._adaptive_alpha_W
        else:
            self.W = self.W - self.dW * self._learning_rate
        # update the bias
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.where(self.db * self._adaptive_alpha_b > 0, self._adaptive_alpha_b * self.adaptive_cont, self._adaptive_alpha_b * self.adaptive_switch)
            self.b = self.b - self.db * self._adaptive_alpha_b
        else:
            self.b = self.b - self.db * self._learning_rate
    
    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W",  data=self.W)
            hf.create_dataset("b",  data=self.b)


    
            
    
    def __str__(self):

        s = self._name + " Layer:\n"

        s += "\tnum_units: " + str(self._num_units) + "\n"

        s += "\tactivation: " + self._activation + "\n"

        if self._activation == "leaky_relu":

            s += "\t\tleaky relu parameters:\n"

            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"

        s += "\tinput_shape: " + str(self._input_shape) + "\n"

        s += "\tlearning_rate (alpha): " + str(self._learning_rate) + "\n"

        #optimization

        if self._optimization == "adaptive":

            s += "\t\tadaptive parameters:\n"

            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"

            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"

        # parameters

        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"

        s += "\t\tshape weights: " + str(self.W.shape)+"\n"

        plt.hist(self.W.reshape(-1))

        plt.title("W histogram")
        
        plt.show()

        return s;

    


class DLModel:
    def __init__(self, name = "Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
    def add(self,layer):
        self.layers.append(layer)
    

    def squared_means(self, AL,Y):
        return np.square(Y - AL)

    def squared_means_backward(self, AL,Y):
         return -2 * (Y - AL)

    def cross_entropy(self, AL,Y):
        #AL = np.where(AL==0, 0.00000000000000001, AL)
        return np.where(Y==0,-np.log(1-AL),-np.log(AL))
            
    def cross_entropy_backward(self, AL,Y):
        #AL = np.where(AL==0, 0.00000000000000001, AL)
        return np.where(Y==0,1/(1-AL), -1/AL)
        
    
    def compile(self, loss, threshold=0.5):
        self._is_compiled = True
        self.loss = loss
        self.threshold = threshold
        if loss == "squared_means":
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        if loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward
    
    def compute_cost(self,AL,Y):
        m = AL.shape[1]
        return np.sum(self.loss_forward(AL,Y))/m
    
    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
        # forward propagation
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,False)
            #backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1,L)):
                dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                self.layers[l].update_parameters()
            #record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ",str(i//print_ind),"%:",str(J))
        return costs

    def predict(self, X):
            L = len(self.layers)
            Al = X
            for l in range(1,L):
                Al = self.layers[l].forward_propagation(Al,True)
            return Al > self.threshold
    
    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        for i in range(1,len(self.layers)):
            self.layers[i].save_weights(path, f"Layer{i}")
    
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s
