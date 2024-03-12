# DL1
# Gad Lidror
import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================
# =============================================================
#              DLModel
# =============================================================
# =============================================================
# This class implements a deep nuarl network (ANN).
# Input / Internal parameters:
# name - A string for the ANN (model)
# layers -  a list of layers that construct the ANN. Starting from the firs Hidden layer, upto the output layer
# is_compile - a boolean to indicate if setting of the ANN is completed
# is_train - a boolean value to indicate if we are using the object to train the internal parameters of the ANN
#      or we are using the ANN, that already have a trained set of parameters (e.g. W's and b's), to 
#      predict the value of a new sample (e.g. picture)
# Algorithm:
#    -The model is instantiated with 'name' and default values for the other internal parameters
#    -It must be activated by calling to 'compile' - setting the internal parameters of 'loss' function
#    -A sequence of calls to the function 'add' will add nuoron layers to the model. Calling to the add must be done in the right order of course
#    -Support activation of regularization mechanism in the different layers
#    * Train the model 
#    * Predict
#
# Predefined loss functions, already implemented, to choose from:
# - "squared_means"
# - "cross_entropy"
# - else - raise "Unimplemented loss function" exception
class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False

    # add a layer to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # loss functions
    # --------------    
    def squared_means(self, AL, Y):
        m = Y.shape[1]
        return np.power(AL-Y, 2) / m
    def squared_means_derivative(self, AL, Y):
        m = Y.shape[1]
        return 2*(AL-Y) / m
    def cross_entropy(self, AL, Y):
        return np.where(Y == 0, -np.log(1-AL), -np.log(AL))/Y.shape[1]
    def cross_entropy_derivative(self, AL, Y):
        return np.where(Y == 0, 1/(1-AL), -1/AL)/Y.shape[1]
    
    # compile the model. must be called prior to training
    def compile(self, loss, threshold = 0.5):
        self._is_compiled = True
        self.loss = loss
        self.threshold = threshold

        if (loss == "squared_means"):
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_derivative
        elif (loss == "cross_entropy"):
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_derivative
        else:
            print("*** Invalid loss function")
            raise NotImplementedError("Unimplemented loss function: " + loss)

    # compute the cost
    def compute_cost(self, AL, Y):
        return np.sum(self.loss_forward(AL, Y))
    
    # train the model
    # ---------------
    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)       # print progress every 1% of the iterations
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
    
    # predicts the value of a new sample
    def predict(self, X):
        L = len(self.layers)
        Al = X
        for l in range(1,L):
            Al = self.layers[l].forward_propagation(Al, True)
        return np.where(Al > self.threshold, True, False)
    
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"
        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

    
# =============================================================
# =============================================================
#              DLLayer
# =============================================================
# =============================================================
# This class implements a one layer of nuorons (Perceptrons).
# Input / Internal parameters:
# name - A string for the ANN (model)
# num_units - number of nuorons in the layer
# input_shape - number of inputs that get into the layer
# activation - name of the activation function (same for all the layer). implemented: 
#    - sigmoid
#    - trim_sigmoid
#    - tanh
#    - trim_tanh
#    - relu ( default )
#    - leaky_relu
#    - softmax
#    - trim_softmax
#    - NoActivation
# W_initialization - name of the initialization funciton (same for all the layer), implemented : zeros, random.
# learning_rate - sometimes called alpha.
# optimization - the algorithm to use for the gradient descent parameters update (e.g. adaptive)
#
# Algorithm:
#    * Forward and Backward propagation 

class DLLayer:
    def __init__(self, name, num_units, input_shape : tuple, activation="relu", W_intialization = "random", learning_rate = 1.2, optimization=None):
        self.name = name
        self.alpha = learning_rate
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self.prediction_function = activation
        self._optimization = optimization

        self.random_scale = 0.01
        self.activation_trim = 0.0000000001
        self._activation_forward = activation;
        
        if (activation == "leaky_relu"):
            self.leaky_relu_d = 0.01 # default value

        if (optimization == "adaptive"):
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, self._input_shape[0]), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5

        self.init_weights(W_intialization) 
    
    def init_weights(self, W_intialization):
        self.b = np.zeros((self._num_units,1), dtype=float) # b is init to zeros, always
        if (W_intialization == "random"):
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        elif (W_intialization == "zeros"):
            self.W = np.zeros((self._num_units, *self._input_shape), dtype=float)
        else:
            print("Invalid W_intialization type")
    
    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"

        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"

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
        return s
    
    # activation functions
    # ---------------------
    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
   
    def _relu(self, Z):
        return np.maximum(0,Z)
    
    def _leaky_relu(self, Z):
        return np.maximum(self.leaky_relu_d*Z,Z)
    
    def _tanh(self, Z):
        return np.tanh(Z)
    
    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))

            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = 1/(1+np.exp(-Z))

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
    
    def activation_forward(self, Z):
        if self._activation_forward == "sigmoid":
            return self._sigmoid(Z)
        
        elif self._activation_forward == "relu":
            return self._relu(Z)
        
        elif self._activation_forward == "leaky_relu":
            return self._leaky_relu(Z)
        
        elif self._activation_forward == "tanh":
            return self._tanh(Z)
        
        elif self._activation_forward == "trim_sigmoid":
            return self._trim_sigmoid(Z)
            
        elif self._activation_forward == "trim_tanh":
            return self._trim_tanh(Z)
        else:
            print("Invalid activation type")
            return None
        
    def prediction(self, Z):
        if self.prediction_function == "sigmoid":
            return self._sigmoid(Z)
        
        elif self.prediction_function == "relu":
            return self._relu(Z)
        
        elif self.prediction_function == "leaky_relu":
            return self._leaky_relu(Z)
        
        elif self.prediction_function == "tanh":
            return self._tanh(Z)
        
        elif self.prediction_function == "trim_sigmoid":
            return self._trim_sigmoid(Z)
        
        elif self.prediction_function == "trim_tanh":
            return self._trim_tanh(Z)
        else:
            print("Invalid activation type")
            return None
        

    # forward propagation
    # -------------------    
    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = A_prev
        self.Z = np.dot(self.W, self._A_prev) + self.b
        if (is_predict):
            self.A = self.prediction(self.Z)
        else:
            self.A = self.activation_forward(self.Z)
        return self.A
    

    # backword activation functions
    # -----------------------------    
    def _sigmoid_backward(self,dA):
        A = self._sigmoid(self.Z)
        dZ = dA * A * (1-A)
        return dZ

    def _relu_backward(self,dA):
        dZ = np.where(self.Z <= 0, 0, dA)
        return dZ
    
    def _leaky_relu_backward(self,dA):
        dZ = np.where(self.Z <= 0, self.leaky_relu_d * dA, dA)
        return dZ
    
    def tanh_backward(self,dA):
        dZ = dA * (1 - np.power(self._tanh(self.Z), 2))
        return dZ

    def activation_backward(self, dA):
        if self._activation == "sigmoid":
            return self._sigmoid_backward(dA)
        elif self._activation == "relu":
            return self._relu_backward(dA)
        elif self._activation == "leaky_relu":
            return self._leaky_relu_backward(dA)
        elif self._activation == "tanh":
            return self.tanh_backward(dA)
        elif self._activation == "trim_sigmoid":
            return self._sigmoid_backward(dA)
        elif self._activation == "trim_tanh":
            return self.tanh_backward(dA)
        else:
            print("Invalid activation type")
            return None
        
    # backword propagation
    # --------------------    
    def backward_propagation(self, dA):
        m = self._A_prev.shape[1]
        dZ = self.activation_backward(dA)
        self.dW = np.dot(dZ, self._A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev

    # update parameters
    # -----------------
    def update_parameters(self):
        if self._optimization == "adaptive":    # Update parameters with adaptive learning rate. keep the sign positive. Update is multiply by the derived value
            self._adaptive_alpha_b = np.where(self.db * self._adaptive_alpha_b >= 0, self._adaptive_alpha_b * self.adaptive_cont, self._adaptive_alpha_b * self.adaptive_switch)
            self._adaptive_alpha_W = np.where(self.dW * self._adaptive_alpha_W >= 0, self._adaptive_alpha_W * self.adaptive_cont, self._adaptive_alpha_W * self.adaptive_switch)
            self.W -= self._adaptive_alpha_W * self.dW
            self.b -= self._adaptive_alpha_b * self.db
        else:
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.db

