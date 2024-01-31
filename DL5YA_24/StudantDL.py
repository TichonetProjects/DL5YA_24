import numpy as np
import matplotlib.pyplot as plt
import random

class DLLayer:  
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate = 0.01, optimization = "None"):
        self._name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._learning_rate = learning_rate
        self._optimization = optimization


        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self._learning_rate)
            self._adaptive_alpha_W = np.full((self._num_units, *(self._input_shape)), self._learning_rate)

        self.adaptive_cont = 1.1
        self.adaptive_switch = -0.5

        if (activation == "relu"):
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
            
        
        self.random_scale = 0.01
        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):

        self.b = np.zeros((self._num_units,1), dtype=float)
        if W_initialization == "zeros":
            self.W = np.zeros((self._num_units, *(self._input_shape)), dtype=float)
        else:
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
    
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
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, self._A_prev) + self.b
        return self.activation_forward(self._Z)

    # backwords:
    def _sigmoid_backward(self, dA):
        A = self.__sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ

    def _leaky_relu_backward(self, dA):
        return np.where(self._Z < 0, dA * self.leaky_relu_d, dA)    

    def _relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ
    
    def _tanh_backward(self, dA):
        return  dA * (1- np.power(self.__tanh(self._Z),2))
    
    def _trim_sigmoid_backward(self, dA):
        return self._sigmoid_backward(dA)
    
    def _trim_tanh_backward(self, dA):
        return self._tanh_backward(dA)
 
    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        self.dW = np.dot(dZ, self._A_prev.T) / self._A_prev.shape[1]
        self.db = np.sum(dZ, axis=1, keepdims=True) / self._A_prev.shape[1]
        dA_prev = np.dot(self.W.T, dZ)
        return dA_prev

    def update_parameters(self):
        if self._optimization == "adaptive":
            self._adaptive_alpha_W = np.where(self._adaptive_alpha_W * self.dW > 0, self._adaptive_alpha_W * self._adaptive_cont, self._adaptive_alpha_W * self._adaptive_switch)
            self._adaptive_alpha_b = np.where(self._adaptive_alpha_b * self.db > 0, self._adaptive_alpha_b * self._adaptive_cont, self._adaptive_alpha_b * self._adaptive_switch)
            self.W -= self._adaptive_alpha_W * self.dW
            self.b -= self._adaptive_alpha_b * self.db
        else:
            self.W -= self._alpha * self.dW
            self.b -= self._alpha * self.db
            


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
        
    def add(self, layer):
        self.layers.append(layer)
        
    def _squared_means(self, AL, Y):
        m = Y.shape[1]
        error = np.sum(np.square(AL - Y)) / m
        return error

    def _squared_means_backward(self, AL, Y):
        m = Y.shape[1]
        dAL = 2 * (AL - Y) / m
        return dAL

    def _cross_entropy(self, AL, Y):
        m = Y.shape[1]
        error = np.where(Y == 0, -np.log(1-AL), -np.log(AL))
        return error

    def _cross_entropy_backward(self, AL, Y):
        m = Y.shape[1]
        dAL = np.where(Y == 0, 1/(1-AL), -1/AL)
        return dAL
    
    def compile(self, loss, threshold=0.5):
        if loss == "squared_means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward
        else:
            raise ValueError("Unsupported loss function. Choose 'squared_means' or 'cross_entropy'.")
        
        self.loss = loss
        self.threshold = threshold
        self._is_compiled = True
    
    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)
        cost = np.sum(errors) / m
        return cost
        
    def loss_backward(self, AL, Y):
        return self.loss_backward(AL, Y)
    
    # traning function
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
     
    # question 2.5    
    def predict(self, X):
        Al = X
        for l in range(1,len(self.layers)):
            Al = self.layers[l].forward_propagation(Al, True)
        return Al > self.threshold
    

    # question 2.6
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s