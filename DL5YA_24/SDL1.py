import numpy as np
import matplotlib.pyplot as plt

class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=0.01, optimization=None):
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._optimization = optimization
        self._alpha = learning_rate
        self._name = name
        self._activation_forward = self.activation
        self._activation_back
        # Activation parameters
        self._activation_trim = 1e-10
        if activation == "leaky_relu":
            self._leaky_relu_d = 0.01

        # Optimization parameters
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self._alpha)
            self._adaptive_alpha_W = np.full(self._get_W_shape(), self._alpha)
        self.adaptive_cont = 1.1
        self.adaptive_switch = 0.5

        self.random_scale = 0.01
        # Initialize weights
        self.init_weights(W_initialization)
        

    def init_weights(self, W_initialization):
       self.b = np.zeros((self._num_units, 1), dtype=float)

       if W_initialization == "zeros":
           self.W = np.zeros((self._num_units, *self._input_shape), dtype=float)
       elif W_initialization == "random":
           self.W = np.random.randn(self._num_units, *self._input_shape) * 0.01


    def _get_W_shape(self):
        return (self._num_units, *self._input_shape)

    #activation functions
        
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    def relu(self, Z):
        return np.maximum(0, Z)
    def leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self._leaky_relu_d)
    def tanh(self, Z):
        return np.tanh(Z)
    

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = 1/(1 + np.exp(-Z))

        TRIM = self._activation_trim

        if TRIM > 0:
            A = np.where(A < TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)

        return A

    def _trim_tanh(self, Z):
        A = np.tanh(Z)

        TRIM = self.activation_trim

        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)

        return A

    def activation(self, Z):
        if self._activation == "sigmoid":
            return self.sigmoid(Z)
        elif self._activation == "relu":
            return self.relu(Z)
        elif self._activation == "leaky_relu":
            return self.leaky_relu(Z)
        elif self._activation == "tanh":
            return self.tanh(Z)
        elif self._activation == "trim_tanh":
            return self._trim_tanh(Z)
        elif self._activation == "trim_sigmoid":
            return self._trim_sigmoid(Z)
        else:
            raise Exception("Activation function not implemented")
        
    def activation_forward(self, Z):
        return self.activation(Z)
                      
   
    def forward_propagation(self, A_prev, is_predict=False):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, self._A_prev) + self.b
        A = self._activation_forward(self._Z)

        if is_predict:
            self.A = self.activation(self._Z)
        else:
            self.A = self.activation_forward(self._Z)

        return self.A

    def _sigmoid_backward(self,dA):
        A = self.sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ

    def _relu_backward(self,dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu_backward(self,dA):
        return np.where(self._Z <= 0, dA * self._leaky_relu_d, dA)
        # dZ = np.where(self._Z <= 0, self.leaky_relu_d, dA)
        # return dZ
    # create this funciotn

    def _tanh_backward(self, dA):
        return dA * (1 - np.power(self.tanh(self._Z),2))
        

    def _trim_sigmoid_backward(self, dA):
        return self._sigmoid_backward(dA)
    
    def _trim_tanh_backward(self, dA):
        return self._tanh_backward(dA)

    def activation_backward(self, dA):
        if (self._activation == "relu"):
            return self._relu_backward(dA)
        elif (self._activation == "leaky_relu"):
            return self._leaky_relu_backward(dA)
        elif (self._activation == "sigmoid"):
            return self._sigmoid_backward(dA)
        elif (self._activation == "tanh"):
            return self._tanh_backward(dA)
        elif (self._activation == "trim_sigmoid"):
            return self._trim_sigmoid(dA)
        elif (self._activation == "trim_tanh"):
            return self._trim_tanh(dA)
        
    
    
    # def backward_propagation(self, dA):
    #     dZ = self.activation_backward(dA)
    #     self.dW = np.dot(dZ, self._A_prev.T) / self._A_prev.shape[1]
    #     self.db = np.sum(dZ, axis=1, keepdims=True) / self._A_prev.shape[1]
    #     dA_prev = np.dot(self.W.T, dZ)
    #     return dA_prev
    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = dA.shape[1]

        self.dW = (1/m) * np.dot(dZ, self._A_prev.T)
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

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
            s += "\t\t\tleaky_relu_d: " + str(self._leaky_relu_d) + "\n"

        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self._alpha) + "\n"

        # Optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"

        # Parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape) + "\n"

        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()

        return s
    


class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        self.loss_function = None
        self.loss_forward = None
        self.loss_backward = None
        self.threshold = 0.5
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function, decision_threshold=0.5):
        self.loss_function = loss_function
        self.decision_threshold = decision_threshold

    def compute_cost(self, Y, Y_pred):
        m = Y.shape[1]
        return -(1/m) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))

    def train(self, X, Y, epochs=100, learning_rate=0.01, optimization=None):
        for epoch in range(epochs):
            A = X
            for layer in self.layers:
                A = layer.forward_propagation(A)

            cost = self.compute_cost(Y, A)
            print(f'Epoch {epoch+1}/{epochs}, Cost: {cost}')

            dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

            for layer in reversed(self.layers):
                dA = layer.backward_propagation(dA)

                # Update parameters
                if hasattr(layer, 'update_parameters'):
                    layer.update_parameters()

    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward_propagation(A, is_predict=True)

        predictions = (A >= self.decision_threshold).astype(int)
        return predictions
    
    def squared_means(self, AL, Y):
        m = Y.shape[1]
        error = np.sum(np.square(AL - Y)) / m
        return error

    def squared_means_backward(self, AL, Y):
        m = Y.shape[1]
        dAL = 2 * (AL - Y) / m
        return dAL

    def cross_entropy(self, AL, Y):
        m = Y.shape[1]
        error = np.where(Y == 0, -np.log(1-AL), -np.log(AL))
        return error

    def cross_entropy_backward(self, AL, Y):
        m = Y.shape[1]
        dAL = np.where(Y == 0, 1/(1-AL), -1/AL)
        return dAL

    def compile(self, loss, threshold=0.5):
        self.loss_function = loss
        self.threshold = threshold

        if loss == "squared_means":
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward
        else:
            raise ValueError("Invalid loss function. Supported options: 'squared_means' or 'cross_entropy'.")

        self._is_compiled = True
        

    def cross_entropy(self, AL, Y):
        epsilon = 1e-15
        AL = np.clip(AL, epsilon, 1 - epsilon)
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return np.mean(error)

    def cross_entropy_backward(self, AL, Y):
        epsilon = 1e-15
        m = Y.shape[1]
        dAL = np.where(Y == 0, 1 / (1 - AL + epsilon), -1 / (AL + epsilon))
        return dAL / m
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        error = self.loss_forward(AL, Y)
        cost = np.sum(error) / m
        return cost
    

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
        result = X

        for l in range(1, L):
            result = self.layers[l].forward_propagation(result, is_predict=True)

        threshold = self.threshold
        prediction = (result > threshold).astype(int)

        return prediction
