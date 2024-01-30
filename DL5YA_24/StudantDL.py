
import numpy as np
import matplotlib.pyplot as plt

class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=0.01, optimization=None, activation_forward = None ):
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._optimization = optimization
        self._alpha = learning_rate
        self._name = name
        self._activation_forward = activation_forward if activation_forward is not None else self.activation

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
    
    def forward_propagation(self, A_prev, is_predict):
        Z = np.dot(self.W, A_prev) + self.b
        A = self._activation_forward(Z)
        
        if not is_predict:
            self._A_prev = A_prev
            self._Z = Z

        return A

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

