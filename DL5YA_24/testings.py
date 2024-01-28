from StudantDL import *
import numpy as np
np.random.seed(1)
l = [None]

l.append(DLLayer("Hidden 1", 6, (4000,)))
#l[1].ab()
#print(l[1])

l.append(DLLayer("Hidden 2", 12, (6,), "leaky_relu", "random", 0.5, "adaptive"))

l[2]._adaptive_cont = 1.2

#print(l[2])

l.append(DLLayer("Neurons 3", 16, (12,), "tanh"))

#print(l[3])

l.append(DLLayer("Neurons 4", 3, (16,), "sigmoid", "random", 0.2, "adaptive"))

l[4]._adaptive_alpha_W = 10.0
l[4].init_weights("random")

#print(l[4])

Z = np.array([[1, -2, 3, -4], [-10, 20, 30, -40]])

l[2]._leaky_relu_d = 0.1

for i in range(1, len(l)):
    print(l[i].activation_forward(Z))

np.random.seed(2)

m = 3

X = np.random.randn(4000, m)

Al = X

for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    print('layer', i, " A", str(Al.shape), ":\n", Al)

Al = X

for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)   
    dZ = l[i].activation_backward(Al)  
    print('layer', i, " dZ", str(dZ.shape), ":\n", dZ)
