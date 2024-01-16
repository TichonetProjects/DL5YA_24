import numpy as np
import matplotlib.pyplot as plt
import random
from DL1 import *



np.random.seed(1)
l = [None]
l.append(DLLayer("Hidden 1", 6, (4000,)))
print(l[1])
l.append(DLLayer("Hidden 2", 12, (6,),"leaky_relu", "random", 0.5,"adaptive"))
l[2].adaptive_cont = 1.2
print(l[2])
l.append(DLLayer("Neurons 3",16, (12,),"tanh"))
print(l[3])
l.append(DLLayer("Neurons 4",3, (16,),"sigmoid", "random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
print(l[4])
