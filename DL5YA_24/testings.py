
from DL1 import *


# Exc 1.1:
print("Exc 1.1:")
np.random.seed(1)

l = [None]
l.append(DLLayer("Hidden 1", 6, (4000,)))

print(l[1])
l.append(DLLayer("Hidden 2", 12, (6,), "leaky_relu", "random", 0.5,"adaptive"))

l[2].adaptive_cont = 1.2
print(l[2])
l.append(DLLayer("Neurons 3",16, (12,),"tanh"))
print(l[3])

l.append(DLLayer("Neurons 4",3, (16,),"sigmoid", "random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
print(l[4])


# Exc 1.2:
print("\n\nExc 1.2:")
Z = np.array([[1,-2,3,-4], [-10,20,30,-40]])

l[2].leaky_relu_d = 0.1

for i in range(1, len(l)):
    print(l[i].activation_forward(Z))

# Exc 1.3:
print("\n\nExc 1.3:")
np.random.seed(2)
m = 3
X = np.random.randn(4000,m)
Al = X

for i in range(1, len(l)):

    Al = l[i].forward_propagation(Al, True)
    print('layer',i," A", str(Al.shape), ":\n", Al)
