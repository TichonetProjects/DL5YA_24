# 110 All Exercises
# Gad Lidror
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10

#from DL1 import *
from DL1_Comp import *



print (" --- Targil 110.1.1 ---")
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

print (" --- Targil 110.1.2 ---")
Z = np.array([[1,-2,3,-4], [-10,20,30,-40]])
l[2].leaky_relu_d = 0.1

for i in range(1, len(l)):
    print(l[i].activation_forward(Z))

print (" --- Targil 110.1.3 ---")
np.random.seed(2)
m = 3
X = np.random.randn(4000,m)
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    print('layer',i," A", str(Al.shape), ":\n", Al)

print (" --- Targil 110.1.4 ---")
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)
    dZ = l[i].activation_backward(Al)
    print('layer',i," dZ", str(dZ.shape), ":\n", dZ)

print (" --- Targil 110.1.5 ---")
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)
    
np.random.seed(3)
fig, axes = plt.subplots(1, 4, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
dAl = np.random.randn(Al.shape[0],m) * np.random.randint(-100, 100+1, Al.shape)

for i in reversed(range(1,len(l))):
    axes[i-1].hist(dAl.reshape(-1), align='left')
    axes[i-1].set_title('dAl['+str(i)+']')
    dAl = l[i].backward_propagation(dAl)
plt.show()


print (" --- Targil 110.1.6 ---")
np.random.seed(4)
l1 = DLLayer("Hidden1", 3, (4,),"trim_sigmoid", "zeros", 0.2, "adaptive")
l2 = DLLayer("Hidden2", 2, (3,),"relu", "random", 1.5)

print("before update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

l1.dW = np.random.randn(3,4) * np.random.randint(-100,101)
l1.db = np.random.randn(3,1) * np.random.randint(-100,101)
l2.dW = np.random.randn(2,3) * np.random.randint(-100,101)
l2.db = np.random.randn(2,1) * np.random.randint(-100,101)

l1.update_parameters()
l2.update_parameters()

print("after update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

print (" --- Targil 110.2.3 ---")
np.random.seed(1)
m1 = DLModel()
AL = np.random.rand(4,3)
Y = np.random.rand(4,1) > 0.7
m1.compile("cross_entropy")
errors = m1.loss_forward(AL,Y)
dAL = m1.loss_backward(AL,Y)
print("cross entropy error:\n",errors)
print("cross entropy dAL:\n",dAL)
m2 = DLModel()
m2.compile("squared_means")
errors = m2.loss_forward(AL,Y)
dAL = m2.loss_backward(AL,Y)
print("squared means error:\n",errors)
print("squared means dAL:\n",dAL)

print (" --- Targil 110.2.4 ---")
print("cost m1:", m1.compute_cost(AL,Y))
print("cost m2:", m2.compute_cost(AL,Y))

print (" --- Targil 110.2.5 ---")
np.random.seed(1)
model = DLModel();
model.add(DLLayer("Perseptrons 1", 10,(12288,)))
model.add(DLLayer("Perseptrons 2", 1,(10,), "trim_sigmoid"))
model.compile("cross_entropy", 0.7)
X = np.random.randn(12288,10) * 256

print("predict:",model.predict(X))

print (" --- Targil 110.2.6 ---")
print(model)


print (" --- Summerize DL1 Build ---")
print (" ---------------------------")

np.random.seed(1)

X, Y = u10.load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y[0,:])
# Plot the decision boundary for logistic regression
u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
m = Y.size
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))[0]/float(m)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

print (" --- Targil 110.3.1 ---")

layer1 = DLLayer("Hidden 1", 4, (2,), "tanh", "random", 10,None)
layer2 = DLLayer("Output", 1, (4,), "sigmoid", "random", 100,None)

model = DLModel()
model.add(layer1)
model.add(layer2)
model.compile("cross_entropy", 0.5)

print(model)

print (" --- Targil 110.3.2 ---")
costs = model.train(X,Y,10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
m = Y.size
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))[0][0]/float(m)*100) + '%')

print (" --- Targil 110.3.3 ---")
#X, Y = u10.load_planar_dataset()
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = u10.load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = "noisy_moons"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y%2

layer1 = DLLayer("Hidden 1", 4, (2,), "tanh", "random", 10,None)
layer2 = DLLayer("Output", 1, (4,), "sigmoid", "random", 100,None)

model = DLModel()
model.add(layer1)
model.add(layer2)
model.compile("cross_entropy", 0.5)

costs = model.train(X,Y,10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
m = Y.size
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))[0][0]/float(m)*100) + '%')


# This function will 
