import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10
from DL1_Comp import *

print (" --- Summerize DL1 Build ---")
print (" ---------------------------")

np.random.seed(1)

X, Y = u10.load_planar_dataset()


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

print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))[0]/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


print (" --- Targil 110.3.1 ---")
layer1 = DLLayer("Hidden 1", 4, (2,), "tanh", "random", 10)
layer2 = DLLayer("Output", 1, (4,), "sigmoid", "random", 100)

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

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))[0][0]/float(Y.size)*100) + '%')
"""
print (" --- Targil 110.3.3 ---")
X, Y = u10.load_planar_dataset()


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

"""