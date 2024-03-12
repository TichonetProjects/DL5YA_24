# 130 All Exercises
# Gad Lidror
import numpy as np
import matplotlib.pyplot as plt
from unit10 import c2w1_init_utils as u10
from DL2_Comp import *

# Stage 2 packages
from PIL import Image
from unit10 import c1w4_utils as u10_2

print (" --- Stage 1 ---")
print (" ---------------")

plt.rcParams['figure.figsize'] = (7.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()

print (" --- Targil 130.1.1 ---")
np.random.seed(1)
hidden1 = DLLayer("Perseptrons 1", 30,(12288,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075, optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2", 15,(30,),"trim_sigmoid",W_initialization = "He",learning_rate = 0.1)
print(hidden1)
print(hidden2)

print (" --- Targil 130.1.2 ---")

hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075)
hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir","Hidden1")
hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization = "SaveDir/Hidden1.h5",learning_rate = 0.1)
print(hidden1)
print(hidden2)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
dir = "model"
model.save_weights(dir)
print(os.listdir(dir))

print (" --- Targil 130.1.3 ---")

layer1 = DLLayer("Perseptrons 1", 10,(2,),"relu", W_initialization = "zeros",learning_rate = 10)
layer2 = DLLayer ("Perseptrons 2", 5, (10,),"relu", W_initialization = "zeros",learning_rate = 10)
output = DLLayer("Output", 1,(5,),"trim_sigmoid", W_initialization = "zeros",learning_rate = 1)
model = DLModel()
model.add(layer1)
model.add(layer2)
model.add(output)
model.compile(loss="cross_entropy", threshold=0.5) #compiling the model

costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
axes = plt.gca()
axes.set_ylim([0.65,0.75])
plt.title("Model with -zeros- initialization")
plt.show()


print (" --- Targil 130.1.4 ---")
layer1 = DLLayer("Perseptrons 1", 10,(2,),"relu", W_initialization = "random",learning_rate = 10, random_scale = 1)
layer2 = DLLayer ("Perseptrons 2", 5, (10,),"relu", W_initialization = "random",learning_rate = 10, random_scale = 1)
output = DLLayer("Output", 1,(5,),"trim_sigmoid", W_initialization = "random",learning_rate = 1, random_scale = 1)

model2 = DLModel()
model2.add(layer1)
model2.add(layer2)
model2.add(output)
model2.compile(loss="cross_entropy", threshold=0.5) #compiling the model

costs = model2.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(" random initialization")
plt.show()

plt.title("Model with -random- initialization")

axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model2.predict(x.T), test_X, test_Y)

predictions = model2.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T)).item()/float(train_Y.size)*100) + '%')
predictions = model2.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T)).item()/float(test_Y.size)*100) + '%')

print (" --- Targil 130.1.5 ---")

layer1 = DLLayer("Perseptrons 1", 10,(2,),"relu", W_initialization = "He",learning_rate = 10, random_scale = 1)
layer2 = DLLayer ("Perseptrons 2", 5, (10,),"relu", W_initialization = "He",learning_rate = 10, random_scale = 1)
output = DLLayer("Output", 1,(5,),"trim_sigmoid", W_initialization = "He",learning_rate = 1, random_scale = 1)

model2 = DLModel()
model2.add(layer1)
model2.add(layer2)
model2.add(output)
model2.compile(loss="cross_entropy", threshold=0.5) #compiling the model

costs = model2.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(" He initialization")
plt.show()

plt.title("Model with -He- initialization")

axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model2.predict(x.T), test_X, test_Y)

predictions = model2.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T)).item()/float(train_Y.size)*100) + '%')
predictions = model2.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T)).item()/float(test_Y.size)*100) + '%')


print (" --- Stage 2 ---")
print (" ---------------")

plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10_2.load_datasetC1W4()
# Example of a picture
index = 87
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[0,index]) + ". It's a " + classes[train_set_y[0,index]].decode("utf-8") +  " picture.")

print (" --- Targil 130.2.1 ---")

m_train = train_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
m_test = test_set_y.shape[1]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

print (" --- Targil 130.2.2 ---")

# Reshape the training and test examples 
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
# normelize data to have feature values between -0.5 and 0.5.
train_x = train_x_flatten/255 - 0.5
test_x = test_x_flatten/255 - 0.5

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print ("normelized train color: ", str(train_x[10][10]))
print ("normelized test color: ", str(test_x[10][10]))

print (" --- Targil 130.2.3 ---")

np.random.seed(5)
hidden1 = DLLayer("h1",7,(12288,),"relu",W_initialization = "Xaviar",learning_rate = 1)
output = DLLayer("output",1,(7,),"sigmoid",W_initialization = "Xaviar",learning_rate = 1)
model = DLModel()
model.add(hidden1)
model.add(output)
model.compile("cross_entropy")
costs = model.train(train_x, train_set_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_x) == test_set_y))


print (" --- Targil 130.2.4 ---")

np.random.seed(5)
lr = 1
hidden1 = DLLayer("h1",30,(12288,),"relu",W_initialization = "Xaviar",learning_rate = lr)
hidden2 = DLLayer("h2",15,(30,),"relu",W_initialization = "Xaviar",learning_rate = lr)
hidden3 = DLLayer("h3",10,(15,),"relu",W_initialization = "Xaviar",learning_rate = lr)
hidden4 = DLLayer("h4",10,(10,),"relu",W_initialization = "Xaviar",learning_rate = lr)
hidden5 = DLLayer("h5",5,(10,),"relu",W_initialization = "Xaviar",learning_rate = lr)
output = DLLayer("output",1,(5,),"sigmoid",W_initialization = "Xaviar",learning_rate = lr)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.add(hidden4)
model.add(hidden5)
model.add(output)
model.compile("cross_entropy")
costs = model.train(train_x, train_set_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_x) == train_set_y))
print("test accuracy:", np.mean(model.predict(test_x) == test_set_y))

print (" --- Targil 130.2.5 ---")

#Test your image
img_path = r'unit10\cat1.png'   # full or relative path of the image
my_label_y = [1]                # the true class of your image (1 -> cat, 0 -> non-cat)
img = Image.open(img_path)
img_rgb = img.convert('RGB')  # Convert image to RGB
image64 = img_rgb.resize((num_px, num_px), Image.LANCZOS)
plt.imshow(img)
plt.show()
plt.imshow(image64)
plt.show();
my_image = np.reshape(image64,(num_px*num_px*3,1))
my_image = my_image/255. - 0.5
p = model.predict(my_image)
print ("L-layer model predicts a \"" + classes[int(p.item()),].decode("utf-8") +  "\" picture.")


#This is a new line
