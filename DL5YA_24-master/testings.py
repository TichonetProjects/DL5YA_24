import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random
import unit10.c1w2_utils as u10
#from MyVector import *

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

# Example of a picture
index = 4 # change index to get a different picture
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[index]) + ", it's a '" + 
classes[np.squeeze(train_set_y[index])].decode("utf-8") +  "' picture.")

m_train = train_set_y.shape[0]
m_test = test_set_y.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
