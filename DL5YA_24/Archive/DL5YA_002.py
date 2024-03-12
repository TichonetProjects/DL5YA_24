# 002
# Gad Lidror
from typing import DefaultDict
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

import unit10.b_utils as u10   # Our file with all the data for exercises

import random
random.seed(1)

X, Y = u10. load_dataB1W3Ex2()  # Load the sample data

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(X[0], X[1], Y)
#plt.show()

# ---------------------------------------

X, Y = u10.load_dataB1W3Ex2()   # Get the data from the exercise file
X1 = np.array(X[0])             # Extract the X1 from the data. Save as numpy array
X2 = np.array(X[1])             # Extract the X2 from the data
Y  = np.array(Y)                # Extract the Y from the data


# function to calculate the given function
# using the paraemters we guesed (a, b, c, d)
# note: using numpy so work is done on the whole list with one command. 
#       No need for loops
def y_function(X1, X2, a, b, c, d):
    calc = a*X1**2 + b*X1*X2 + c*X2**2 + d
    return calc


# The cost function J
# This function get the data (X1 and X2), the real result value (Y)
# and the 4 current parameters we set (a, b, c, d)
# functionality: Calculate for each input, the function (predict)
#                Calculate the sum of the errors of all the samples 
#                and divide by num of samples
# return: the cost of those parameters (a, b, c, d)
def J_function(X1, X2, Y, a, b, c, d):
    Y_hat = y_function(X1, X2, a, b, c, d)
    error = (Y_hat - Y)**2
    return np.sum(error) / len(X1)


# set first values to the result (parameters and cost)
min_a, min_b, min_c, min_d = random.randint(-10,10),random.randint(-10,10),random.randint(-10,10),random.randint(-10,10)
min_cost = J_function(X1,X2,Y,min_a, min_b, min_c, min_d)

# do the guesing 999 more times to get to the 1000 as requested
for i in range(999):
    a = random.randint(-10,10)
    b = random.randint(-10,10)
    c = random.randint(-10,10)
    d = random.randint(-10,10)
    cost = J_function(X1,X2,Y,a,b,c,d)  # calculte cost to the currernt guesing
    if (cost < min_cost):   # set the parameters that have the minimum cost
        min_a = a
        min_b = b
        min_c = c
        min_d = d
        min_cost = cost

print(f"Best values for the parameters: a= {min_a}, b= {min_b}, c= {min_c}, d= {min_d}. min cost= {min_cost}")


fig = plt.figure()
ax = plt.axes(projection="3d")
# set X1 and X2 to the grid values between -15 and 15
X1, X2 = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15,30))

# calculte for each of the grid points the values of Y hat, using the
# parameters we found
Ywire = y_function(X1, X2, min_a, min_b, min_c, min_d)

ax.plot_wireframe(X1, X2, Ywire, color='orange')
ax.scatter3D(X[0], X[1], Y);
plt.show()

