# 004 
# Gad Lidror
from functools import lru_cache
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random

print ("------------- Targil 004-Ex1 ------------")
alpha = 0.001

def f2(x):
    return 1.1*x*x - 7*x + 6, 2.2*x - 7

def g2(a):
    return (a**2 - 9) / 4**a , (2**(1+2*a) * a - np.log(2) * (2**(2*a+1))*(a**2-9)) / (16**a)

# This function will find the parameter (x) that is the closest 
# to the minimum (or maximum) point of the function, in epocs iterations
def train(alpha, epocs, func, minmax = 1):
    x = random.randint(-10,10)  # initial random value for parameter
    prev_d = func(x)[1]         # Get derivative for the initial point
    if (prev_d>0): alpha *= -1  # set initial direction of alpha

    for i in range(epocs):
        res = func(x)           # get the derivative of the current parameter

        if (res[1] == 0):       # if derivative is zero - we reached the minimum !
            print (f"Left after {i} iterations, instead of {epocs}")
            return (x,func(x))  

        # set the alpha according to the adaptive algorithm
        # -------------------------------------------------
        if (prev_d *  res[1] < 0): alpha *= -0.5
        else:                      alpha *=  1.1
            
        x = x + alpha*minmax           # correct the parameter 
        prev_d = res[1]         # save the current derived value
    return (x,func(x))


       
random.seed(5)

print("MIN INFO: ", train(0.001, 100000, f2, minmax=1))
print("MIN INFO: ", train(0.001, 100000, g2, minmax=1), "MAX INFO: ", train(0.001, 1000000, g2, minmax=-1))
