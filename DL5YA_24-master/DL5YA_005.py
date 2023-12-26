# 005
# Gad Lidror
import unit10.b_utils as u10
import matplotlib.pyplot as plt
import numpy as np

import random


def calc_J(X, Y, a, b):
    Y_hat = a*X+b        # derivative for a is X. derivative for b is 1
    m = len(Y)
    J = sum((Y_hat-Y)**2)/m
    da = (2/m)*sum((Y_hat-Y)*X)
    bd = (2/m)*sum(Y_hat-Y)
    return J, da, bd

def train_adaptive(X, Y, alpha, epocs, func):
    random.seed(1)
    a, b = random.randint(-10,10), random.randint(-10,10)
    for i in range (epocs):
        J, da, db = func(X, Y, a, b)
        a -= alpha*da  # a = a-
        b -= alpha*db
    return J, a, b
    

X1, Y1 = u10.load_dataB1W3Ex1()
X = np.array(X1)
Y = np.array(Y1)

J, a, b = train_adaptive(X, Y, 0.0001, 100, calc_J)
print('J='+str(J)+', a='+str(a)+", b="+str(b))
plt.plot(X, Y, 'r.')
plt.plot([0,100],[a*0+b, 100*a+b])
plt.show()
