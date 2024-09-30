# 003 - Ex 2 and 3
# Gad Lidror
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(2)

print ("------------- Targil 003-Ex1 ------------")
a = random.randint(1,100)
b = random.randint(-100,100)
c = random.randint(-100,100)

print (f"Calculated X min =  {-1*b/(2*a)}")

x = 0
lr=0.5
ytag = prev_ytag = 2*a*x+b

for i in range (1000):
    if (prev_ytag*ytag < 0):
        lr = lr /2
        prev_ytag = ytag

    if (ytag>0):
        x=x-lr
    else:
        x=x+lr
    
    ytag = 2*a*x+b

print (f"Gradient Descent x = {x}")

print ("------------- Targil 003-Ex2 ------------")

# seperated the func and the derivative for more clear code
def f2(x):
    return 1.1*x*x - 7*x + 6

def f2_tag(x):
    return 2.2*x - 7

def g2(a):
    return (a**2 - 9) / 4**a

def g2_tag(a):
    return (2**(1+2*a) * a - np.log(2) * (2**(2*a+1))*(a**2-9)) / (16**a)


def train_min(alpha, epocs, func, funcD):
    x = random.randint(-10,10)
    for i in range(epocs):
        x = x - alpha * funcD(x)
    return (x,func(x), funcD(x))
    
def train_max(alpha, epocs, func, funcD):
    x = random.randint(-10,10)
    for i in range(epocs):
        x = x + alpha * funcD(x)
    return (x,func(x), funcD(x))
        
random.seed(5)
print("MIN INFO: ", train_min(0.001, 10000, f2, f2_tag))
print("MIN INFO: ", train_min(0.001, 100000, g2, g2_tag), "MAX INFO: ", train_max(0.001, 1000000, g2, g2_tag))

print ("------------- Targil 003-Ex3 ------------")

def f3(x):
    return x**3 - 107*x**2 - 9*x + 3

def f3_tag(x):
    return 3*x**2 - 214*x - 9

def train3_min(alpha, epocs, func, funcD):
    x = 0
    for i in range(epocs):
        x = x - alpha * funcD(x)
    return (x,func(x), funcD(x))
    
def train3_max(alpha, epocs, func, funcD):
    x = 0
    for i in range(epocs):
        x = x + alpha * funcD(x)
    return (x,func(x), funcD(x))

#random.seed(42)

print("MAX INFO: ", train3_max(0.001, 1000000, f3, f3_tag))
print("MIN INFO: ", train3_min(0.001, 1000000, f3, f3_tag))