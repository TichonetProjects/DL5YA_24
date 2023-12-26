import unit10.b_utils as u10
import matplotlib.pyplot as plt
import numpy as np
import random


print ("------------- Targil 006-Ex1 ------------")


# Function to calculate COST function and the derivatives of W and b
# Input: Data (X)
#        Real result (Y)
#        Parameters to check (W, b)
def calc_J(X, Y, W, b):
    m = len(Y)                  # Number of samples
    n = len(W)                  # Size of input
    dW, db = np.zeros(n), 0     # Set initial values for dW and db
    J = 0                       # Set initial values for the cost function 
    
    for i in range(m):          # Iterate over all of the samples
       y_hat_i = b              # Calculate Y_hat = w1*x1 + w2*x2+...wn*xn + b 
       for j in range(n):
         y_hat_i += W[j]*X[j][i]
       diff = (float)(y_hat_i-Y[i]) # find the error for each sample
       J += (diff**2)/m         # add the error to the COST J

       for j in range(n):       # Calculate the derivative for each W and b
         dW[j] += 2*X[j][i]*(diff/m)
       db += 2 * diff/m

       dW, db = boundary(dW, db)
    return J, dW, db            # return the cost and derivatives 

# Train function using adaptive algorithm
def train_n_adaptive(X, Y, alpha, num_iterations, calc_J):
    m,n = len(Y), len(X)
    costs, W, alpha_W, b = [],[],[],0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha
    
    for i in range(num_iterations):
        cost,dW,db = calc_J(X,Y,W,b)
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j]*alpha_W[j] > 0 else -0.5
        alpha_b *= 1.1 if db*alpha_b > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        b -= alpha_b
        if ((i%10000)==0 and i != 0):
            print('Iteration :' + str(i) + "  cost= " + str(cost))
            costs.append(cost)
    return costs[1:], W, b


# Train function that combine the two algorithms - derivative and adaptive
def train_n_combined(X, Y, alpha, epocs, func):
    random.seed(1)
    n = X.shape[0]
    lrW = np.full(n,alpha)
    lrb = alpha
    W = np.zeros(n)
    b = 0
    SaveJ = []
    
    J, SavedW, Savedb = func(X, Y, W, b)
    for i in range (epocs):
        J, dW, db = func(X, Y, W, b)
 
        for j in range (len(W)):
            if (dW[j]*SavedW[j]< 0):
                lrW[j] *= 0.5   # Note: sign is not changed becuase the derivative give the sign to the calculation
            else: 
                lrW[j] *= 1.1
        if (db*Savedb < 0):
            lrb *= 0.5
        else: 
            lrb *= 1.1

        W -= lrW*dW
        b -= lrb*db
            
        SavedW = dW     # save the derivatives so direction change can be checked
        Savedb = db

        if ((i%10000) == 0 and i != 0):
            SaveJ.append(J)  
            print ("i=",i,"J=",J)
    return SaveJ, W, b

# ------------------------------------------------
# Mechanism for bounding the upper and lower numbers
# limits of the computer
# ------------------------------------------------
MAX_D = 1e+100
def boundary1(d):
    return  MAX_D if d > MAX_D else -MAX_D if d < -MAX_D else d
def boundary(dW, db):
    for i in range(len(dW)):
        dW[i] = boundary1(dW[i])
    return dW, boundary1(db)
# ------------------------------------------------


X, Y = u10.load_dataB1W4_trainN()
X = np.array(X)
Y = np.array(Y)

###costs, W, b = train_n_adaptive(X, Y, 0.0001, 150000, calc_J)
###print('w1='+str(W[0])+', w2='+str(W[1])+', w3='+str(W[2])+', w4='+str(W[3])+", b="+str(b))
###plt.plot(costs)
###plt.ylabel('cost')
###plt.xlabel('iterations (per 10,000)')
###plt.show()


print (f"------------- Targil 006-Ex2 ------------")


# a General polinom function from 'low' to 'high'
def PolinomFunction(x, W, low, high):
    y = 0
    for i in range(low,high+1):
        y += W[i-low]*x**i
    return y

# a parabula function that returns the cost and the derivatives of 
# the parametera a, b, c (W[0], W[1], b)
def func_Ex2(X, Y, W, b):
    m = len(Y)
    dW, db = [0,0], 0
   
    # seting the parameters so I can use a general polinom function 
    # to calculate the parabula
    poli_params = []
    poli_params.append (b)      # X**0
    poli_params.append (W[1])   # X**1
    poli_params.append (W[0])   # X**2

    Y_hat = PolinomFunction(X, poli_params, 0, 2)
    diff = Y_hat - Y

    dW[0] = 2 * sum(X**2 * diff) / m
    dW[1] = 2 * sum(X * diff) / m
    db    = 2 * sum(diff) / m

    cost = sum(diff**2) / m
    
    return cost, dW, db

# non-adaptive train the parabula function
def train_Ex2(X, Y, alpha, epochs, func):
        W = [0,0]
        b = 0

        for i in range(epochs):
            cost, dW, db= func(X, Y, W, b)
            W[0] -= dW[0] * alpha
            W[1] -= dW[1] * alpha
            b    -= db * alpha

        return cost, W, b


X2 = np.array([i for i in range(-10, 17)])
Y2 = np.array([230.0588, 160.9912, 150.4624, 124.9425, 127.4042, 95.59201, 69.67605, 40.69738, 28.14561, 14.42037, 7, 0.582744,
              -1.27835, -15.755, -24.692, -23.796, 12.21919, 9.337909, 19.05403, 23.83852, 9.313449, 66.47649, 10.60984, 77.97216, 27.41264, 149.7796, 173.2468])

J, W, b = train_Ex2(X2, Y2, 0.00001, 10000, func_Ex2)

print(f"Cost: {J}; f(x) = {W[0]}*x^2 + {W[1]}*x + {b}")

x_graph = np.linspace(-25, 25)
plt.plot(X2, Y2, 'r.')

poli_params = []
poli_params.append (b)      # X**0
poli_params.append (W[1])   # X**1
poli_params.append (W[0])   # X**2

plt.plot(x_graph, PolinomFunction(x_graph, poli_params, 0, 2))
plt.title("Parabola Gradient Descent")
plt.show()




print ("------------- Targil 006-Ex3 ------------")

random.seed(42)     # 42 is a significant umber :)

def func_Ex3(X, Y, W, low, high):
    m = len(Y)
    n = high-low+1
    dW = np.zeros(n)
    cost = 0

    Y_hat = PolinomFunction(X, W, low, high)
    diff = Y_hat - Y

    for j in range(high-low+1):
        dW[j] = 2*sum(diff*(X**(j+low)) ) /m

    cost = sum(diff**2) / m

    return cost, dW # the dW inckudes akso the db

def train_n_adaptive_exc3(X, Y, low, high, alpha, epochs, func):
    m = len(Y)
    n = high-low+1

    costs = []
    W = [random.uniform(-1, 1) for i in range(n)]  ## random initial values for W
    prev_dW = np.zeros(n)
    alphas = [alpha for i in range(n)]  # a list of alphas 

    for i in range(epochs):
        cost, dW = func(X, Y, W, low, high)

        for j in range(n):
            if (prev_dW[j] * dW[j] < 0): alphas[j] *= 0.5
            else: alphas[j] *= 1.1

            W[j] -= alphas[j] * dW[j]
            prev_dW[j] = dW[j]

        if (i != 0 and i % 10000 == 0):
            costs.append(cost)
            print(f'({i}): J = {cost}')
            
    return costs, W


X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
Y = np.array([1.2, 33693.8, 28268.28, 13323.43, -3.4, -9837.28, -17444.1, -16850.2, -13581.9, 12, 20232.34, 40051.71, 74023.28, 96583.9, 123678.4, 152887.4, 141921.4,
                137721.4, 99155.27, 17.8, -135173])

low , high = 0,10 # Power low-high
m = len(Y)
n = high-low+1

costs, W = train_n_adaptive_exc3(X, Y, low, high, 0.00000000000000001, 3000000 , func_Ex3)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10000)')
plt.show()

for i in range(n): print(f"W[{i}] = {W[i]}")

plt.plot(X, Y, 'b.')
plt.plot(X, PolinomFunction(X, W, low, high) )
plt.show()

