# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:54:14 2022

@author: sheha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




data = pd.read_csv('data1.txt' , names = ['size' , 'price'] , header = None)
data.insert(0,'ones',1)

m = data.shape[1]

#printing data
print('data: \n',data.head(5))

plt.scatter(data['size'] , data['price'])
plt.xlabel('size')
plt.ylabel('price')
plt.title('houses info')



cols = data.shape[1]
X = data.iloc[:,0:cols-1]
Y = data.iloc[:,cols-1:cols]


X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0,0]))
theta2 = np.matrix(np.zeros(2))

def J(theta , X, Y):
    z = np.power(((X * theta.T) - Y) , 2)
    return z.sum() / (2*len(X))
print(J(theta , X , Y))

a = 0.01
iters = 10000

def GradientD(X , Y , theta , alpha , iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    
    for i in range(iters):
        error = X * theta.T - Y
        
        for j in range(2):
            term = np.multiply(error , X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        
        theta = temp
    return theta



theta = GradientD(X, Y, theta, a, iters)

def hypotheis(x):
    return x * theta.T 

print(theta)
plt.plot(data['size'] , hypotheis(X))
plt.show()
