import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



path = 'data.txt'


data = pd.read_csv(path , header = None , names = ['length' , 'width' , 'price'])


scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data))
data.insert(0 , 'ones' , 1)


print(data.head(5))

m = len(data)
cols = data.shape[1]


X = np.matrix(data.iloc[:,0:cols-1])
Y = np.matrix(data.iloc[:,cols-1])
theta = np.matrix(np.zeros((1,cols-1)))
theta = np.matrix(theta)
print(theta)


def J(X,Y,theta):
    z = np.power(X * theta.T - Y , 2)
    return  z.sum() / (2*m)


alpha = 0.01
iters = 1000
params = theta.shape[1]
print(params)

def GD(alpha , iters , X , Y , theta):
    errors = np.zeros(1000)
    temp = theta
    
    for i in range(iters):
        
        term = X*theta.T - Y
        
        for j in range(params):
            term = np.multiply(term, X[:,j])
            temp[0,j] = theta[0,j] - (alpha / m) * np.sum(term)
            
        theta = temp
        errors[i]  = J(X,Y,theta) 
    
    return theta , errors


errors = np.zeros(1000)

theta , errors= GD(alpha , iters , X , Y , theta)



plt.plot(np.arange(0,1000) , errors)


            
    

    

