# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MOUNESH P
RegisterNumber: 212222230084

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("dataset/ex2data1.txt", delimiter = ",")
X = data[:, [0, 1]]
y = data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0],  X[y == 1][:, 1], label = "Admitted")
plt.scatter(X[y == 0][:, 0],  X[y == 0][:, 1], label = "Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10, 10 , 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun = cost, x0 = theta, args = (X_train, y), method = "Newton-CG", jac = gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
    plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
    
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
prob

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X)==y)

*/
```
## Output:
![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/1a4b03e0-dc0c-4356-b1f5-b3abef6f6aeb)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/575c955b-41cc-4fb9-a901-129c8868c1e7)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/0424d3b4-ece7-4453-affb-8d57bf543cb8)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/ece44596-87b2-47df-a0d4-e67fc0b51e85)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/73934e45-fd23-4be7-ac4b-ea264128400a)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/554e3664-3b53-4612-b8db-fbd4d9e96c6e)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/62a128cc-abc3-4f1b-bdf6-1696a7da67cb)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/ef1a73b4-b6de-490c-8029-d93afc3a6f92)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/19ce2970-512e-44b8-b020-039956233da7)

![image](https://github.com/Mounesh07/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343401/16b2f0ec-f84b-4111-9664-c2b91396cd23)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

