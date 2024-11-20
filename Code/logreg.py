import numpy as np
import math


class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        self.theta = None
        self.alpha = alpha          # learning rate
        self.regLambda = regLambda   # regularization parameter
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters



        
    def computeCost(self, theta, X, y, regLambda):

        n = len(y)  # number of rows in X

        # cost function with regularization
        costVal = (-1 / n) * (np.dot(y.T, np.log(self.sigmoid(np.dot(X, theta)))) +  np.dot((1 - y).T, np.log(1 - self.sigmoid(np.dot(X, theta)))))   +   (regLambda / (2 * n)) * np.sum(np.square(theta[1:]))

        return costVal  # scalar value of cost
    



    def computeGradient(self, theta, X, y, regLambda):
        n = len(y)  # number of rows in X
        
        # gradient for logistic regression with regularization
        gradVal = (1 / n) * (np.dot(X.T, (self.sigmoid(np.dot(X, theta)) - y)))
        
        # no regularization for the bias term
        gradVal[1:] += (regLambda / n) * theta[1:]
        
        return gradVal
    



    def hasConverged(self, oldTheta, newTheta):
        
        # the sum of the squares of (new theta - old theta) from 1 to n inside the square root
        difference = newTheta - oldTheta
        
        # euclidean
        distance = np.sqrt(np.sum(difference**2))
        
        # distance is less than epsilon -> the model has converged
        return distance < self.epsilon
    



    # train the logistic regression model
    def fit(self, X, y):

        y = y.reshape(-1, 1)  # (n,) -> (n, 1) to ensure y is a column vector

        n, d = X.shape  # n: rows , d: columns
        X = np.c_[np.ones((n, 1)), X]  # add a column of ones for the bias term

        np.random.seed(0)
        self.theta = np.random.rand(d + 1, 1)  # initialize theta with random values

        for i in range(self.maxNumIters):
            
            # update theta using gradient descent
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            newTheta = self.theta - self.alpha * gradient

            # if the model has converged, stop the optimization process
            if self.hasConverged(self.theta, newTheta):
                self.theta = newTheta
                print(f"Model converged at iteration {i}")
                break

            # if not converged, update theta
            self.theta = newTheta

            # print cost for debugging every 500 iterations
            if i % 500 == 0:
                cost = self.computeCost(self.theta, X, y, self.regLambda)
                print(f"Iteration {i}, Cost: {cost}")


                

    def predict(self, X):

        n = X.shape[0]  # get the number of samples
        
        # add a column for bias term
        X = np.c_[np.ones((n,1)), X]  # add a column of ones for the bias term

        predictedValue = self.sigmoid(np.dot(X, self.theta))
        
        return (predictedValue >= 0.5).astype(int)  # return 0/1 according to the threshold
    



    def sigmoid(self, z):

        # 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-z))





#######  Written by  #######
######    Zer0-Bug    ######