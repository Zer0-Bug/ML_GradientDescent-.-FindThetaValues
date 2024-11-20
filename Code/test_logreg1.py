from numpy import loadtxt, ones, zeros, where
import numpy as np
from pylab import plot, legend, show, where, scatter, xlabel, ylabel, linspace, contour, title
import matplotlib.pyplot as plt
from logreg import LogisticRegression

if __name__ == "__main__":
    # Load Data
    filename = 'data1.dat'
    data = loadtxt(filename, delimiter=',')
    X = data[:, 0:2]
    y = np.squeeze(np.array([data[:, 2]]).T)
    n, d = X.shape

    # Standardize the data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # train logistic regression
    logregModel = LogisticRegression(regLambda=0.00000001)
    logregModel.fit(X, y)

    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logregModel.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    # Configure the plot display
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

