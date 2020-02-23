# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:18:05 2020

@author: Liam
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#TODO: normalize the input variables (e.g. 0-1), and apply weights to them !






#NOTATION: Throughout these comments, the number of data points is N, while
#the number of parameters ('X values') is M


#Takes in a dataset, and returns a Nx(M+1) array of the data, with rows
# of the form (t,x,x,x...)
def getdataset(file):
    data = pd.read_csv(file).to_numpy()
    N = len(data)
    ones = np.repeat(1,N)
    return np.column_stack((ones, data))

#(e)
#takes in a dataset (as returned by getdataset), and converts it into a data-set
# corresponding to a degree-D polynomial regression.
# note: will break if you put D=0
def convertpoly(data, D, axis=1):
    ones = data[:,0]
    col = data[:,axis]
    t = data[:,-1]
    #TODO: make this line prettier
    pdata = np.array([col**i for i in range(1,D+1)]).T
    return np.column_stack((ones, pdata,t))


#gets the big X matrix
def getX(data):
    return data[:,:-1]

#(a.)
#This function takes in a data set (as returned by getdataset) and returns the 
# OLS parameters (w_0,w_1) such that t_n ~ w_0 + w_1*x_n
# regparam is the regularization parameter (lambda) if you are using it
def getOLS(data, regparam=0):
    X = getX(data) #the big X matrix
    t = data[:,-1] #the target data
    N = len(t)
    
    operation = np.dot( np.linalg.inv(np.dot(X.T, X)), X.T)
    return np.dot(operation, t)



#Takes in a data-set (as given by getdataset) and returns the prediction according
# to classifier, which is a 1D array
def getpred(classifier, data):
    return np.dot(getX(data), classifier)

#This takes a (one-dimensional) array and returns the same size array, kind of a
# polynomial version of getpred, but different syntax
def applypoly(poly, x):
    xacc = np.repeat(1.0,len(x)) #the "current value of x^i", which starts as all 1 when i = 0
    ret = np.repeat(0.0,len(x)) #accumulates the answer
    for i in range(len(poly)):
        ret += xacc*poly[i]
        xacc *= x
    return ret


#Takes in a data-set (as given by getdataset) and a pair (w_0, w_1), and plots everything
#Note that this only really works if M = 1.
def plotoutput(data, classifier, title):
    X_values = data[:,1]
    Y_true = data[:,-1] #actual data points
    Y_pred = getpred(classifier, data) #predicted output
    plt.title(title)
    plt.scatter(X_values, Y_true, c='b', label='true data')
    plt.plot(X_values, Y_pred, c='g', label='line of best fit')
    plt.legend()
    plt.show()

#Like plotoutput, but accomodates polynomial functions!
def plotoutputpoly(data, classifier, title):
    X_values = data[:,1]
    Y_true = data[:,-1] #actual data points
    
    X_fine = np.linspace(min(X_values), max(X_values), num=20) # for plotting the polynomial
    Y_prediction = applypoly(classifier, X_fine) #predicted output
    
    plt.title(title)
    plt.scatter(X_values, Y_true, c='y', label='true data')
    plt.plot(X_fine,Y_prediction, c='r', label='polynomial of best fit')
    plt.legend()
    plt.show()


#(d.)
#returns the list of squared errors
def geterrors(data, classifier):
    true = data[:,-1]
    pred = getpred(classifier, data)
    ret = (pred-true)*(pred-true)
    return ret
def getOLSerror(data, classifier):
    return np.sum(geterrors(data, classifier))

#(f)
#def getOLSpoly(ordinarydata, D, axis=1):
#    return getOLS(convertpoly(ordinarydata, D, axis))

mydata = getdataset("synthdata2016.csv")
mypolydata = convertpoly(mydata, 3)
myclassifier = getOLS(mypolydata)
plotoutputpoly(mypolydata, myclassifier, "graph of a quadratic fit to womens 100m times")


#myclassifier = getOLS(mydata)
#plotoutput(mydata, myclassifier, "Graph of OLS error for womens top 100 racetimes")
#print("The total error is", getOLSError(mydata, myclassifier))
