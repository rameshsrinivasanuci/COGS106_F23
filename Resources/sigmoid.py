#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 01:55:35 2020

@author: ramesh
"""

#%%
import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares
#%%
#Lets define a sigmoidal function
def model(params, x):
    '''
    params: is a list or numpy array containing two parameters
    x: are the model inputs
    '''
    p0 = params[0] #SLOPE OF SIGMOID
    p1 = params[1] #MEAN OF SIGMOID
    y = 1/(1+np.exp(-p0*(x-p1)))
    return y 
#%%
#Let's generate some data from the model 
np.random.seed(101)
N = 20
x = np.linspace(0,30,N) 
true_parameters = [0.5,15] #SLOPE AND MEAN
sdnoise = 0.05 
y = model(true_parameters,x) + np.random.normal(0,sdnoise,N)
plt.plot(x,y,'bo')
#%%    
#Let's define a cost function (or loss function)
#The inputs should just be the parameters.  
def rmse(params):
    yhat = model(params, x)
    cost = np.sqrt(np.mean((y-yhat)**2))
    return cost
#%%  Lets make a plot of the cost function 
#FIRST FOR  MEAN ONLY
mumat = np.linspace(1,30,60) #DIFFERENT POSSIBLE VALUES OF MU
error = np.zeros(60)
for j in range(60):
    error[j] = rmse([0.5,mumat[j]]) #COMPUTE ERRORS FOR DIFFERENT VALUES OF MU
plt.figure()
plt.plot(mumat,error,'ko-')
plt.xlabel('mu')
plt.ylabel('error')
#FOR SLOPE ONLY 
slopemat = np.linspace(0.025,1,40) #DIFFERENT VALUES OF SLOPE
error = np.zeros(40)
for j in range(40):
    error[j] = rmse([slopemat[j],15]) #COMPUTE ERRORS FOR DIFFERENT VALUES OF SLOPE
plt.figure()
plt.plot(slopemat,error,'ko-')
plt.xlabel('slope')
plt.ylabel('error')
#FOR BOTH SLOPE AND MEAN 
error2 = np.zeros((40,60))
for j in range(40):
    for k in range(60):
        error2[j,k] = rmse([slopemat[j],mumat[k]]) #COMPUTE ERRORS FOR BOTH
plt.figure()
plt.imshow(error2) 
plt.xlabel('mu')
plt.ylabel('slope')
plt.colorbar()


#%%
initial_guess =  [.1,10]
return_fit = minimize(rmse, initial_guess, method = 'Nelder-Mead') #This is the Nelder Mead (simplex) optimizer seen in class
fitted_parameters = return_fit['x'] #this extracts the fitted parameters
#%% plot it all
plt.plot(x,y, 'bo', label = 'data');
plt.plot(x, model(fitted_parameters, x), label = 'fit');
plt.plot(x, model(true_parameters, x), label = 'true');
plt.legend()
#%%
#Fit model using least squares curve fitting (gradient descent)
return_fit_lsq = least_squares(rmse,initial_guess)
fitted_parameters_lsq = return_fit['x']
#%%