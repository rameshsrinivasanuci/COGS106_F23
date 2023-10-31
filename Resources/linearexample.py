#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:53:30 2020

@author: ramesh
"""
#%%
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt 
#%% Example 1 X deterministic 
np.random.seed(1)
x = np.linspace(0, 10, 10);
beta0 = 5
beta1 = 2
sdepsilon = 5
epsilon = np.random.normal(0,sdepsilon,10)
ylinear = beta0+beta1*x
y = beta0+beta1*x+ epsilon
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
rsquared = r_value**2
yfit = intercept+slope*x
plt.figure()
plt.plot(x,y,'bo')
plt.plot(x,ylinear,'k-')
plt.plot(x,yfit,'rs-')
plt.xlabel('x')
plt.ylabel('y')
#%% Example 2 X random variable 
np.random.seed(2)
mux = 5
sdx = 2
x = np.random.normal(mux,sdx,10)
beta0 = 5
beta1 = 2
sdepsilon = 5
epsilon = np.random.normal(0,sdepsilon,10)
ylinear = beta0+beta1*x
y = beta0+beta1*x+ epsilon
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
rsquared = r_value**2
yfit = intercept+slope*x
plt.figure()
plt.plot(x,y,'bo')
plt.plot(x,ylinear,'k-')
plt.plot(x,yfit,'rs-')
plt.xlabel('x')
plt.ylabel('y')
#%%Error plots 
#Here I am making a simplified case, where the mean of x is zero, and 
#since the error has a mean of zero, the mean of y is zero.  Then, beta0 is 
#zero as well and the only unknown parameter is beta1. 
np.random.seed(505)
mux = 0
sdx = 5
x = np.random.normal(mux,sdx,100)
beta0 = 0
beta1 = 2
sdepsilon = 500
epsilon = np.random.normal(0,sdepsilon,100)
ylinear = beta0+beta1*x
y = beta0+beta1*x+ epsilon
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
rsquared = r_value**2
yfit = intercept+slope*x
plt.figure()
plt.plot(x,y,'bo')
plt.plot(x,ylinear,'k-')
plt.plot(x,yfit,'rs-')
plt.xlabel('x')
plt.ylabel('y')
betatest = np.linspace(0.1, 4,40)
error = np.zeros(40)
for j in range(len(betatest)):
    yprime = beta0 + betatest[j]*x
    error[j] = np.mean(np.square(y-yprime))
    
plt.figure()
plt.plot(betatest,error,'ro-')
plt.xlabel('Beta')
plt.ylabel('Sum of Squared Error')
