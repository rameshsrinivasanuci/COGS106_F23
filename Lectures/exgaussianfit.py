#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:06:15 2020

@author: ramesh
"""
#%%
import numpy as np
from matplotlib import pylab as plt
from scipy.special import erf
from scipy.io import loadmat
from scipy.optimize import minimize
#%%
#Lets define a exGaussian probability density function
def exgauss(params, data):
    '''
    params: is a list or numpy array containing two parameters
    x: are the model inputs
    '''

    mu = params[0] #MEAN OF NORMAL
    sigma = params[1] #STANDARD DEVIATION OF NORMAL
    tau = params[2] #TIME CONSTANT (MEAN) OF EXPONENETIAL DECAY
    z = (data-mu)/sigma -sigma/tau 
    cdf = 0.5*(1+erf(z/np.sqrt(2))) #THIS IS THE CUMULATIVE DENSITY OF NORMAL
    p = (mu/tau)+(sigma**2/(2*tau**2))-(data/tau) 
    pdf = (1/tau)*np.exp(p) #THIS IS PDF OF EXPONENTIAL
    y = pdf*cdf
    return y 
#%%
#%%
#GET SOME DATA
data = loadmat('decisiondata.mat')
subject = data['subject']
condition = data['condition']
correct = data['correct']
rt = data['rt']
rt = np.array(rt)
thesubject = 20
data = rt[(condition == 3) & (subject == thesubject) & (correct == 1)] #select RT for a single subject in one condition
#%%
#Lets define negative log likelihood of ex Gaussian.  Note that  log of products is sum of logs 
def exgausslike(params):
    cprob = exgauss(params,data)
    like = -np.sum(np.log(cprob))
    return like
#%%
paramfit = minimize(exgausslike,[200,200,200],method = 'Nelder-Mead')
#%%
thebins = np.linspace(0,2000,21)
thebins2 = np.linspace(0,2000,21)
nhist,bin_edges = np.histogram(rt[(condition == 3) &(subject == thesubject) & (correct == 1)], bins=thebins)
rhist = nhist/np.sum(nhist)
bincenter = bin_edges[0:20]+bin_edges[1:21]
bincenter = bincenter/2
exgaussfit = exgauss(paramfit['x'],thebins2) 
exgaussfit = exgaussfit/np.sum(exgaussfit) #FACTOR OF 10 because I made 10 times as many bins
plt.figure()
plt.bar(bincenter,rhist,width=50,color='r',label='data')
plt.plot(thebins2,exgaussfit,label='exGauss fit')
plt.legend()
plt.xlabel('Response Time')
plt.ylabel('Fraction of Trials')
