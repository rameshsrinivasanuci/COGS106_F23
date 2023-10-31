#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 08:45:19 2020

@author: ramesh
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom
from scipy.optimize import minimize
#%%
#binomial likelihood 
#
nsuccess = 6
nfailures = 4
n = nsuccess + nfailures

p = np.linspace(0,1,101);
likelihood = np.zeros(101)
for j in range(len(p)):
    likelihood[j] = binom.pmf(nsuccess,n,p[j]);

plt.figure()
plt.plot(p,likelihood,'k-')
plt.xlabel('Parameter p')
plt.ylabel('Likelihood')
#%%
def likebin(p):
    y = -binom.pmf(nsuccess,n,p)
    return y

paramfit = minimize(likebin,0.5,method = 'Nelder-Mead');
pfit = paramfit['x']
#%%

def likebin(p):
    y = -np.log(binom.pmf(nsuccess,n,p))
    return y

paramfit = minimize(likebin,0.5,method = 'Nelder-Mead');
pfit = paramfit['x']
#%%