#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:25:29 2023

@author: dliu
"""

import h5py
import numpy as np
from numpy.linalg import norm    
from sklearn.linear_model import ridge_regression, SGDRegressor
import matplotlib.pyplot as plt
from glob import glob
from sklearn.linear_model import ridge_regression, LinearRegression

def SLS(Theta, DXdt, threshold):
    n_feature = DXdt.shape[1]
    Xi = ridge_regression(Theta,DXdt, alpha=0.05).T
    # Xi = np.linalg.lstsq(Theta,DXdt, rcond=None)[0]
    Xi[np.abs(Xi)<threshold] = 0
    # print(Xi)
    for _ in range(10):
        smallinds = np.abs(Xi)<threshold
        Xi[smallinds] = 0
        for ind in range(n_feature):
            biginds = ~smallinds[:,ind]
            Xi[biginds,ind] = ridge_regression(Theta[:,biginds], DXdt[:,ind], alpha=.05).T
            # Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],DXdt[:,ind], rcond=None)[0]
    # print(Xi)
    # np.mean((Theta@Xi-DXdt)**2)
    
    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-10
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T
    Xi[np.abs(Xi)<threshold] = 0

    # # sgd = LinearRegression(fit_intercept=False)
    # # biginds = np.abs(Xi)>1e-10
    # # Xi[biginds[:,0],0] = sgd.fit(Theta[:,biginds[:,0]],DXdt[:,0]).coef_
    # # Xi[biginds[:,1],1] = sgd.fit(Theta[:,biginds[:,1]],DXdt[:,1]).coef_
    # # Xi[np.abs(Xi)<threshold] = 0
    # # # print(Xi)
    return Xi


def Euler_method(init, t, func_drift, func_diffusion):
    dt = t[1:]-t[:-1]
    cur = init 
    res = []
    res.append(cur)
    for i in range(dt.shape[0]):
        cur = cur + func_drift(cur)*dt[i] + func_diffusion(cur)*dt[i]**.5*np.random.randn()
        res.append(cur)
    return np.array(res)

def func_drift(variable, parameter=None):
    X = variable
    dX = -X**3 + 1/2*X
    return dX

def func_diffusion(variable, parameter=None):
    X = variable
    return 1 + 1/4*X**2

def numerical_derivative(X, dt, order=1):
    assert order==1 or order==2, 'wrong order'
    
    if order==1:
        drift_num = (X[1:]-X[:-1])/dt
        diffusion_num = (X[1:]-X[:-1])**2/(2*dt)
        
        return drift_num, diffusion_num, X[:-1]
    
    if order==2:
        drift_num = (-3*X[:-2]+4*X[1:-1]-X[2:])/(2*dt)
        
        q1 = 4*(X[1:-1]-X[:-2])**2
        q2 = (X[2:]-X[:-2])**2
        diffusion_num = (q1 - q2)/(4*dt)
    
        return drift_num, diffusion_num, X[:-2]


# import pysindy as ps
# model = ps.SINDy(feature_names=["x"])
# model.fit(X, t=dt)
# model.print()

dt_ = 2e-4
t_ = np.arange(0,20000,dt_)
for i in range(1000):
    init = np.random.randn()
    X_ = Euler_method(init, t_, func_drift, func_diffusion)
    
    np.save(f'./data_/dw/dw_{i}', X_)
    # hf = h5py.File(f'./data_/dw/dw_{i}', 'w')
    # hf.create_dataset('dataset', data=X_,compression='gzip', compression_opts=9)
    print(i)


xi_drift_list, xi_diffusion_list = [], []
for i, path in enumerate(glob('./data_/dw/*.npy')):
    X_ = np.load(path)
    
    ### sampling
    step = 10
    dt = dt_*step
    X = X_[::step]
    
    ############# first order ###################
    drift_num, diffusion_num, X__ = numerical_derivative(X, dt, order=1)
    # ############# second order ##################
    # drift_num, diffusion_num, X__ = numerical_derivative(X, dt, order=2)
    
    ## create monomials
    theta = np.c_[np.ones_like(X__), X__, X__**2, X__**3, X__**4, X__**5, X__**6, \
                  X__**7, X__**8, X__**9, X__**10, X__**11, X__**12, X__**13, X__**14]
    
    ### LSQ 
    xi_drift = SLS(theta, drift_num.reshape(-1,1), threshold=0.005)
    xi_diffusion = SLS(theta, diffusion_num.reshape(-1,1), threshold=0.001)

    xi_drift_list.append(xi_drift)
    xi_diffusion_list.append(xi_diffusion)
    print(i)


xi_drift_real = np.array([0,1/2,0,-1,0,0,0,0,0,0,0,0,0,0,0])
xi_diffusion_real = np.array([1,0,1/4,0,0,0,0,0,0,0,0,0,0,0,0])

xi_drift_list = np.array(xi_drift_list)[:,:,0]
xi_diffusion_list = np.array(xi_diffusion_list)[:,:,0]
print(f"Mean drift error(lstsq): {norm(xi_drift_list.mean(0)-xi_drift_real)/norm(xi_drift_real):.4f}")
print(f"Mean diffusion error(lstsq): {norm(xi_diffusion_list.mean(0)-xi_diffusion_real)/norm(xi_diffusion_real):.4f}")

# print(f"Mean drift error(ridge): {norm(np.array(xi_drift_ridge_list).mean(0)-xi_drift_real)/norm(xi_drift_real):.4f}")
# print(f"Mean diffusion error(ridge): {norm(np.array(xi_diffusion_ridge_list).mean(0)-xi_diffusion_real)/norm(xi_diffusion_real):.4f}")
