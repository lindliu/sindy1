#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 08:59:02 2023

@author: do0236li
"""

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
            
            if Xi[:,ind].sum()==0:
                break
        
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


def Euler_method(init, t_, func_drift, func_diffusion):
    dt = t_[1:]-t_[:-1]
    cur = init 
    res = np.zeros([2, dt.shape[0]])
    res[:,[0]] = init
    for i in range(1,dt.shape[0]):
        cur = cur + func_drift(cur)*dt[i] + func_diffusion(cur)@np.random.randn(2,1)*dt[i]**.5
        res[:,[i]] = cur
    return np.array(res)

def func_drift_van(X, parameter=None):
    dXdt = np.zeros_like(X)
    x1, x2 = X
    
    dXdt[0] = x2**2
    dXdt[1] = (1-x1**2)*x2 - x1
    return dXdt

def func_diffusion_van(X, parameter=None):
    sigma = np.zeros([2,2])
    x1, x2 = X
    
    sigma[0,0] = 1/2*(1+.3*x2)
    sigma[1,1] = 1/2*(.5+.2*x1)
    return sigma

def numerical_derivative(X, dt, order=1):
    assert order==1 or order==2, 'wrong order'
    
    def first_order_diffusion(x1, x2, dt):
        return (x1[1:]-x1[:-1])*(x2[1:]-x2[:-1])/(2*dt)
        
    if order==1:
        drift_num = (X[1:]-X[:-1])/dt
        
        diffusion_num_00 = first_order_diffusion(X[:,[0]],X[:,[0]], dt)
        diffusion_num_01 = first_order_diffusion(X[:,[0]],X[:,[1]], dt)
        diffusion_num_10 = first_order_diffusion(X[:,[1]],X[:,[0]], dt)
        diffusion_num_11 = first_order_diffusion(X[:,[1]],X[:,[1]], dt)
        diffusion_num = np.c_[diffusion_num_00,diffusion_num_01,diffusion_num_10,diffusion_num_11]
        return drift_num, diffusion_num, X[:-1]
    
    def second_order_diffusion(x1, x2, dt):
        q1 = 4*(x1[1:-1]-x1[:-2]) * (x2[1:-1]-x2[:-2])
        q2 = (x1[2:]-x1[:-2]) * (x2[2:]-x2[:-2])
        return (q1 - q2)/(4*dt)
        
    if order==2:
        drift_num = (-3*X[:-2]+4*X[1:-1]-X[2:])/(2*dt)
        
        diffusion_num_00 = second_order_diffusion(X[:,[0]],X[:,[0]], dt)
        diffusion_num_01 = second_order_diffusion(X[:,[0]],X[:,[1]], dt)
        diffusion_num_10 = second_order_diffusion(X[:,[1]],X[:,[0]], dt)
        diffusion_num_11 = second_order_diffusion(X[:,[1]],X[:,[1]], dt)
        diffusion_num = np.c_[diffusion_num_00,diffusion_num_01,diffusion_num_10,diffusion_num_11]
    
        return drift_num, diffusion_num, X[:-2]

def monomial(X):
    x1, x2 = X[:,[0]], X[:,[1]]
    theta = np.c_[np.ones_like(x1), \
                  x1, x2, \
                  x1**2, x1*x2, x2**2, \
                  x1**3, x1**2*x2, x1*x2**2, x2**3, \
                  x1**4, x1**3*x2, x1**2*x2**2, x1*x2**3, x2**4, \
                  x1**5, x1**4*x2, x1**3*x2**2, x1**2*x2**3, x1*x2**4, x2**5, \
                  x1**6, x1**5*x2, x1**4*x2**2, x1**3*x2**3, x1**2*x2**4, x1*x2**5, x2**6]
        
    return theta



# import pysindy as ps
# model = ps.SINDy(feature_names=["x"])
# model.fit(X, t=dt)
# model.print()

dt_ = 2e-5
t_ = np.arange(0,1000,dt_)
for i in range(1000):
    init = np.random.randn(2,1)
    X_ = Euler_method(init, t_, func_drift_van, func_diffusion_van).T
    
    np.save(f'./data_/van/van_{i}', X_)
    # hf = h5py.File(f'./data_/dw/dw_{i}', 'w')
    # hf.create_dataset('dataset', data=X_,compression='gzip', compression_opts=9)
    print(i)


xi_drift_list, xi_diffusion_list = [], []
for i, path in enumerate(glob('./data_/van/*.npy')):
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
    theta = monomial(X__)
    
    ### LSQ 
    xi_drift = SLS(theta, drift_num, threshold=0.05)  #5;    
    xi_diffusion = SLS(theta, diffusion_num, threshold=0.02)

    xi_drift_list.append(xi_drift)
    xi_diffusion_list.append(xi_diffusion)
    print(i)


xi_drift_real = np.array([[0, 0,0,0,1,0,0, 0,0,0,0,0,0,0,0],
                          [0,-1,1,0,0,0,0,-1,0,0,0,0,0,0,0]])
xi_diffusion_real = np.array([[.5, 0, .15,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,  0, 0,  0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,  0, 0,  0,0,0,0,0,0,0,0,0,0,0,0],
                              [.25,.1,0,  0,0,0,0,0,0,0,0,0,0,0,0]])

xi_drift_list = np.array(xi_drift_list)[:,:,0]
xi_diffusion_list = np.array(xi_diffusion_list)[:,:,0]
print(f"Mean drift error(lstsq): {norm(xi_drift_list.mean(0)-xi_drift_real)/norm(xi_drift_real):.4f}")
print(f"Mean diffusion error(lstsq): {norm(xi_diffusion_list.mean(0)-xi_diffusion_real)/norm(xi_diffusion_real):.4f}")

# print(f"Mean drift error(ridge): {norm(np.array(xi_drift_ridge_list).mean(0)-xi_drift_real)/norm(xi_drift_real):.4f}")
# print(f"Mean diffusion error(ridge): {norm(np.array(xi_diffusion_ridge_list).mean(0)-xi_diffusion_real)/norm(xi_diffusion_real):.4f}")
