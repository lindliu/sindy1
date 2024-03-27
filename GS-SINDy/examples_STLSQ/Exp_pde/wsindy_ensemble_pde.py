#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:02:36 2024

@author: do0236li
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../tools')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

import pysindy_ as ps

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
# np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


ensemble = True

K = 5000
noise_l = .9
threshold = 5e-2
step = 4



# # real = np.array([0,0,0,0,-1,0,0,0])
# # order = 2
# real = np.array([0,0,0,0,0,-1,0,0,0,0,0])
# order = 3
# IB_1 = np.load('data/IB_1.npz')
# t = np.ravel(IB_1['t'])[::step]
# x = np.ravel(IB_1['x'])[::step]
# u = np.real(IB_1['usol'])[::step,::step]


# real = np.array([0,0,0,0,-1,-1,0,0,0,0,0])
# order = 3
# KDV_1 = np.load('data/KDV_1.npz')
# t = np.ravel(KDV_1['t'])[::step]
# x = np.ravel(KDV_1['x'])[::step]
# u = np.real(KDV_1['usol'])[::step,::step]


real = np.array([0,0,0,-1,0,-1,-1,0,0,0,0,0,0,0])
order = 4
KS_1 = np.load('data/KS_1.npz')
t = np.ravel(KS_1['t'])[::step]
x = np.ravel(KS_1['x'])[::step]
u = np.real(KS_1['usol'])[::step,::step]





mask = real!=0
coeffs = []
for ii in range(20):
    u_noise = u + noise_l*np.mean(u**2)**.5 * np.random.normal(0,1,u.shape)
    u_noise = u_noise[:,:,np.newaxis]


    ############################# weak form SINDy ##############################
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T
    
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        function_names = library_function_names,
        # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
        derivative_order=order,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=K,
        include_bias=False
    )
    
    
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=False)
    # optimizer = ps.SR3(threshold=threshold, thresholder="l0", tol=1e-8, normalize_columns=True, max_iter=1000)
    # np.random.seed(1)
    
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_noise, ensemble=ensemble)
    print(f'{ii}. WSINDy: ')
    # print(pde_lib.get_feature_names())
    model.print()
    
    # print(np.linalg.norm(model.coefficients()[0,...][mask]-real[mask])/np.linalg.norm(real[mask]))
    ###########################################################################
    
    coeffs.append(model.coefficients()[0,...])


    ################### print results ######################
    coeff_error = np.linalg.norm(np.c_[coeffs][:,mask]-real[mask], axis=1)/np.linalg.norm(real[mask])
    print(f"{ii} average coeff error: {coeff_error.mean():.4f}")
    
    zz = (np.c_[coeffs]==0)[:,real==0].sum(1)  ## identified zero term
    non_zz = (np.c_[coeffs]!=0)[:,real!=0].sum(1)  ## identified non-zero term
    rate_avg = ((zz+non_zz)/real.shape[0]).mean()
    print(f"{ii} average success rate: {rate_avg:.4f}")
    
    TP = (np.c_[coeffs]!=0)[:,real!=0].sum(1) ## correctly identified nonzero coefficients
    FN = (np.c_[coeffs]==0)[:,real!=0].sum(1) ## coefficients falsely identified as zero
    FP = (np.c_[coeffs]!=0)[:,real==0].sum(1) ## coefficients falsely identified as nonzero
    TPR = TP/(TP+FN+FP)
    print(f"{ii} average TPR: {TPR.mean():.4f}")

    
    
    
