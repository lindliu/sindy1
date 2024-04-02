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

import time

ensemble = True

precision = 1e-2
K = 2000
noise_l = 0.1
threshold = 1e-2
step_x, step_t = 1, 1


t1 = time.time()
metric = []
for step_t in [1,2,3,4,5,6,7]:
    # # real = np.array([0,0,0,0,-1,0,0,0])
    # # order = 2
    # real = np.array([0,0,0,0,0,-1,0,0,0,0,0])
    # order = 3
    # IB_1 = np.load('data/IB_1.npz')
    # t = np.ravel(IB_1['t'])[::step_t]
    # x = np.ravel(IB_1['x'])[::step_x]
    # u = np.real(IB_1['usol'])[::step_x,::step_t]
    
    
    # real = np.array([0,0,0,0,0,-.7,0,0,0,0,0])
    # order = 3
    # IB_2 = np.load('data/IB_2.npz')
    # t = np.ravel(IB_2['t'])[::step_t]
    # x = np.ravel(IB_2['x'])[::step_x]
    # u = np.real(IB_2['usol'])[::step_x,::step_t]
    
    
    # real = np.array([0,0,0,0,-1,-1,0,0,0,0,0])
    # order = 3
    # KDV_1 = np.load('data/KDV_1.npz')
    # t = np.ravel(KDV_1['t'])[::step_t]
    # x = np.ravel(KDV_1['x'])[::step_x]
    # u = np.real(KDV_1['usol'])[::step_x,::step_t]
    
    
    # real = np.array([0,0,0,0,-.7,-1.5,0,0,0,0,0])
    # order = 3
    # KDV_2 = np.load('data/KDV_2.npz')
    # t = np.ravel(KDV_2['t'])[::step_t]
    # x = np.ravel(KDV_2['x'])[::step_x]
    # u = np.real(KDV_2['usol'])[::step_x,::step_t]
    
    
    real = np.array([0,0,0,-1,0,-1,-1,0,0,0,0,0,0,0])  ## works
    order = 4
    KS_1 = np.load('data/KS_1.npz')
    t = np.ravel(KS_1['t'])[::step_t]
    x = np.ravel(KS_1['x'])[::step_x]
    u = np.real(KS_1['usol'])[::step_x,::step_t]
    
    
    # real = np.array([0,0,0,.5,0,-5,0,0,0,0,0])
    # order = 3
    # Burger_1 = np.load('data/Burger_1.npz')
    # t = np.ravel(Burger_1['t'])[::step_t]
    # x = np.ravel(Burger_1['x'])[::step_x]
    # u = np.real(Burger_1['usol'])[::step_x,::step_t]
    
    
    mask = real!=0
    coeffs = []
    for ii in range(10):
        
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
        
        
        optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=True)
        # optimizer = ps.SR3(threshold=threshold, thresholder="l0", tol=1e-8, normalize_columns=True, max_iter=1000)
        # np.random.seed(1)
        
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u_noise, ensemble=ensemble)
        print(f'{ii}. WSINDy: ')
        # print(pde_lib.get_feature_names())
        model.print()
        
        # print(np.linalg.norm(model.coefficients()[0,...][mask]-real[mask])/np.linalg.norm(real[mask]))
        ###########################################################################
        
        coef = model.coefficients()[0,...]
        coef[np.abs(coef)<precision] = 0
        coeffs.append(coef)
    
    
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
    
    
    metric.append([coeff_error.mean(), rate_avg, TPR.mean()])
        
        

print(f'using {(time.time()-t1)/60:.1f} mins')

# import pickle
# with open('KDV_1_e.pickle', 'wb') as file:
#     pickle.dump(metric, handle)


# with open("KDV_1_e.pickle", "rb") as file:
#     metric = pickle.load(file)
    