#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:56:05 2023

@author: dliu
"""
import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from GSINDy import *
from utils import ode_solver, get_deriv, get_theta

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()


#%% pysindy
import pysindy_ as ps
from pysindy_.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
from sklearn.linear_model import Lasso

def SINDy_by_pysindy(sol_, sol_deriv_, t_, basis, threshold_sindy, opt, ensemble, alpha=0.05):
    ### pysindy settings
    basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
    
    assert (basis_functions_list[0]==basis_functions_list[1]).all(), 'pysindy does not support different features with different basis functions'
    
    basis_functions = basis_functions_list[0]
    basis_functions_name = basis_functions_name_list[0]
        
    lib_custom = CustomLibrary(library_functions=basis_functions, function_names=basis_functions_name)
    lib_generalized = GeneralizedLibrary([lib_custom])
    
    if opt=='SQTL':
        optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=alpha)
    elif opt=='LASSO':
        optimizer = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
    elif opt=='SR3':
        optimizer = ps.SR3(threshold=threshold_sindy, nu=.1)
    
    model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)

    ### sindy
    model.fit(sol_, t=t_, x_dot=sol_deriv_, ensemble=ensemble, quiet=True)
    # model.print()
    # model.coefficients()
    
    return model

def SINDy_by_coeff(sol_, sol_deriv_, t_, basis, threshold_sindy, alpha):
    basis_functions_list = basis['functions']
    # basis_functions_name_list = basis['names']
    
    Theta0 = get_theta(sol_, basis_functions_list[0])
    Theta1 = get_theta(sol_, basis_functions_list[1])
    DXdt0 = sol_deriv_[:,[0]]
    DXdt1 = sol_deriv_[:,[1]]
    
    num_feature, num_basis = sol_deriv_.shape[1], Theta0.shape[1]
    Xi = np.zeros([num_feature, num_basis])
    
    Xi[0,:] = SLS(Theta0, DXdt0, threshold_sindy, alpha)[...,0]
    Xi[1,:] = SLS(Theta1, DXdt1, threshold_sindy, alpha)[...,0]
    
    return Xi

def fit_sindy_2d(sol_, sol_deriv_, t_, real_list, basis, alpha, opt, deriv_spline, ensemble, \
                 threshold_sindy_list=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1]):
    ### sindy 
    model_set = []
    for threshold_sindy in threshold_sindy_list:
        if opt in ['SQTL', 'LASSO', 'SR3']:
            model = SINDy_by_pysindy(sol_, sol_deriv_, t_, basis, threshold_sindy, opt, ensemble, alpha)
            
        elif opt == 'Manually':
            model = SINDy_by_coeff(sol_, sol_deriv_, t_, basis, threshold_sindy, alpha)
        
        model_set.append(model)
        
    return model_set




from ModelSelection import ModelSelection

def model_selection_pysindy_2d(model_set, traj_i, x0_i, t, precision=None):
    t_steps = t.shape[0]
        
    ms = ModelSelection(model_set, t_steps)
    ms.compute_k_sindy()
    
    for model_id, model in enumerate(model_set):    
        sse_sum = 0
        
        if np.abs(model.coefficients()).sum()>1e3:
            ms.set_model_SSE(model_id, 1e5)
            continue
            
        ##### simulations #####
        simulation = model.simulate(x0_i, t=t, integrator = "odeint")
        
        sse_sum += ms.compute_SSE(traj_i, simulation)
        
        # ms.set_model_SSE(model_id, sse_sum/num_traj)
        ms.set_model_SSE(model_id, sse_sum)
    
    best_AIC_model = ms.compute_AIC()
    best_AICc_model = ms.compute_AICc()
    best_BIC_model = ms.compute_BIC()
    
    ### Get best model
    print("Melhor modelo AIC = " + str(best_AIC_model) + "\n")
    print("Melhor modelo AICc = " + str(best_AICc_model) + "\n")
    print("Melhor modelo BIC = " + str(best_BIC_model) + "\n")
    
    if precision is not None:
        def count_decimal_places(number):
            return len(str(1e-3).split('.')[1])
        ### best model 
        print('*'*65)
        print('*'*16+' The best model of trajectory '+'*'*16)
        print('*'*65)
        model_best = model_set[best_BIC_model]
        model_best.print(precision=count_decimal_places(precision))
        print('*'*65)
        print('*'*65)

    return ms, best_BIC_model




def model_selection_coeff_2d(model_set_, sol_, x0_, t_, a, real_list, basis):
    def func_simulation(x, t, param, basis_functions_list):
        mask0 = param[0]!=0
        mask1 = param[1]!=0
        
        x1, x2 = x
        dx1dt = 0
        for par,f in zip(param[0][mask0], basis_functions_list[0][mask0]):
            dx1dt = dx1dt+par*f(x1,x2)
        
        dx2dt = 0
        for par,f in zip(param[1][mask1], basis_functions_list[1][mask1]):
            dx2dt = dx2dt+par*f(x1,x2)
            
        dxdt = [dx1dt, dx2dt]
        return dxdt
    
    from ModelSelection import ModelSelection
    from scipy.integrate import odeint
    
    basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
    basis_functions_name_list_ = [np.array([f(1,1) for f in basis['names'][0]]), \
                                 np.array([f(1,1) for f in basis['names'][1]])]
    t_steps = t_.shape[0]
    ms = ModelSelection(model_set_, t_steps)
    ms.compute_k_gsindy()
    
    for model_id, Xi in enumerate(model_set_):
        sse_sum = 0
        
        args = (Xi, basis_functions_list)
        ##### simulations #####
        simulation = odeint(func_simulation, x0_, t_, args=args)
        # SSE(sol_org_list[j], simulation)
        
        sse_sum += ms.compute_SSE(sol_, simulation)
            
        ms.set_model_SSE(model_id, sse_sum)
        # ms.set_model_SSE(model_id, sse_sum)
    
    best_AIC_model = ms.compute_AIC()
    best_AICc_model = ms.compute_AICc()
    best_BIC_model = ms.compute_BIC()
    best_HQIC_model = ms.compute_HQIC()
    best_BICc_model = ms.compute_BIC_custom()
    ### Get best model
    print("Melhor modelo AIC = " + str(best_AIC_model) + "\n")
    print("Melhor modelo AICc = " + str(best_AICc_model) + "\n")
    print("Melhor modelo BIC = " + str(best_BIC_model) + "\n")
    print("Melhor modelo HQIC = " + str(best_HQIC_model) + "\n")
    print("Melhor modelo BICc = " + str(best_BICc_model) + "\n")

    ### Xi of best model 
    Xi_best = model_set_[best_BIC_model]
    
    print('*'*25+'real'+'*'*25)
    print(f'real a: {a}')
    print(f'real0: {real_list[0]}')
    print(f'real1: {real_list[1]}')
    print('*'*25+'pred'+'*'*25)

    dx1dt = "x'="
    dx2dt = "y'="
    for j,pa in enumerate(Xi_best[0]):
        if pa!=0:
            dx1dt = dx1dt+f' + {pa:.4f}{basis_functions_name_list_[0][j]}'
    for j,pa in enumerate(Xi_best[1]):
        if pa!=0:
            dx2dt = dx2dt+f' + {pa:.4f}{basis_functions_name_list_[1][j]}'
                
    print(dx1dt)
    print(dx2dt)
    
    return ms, best_BIC_model