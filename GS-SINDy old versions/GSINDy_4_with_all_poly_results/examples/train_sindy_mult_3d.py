#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:45:49 2023

@author: do0236li
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')

import numpy as np
import matplotlib.pyplot as plt
from GSINDy import *

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()


#%% generate data
def data_generator(func, x0, t, a, real_list, num=None, num_split=None):
    from utils import get_multi_sol
    sol_org_list = get_multi_sol(func, x0, t, a)
    
    if num==1:    
        ll = t.shape[0]//num_split
        # idx_init = list(range(0,ll*num_split,ll))
        
        sol_org_list_ = list(sol_org_list[0][:num_split*ll,:].reshape([num_split,ll,-1]))
        t_ = t[:ll]
        x0_ = [list(sub[0]) for sub in sol_org_list_]
        a_ = [a[0] for _ in range(num_split)]

        t, x0, a, sol_org_list = t_, x0_, a_, sol_org_list_

    ### generate data ###
    num_traj = len(a)
    num_feature = len(x0[0])
    
    ### plot data ###
    fig, ax = plt.subplots(1,1,figsize=[6,3])
    for i in range(num_traj):
        ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
    ax.legend()
    ax.text(1, .95, f'${real_list[0]}$', fontsize=12)
    ax.text(1, .8, f'${real_list[1]}$', fontsize=12)
    ax.text(1, .65, f'${real_list[2]}$', fontsize=12)
    return t, x0, a, sol_org_list, num_traj, num_feature

# t, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real0, real1, num, num_split)

#%%
def fit_sindy_mult_3d(func, x0, t, a, num, num_split, real_list, monomial, monomial_name, \
           precision, alpha, opt, deriv_spline, ensemble):
    ### generate data
    t, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
    
    ### pysindy settings
    import pysindy_ as ps
    from pysindy_.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
    from sklearn.linear_model import Lasso
    
    if monomial.__name__ == 'monomial_poly':
        # lib_generalized = PolynomialLibrary(degree=5)
        basis_functions = [lambda x,y: 1, \
                lambda x,y: x, lambda x,y: y, \
                lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
                lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
                lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
                lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5]
        names = [lambda x,y: '1', \
                lambda x,y: 'x', lambda x,y: 'y', \
                lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
                lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
                lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4', \
                lambda x,y: 'x^5', lambda x,y: 'x^4y', lambda x,y: 'x^3y^2', lambda x,y: 'x^2y^3', lambda x,y: 'xy^4', lambda x,y: 'y^5']
    
    if monomial.__name__ == 'monomial_all':
        basis_functions = [lambda x,y: 1, \
                lambda x,y: x, lambda x,y: y, \
                lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
                lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
                lambda x,y: x**4, lambda x,y: y**4, \
                lambda x,y: 1/x, lambda x,y: 1/y, 
                lambda x,y: np.sin(x), lambda x,y: np.sin(y),lambda x,y: np.cos(x), lambda x,y: np.cos(y),\
                lambda x,y: np.exp(x), lambda x,y: np.exp(y)]
        names = [lambda x,y: '1', \
                lambda x,y: 'x', lambda x,y: 'y', \
                lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
                lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
                lambda x,y: 'x^4', lambda x,y: 'y^4', \
                lambda x,y: '1/x', lambda x,y: '1/y', \
                lambda x,y: 'sin(x)', lambda x,y: 'sin(y)',lambda x,y: 'cos(x)', lambda x,y: 'cos(y)',\
                lambda x,y: 'exp(x)', lambda x,y: 'exp(y)']
            
    if monomial.__name__ == 'monomial_lorenz':
        basis_functions = [lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
                lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
                lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
                lambda x,y,z: x**2*y, lambda x,y,z: x*y**2, lambda x,y,z: x*z**2, lambda x,y,z: y**2*z, lambda x,y,z: y*z**2,  \
                lambda x,y,z: x**4,  lambda x,y,z: y**4,  lambda x,y,z: z**4, \
                lambda x,y,z: 1/x, lambda x,y,z: 1/y, lambda x,y,z: 1/z, \
                lambda x,y,z: np.exp(x), lambda x,y,z: np.exp(y), lambda x,y,z: np.exp(z), \
                lambda x,y,z: np.sin(x), lambda x,y,z: np.sin(y), lambda x,y,z: np.sin(z), \
                lambda x,y,z: np.cos(x), lambda x,y,z: np.cos(y), lambda x,y,z: np.cos(z)]
        names = [lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
                lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
                lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
                lambda x,y,z: 'x^2y', lambda x,y,z: 'xy^2', lambda x,y,z: 'xz^2', lambda x,y,z: 'y^2z', lambda x,y,z: 'yz^2',  \
                lambda x,y,z: 'x^4',  lambda x,y,z: 'y^4',  lambda x,y,z: 'z^4', \
                lambda x,y,z: '1/x', lambda x,y,z: '1/y', lambda x,y,z: '1/z', \
                lambda x,y,z: 'exp(x)', lambda x,y,z: 'exp(y)', lambda x,y,z: 'exp(z)', \
                lambda x,y,z: 'sin(x)', lambda x,y,z: 'sin(y)', lambda x,y,z: 'sin(z)', \
                lambda x,y,z: 'cos(x)', lambda x,y,z: 'cos(y)', lambda x,y,z: 'cos(z)']
            
    lib_custom = CustomLibrary(library_functions=basis_functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])

    #%% 
    threshold_sindy_list =  [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    # threshold_sindy_list =  [1e-2]
    from utils import ode_solver, get_deriv
    
    ##################################
    ############# sindy ##############
    ##################################
    model_set = []
    parameter_list = []
    for threshold_sindy in threshold_sindy_list:
        # optimizer = STLSQ(threshold=threshold_sindy, alpha=alpha)
        if opt=='SQTL':
            optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=alpha)
        elif opt=='LASSO':
            optimizer = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
        elif opt=='SR3':
            optimizer = ps.SR3(threshold=threshold_sindy, nu=.1)
            
        model = ps.SINDy(feature_names=["x", "y", "z"], feature_library=lib_generalized, optimizer=optimizer)
        
        print(f'################### [SINDy] threshold: {threshold_sindy} ################')

        sol_list = []
        sol_deriv_list = []
        for traj_i in range(num_traj):
            sol_, t_ = ode_solver(func, x0[traj_i], t, a[traj_i])
            _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
            
            sol_list.append(sol_)
            sol_deriv_list.append(sol_deriv_)
            
        if ensemble:
            model.fit(sol_list, t=t_, x_dot=sol_deriv_list, ensemble=True, quiet=True, multiple_trajectories=True)
        else:
            model.fit(sol_list, t=t_, x_dot=sol_deriv_list, multiple_trajectories=True)
        model.print()
        # model.coefficients()
        
        model_set.append(model)
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol_deriv_, threshold_sindy, precision))
        
        parameter_list.append(threshold_sindy)


    ###########################
    ##### model selection #####
    ###########################
    
    from ModelSelection import ModelSelection
    
    t_steps = t.shape[0]
    ms = ModelSelection(model_set, t_steps)
    ms.compute_k_sindy()
    
    for model_id, model in enumerate(model_set):    
        if np.abs(model.coefficients()).sum()>1e3:
            ms.set_model_SSE(model_id, 1e5)
            continue
        
        sse_sum = 0
        ##### simulations #####
        for traj_i in range(num_traj):
            simulation = model.simulate(x0[traj_i], t=t, integrator="odeint")
        
            sse_sum += ms.compute_SSE(sol_list[traj_i], simulation)
        
        ms.set_model_SSE(model_id, sse_sum/num_traj)
        # ms.set_model_SSE(model_id, sse_sum)
    
    best_AIC_model = ms.compute_AIC()
    best_AICc_model = ms.compute_AICc()
    best_BIC_model = ms.compute_BIC()
    
    ### Get best model
    print("Melhor modelo AIC = " + str(best_AIC_model) + "\n")
    print("Melhor modelo AICc = " + str(best_AICc_model) + "\n")
    print("Melhor modelo BIC = " + str(best_BIC_model) + "\n")
    
    def count_decimal_places(number):
        return len(str(1e-3).split('.')[1])
    
    ### best model 
    print('*'*65)
    print('*'*16+' The best model '+'*'*16)
    print('*'*65)
    model_best = model_set[best_BIC_model]
    model_best.print(precision=count_decimal_places(precision))
    print('*'*65)
    print('*'*65)
    
    print(f'threshold: {parameter_list[best_AIC_model]}')