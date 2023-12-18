#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:43:45 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, func8, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name, \
                monomial_all, monomial_all_name
from GSINDy import *

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()

opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name

dt = .1    
t = np.arange(0,20,dt)
num = 6

# ################## 1 variable ####################
x0 = [[1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5]]
a = [(.5,), (.3,), (.4,),(.2,), (.35,), (.6,)]

func = func5
real0 = "x'=5*(x - y- a*x^3)"
real1 = "y'=.2*x"    
####################################################



if __name__ == "__main__":
    #%% generate data
    from utils import get_multi_sol
    sol_org_list = get_multi_sol(func, x0, t, a)

    ### generate data ###
    num_traj = len(a)
    num_feature = len(x0[0])

    ### plot data ###
    fig, ax = plt.subplots(1,1,figsize=[6,3])
    for i in range(num_traj):
        ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
    ax.legend()
    ax.text(1, .95, f'${real0}$', fontsize=12)
    ax.text(1, .8, f'${real1}$', fontsize=12)

    #%% pysindy settings
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
        
    lib_custom = CustomLibrary(library_functions=basis_functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])
    
    
    
    #%% 
    threshold_sindy_list =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    # threshold_sindy_list =  [1e-2]
    from utils import ode_solver, get_deriv
    from sklearn.linear_model import Lasso

    for traj_i in range(len(a)):
        
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
            
            model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
        
            # print(f'################### [SINDy] threshold: {threshold_sindy} ################')
            sol_, t_ = ode_solver(func, x0[traj_i], t, a[traj_i])
            _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
            
            if ensemble:
                model.fit(sol_, t=t_, x_dot=sol_deriv_, ensemble=True, quiet=True)
            else:
                model.fit(sol_, t=t_, x_dot=sol_deriv_)
            # model.print()
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
            sse_sum = 0
            
            if np.abs(model.coefficients()).sum()>1e3:
                ms.set_model_SSE(model_id, 1e5)
                continue
                
            ##### simulations #####
            simulation = model.simulate(x0[traj_i], t=t, integrator = "odeint")
            
            sse_sum += ms.compute_SSE(sol_org_list[traj_i], simulation)
            
            # ms.set_model_SSE(model_id, sse_sum/num_traj)
            ms.set_model_SSE(model_id, sse_sum)
        
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
        print('*'*16+f' The best model of trajectory: {traj_i} '+'*'*16)
        print('*'*65)
        model_best = model_set[best_BIC_model]
        model_best.print(precision=count_decimal_places(precision))
        print('*'*65)
        print('*'*65)
        
        print(f'threshold: {parameter_list[best_BIC_model]}')