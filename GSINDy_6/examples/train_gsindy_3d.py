#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:07:31 2023

@author: do0236li
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, func9, \
                monomial_poly, monomial_trig, monomial_lorenz, monomial_lorenz_name, \
                monomial_all, monomial_all_name
from GSINDy import *

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

#%% 
########################################
############# group sindy ##############
########################################
def fit_gsindy_3d(sol_org_list, num_traj, num_feature, t, num, real_list, \
                  monomial, monomial_name, precision, alpha, opt, deriv_spline, ensemble, print_results=True):
    
    model_set = []
    threshold_sindy_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    threshold_group_list = [1e-3, 1e-2]
    threshold_similarity_list = [1e-3, 1e-2] if num!=1 else [1e-1]
    # threshold_sindy_list = [5e-3, 1e-2]
    # threshold_group_list = [1e-4, 1e-3]
    # threshold_similarity_list = [1e-3, 1e-2]
    parameter_list = []
    diff0_basis_list, diff1_basis_list, diff2_basis_list = [], [], []
    same0_basis_list, same1_basis_list, same2_basis_list = [], [], []
    for threshold_sindy in threshold_sindy_list:
        for threshold_group in threshold_group_list:
            for threshold_similarity in threshold_similarity_list:

                gsindy = GSINDy(monomial=monomial,
                                monomial_name=monomial_name, 
                                num_traj = num_traj, 
                                num_feature = num_feature, 
                                threshold_sindy = threshold_sindy, 
                                threshold_group=threshold_group, 
                                threshold_similarity=threshold_similarity, 
                                precision = precision, 
                                alpha=alpha,
                                deriv_spline=deriv_spline,
                                max_iter = 20, 
                                optimizer=opt, ##['Manually', 'SQTL', 'LASSO', 'SR3']
                                ensemble=ensemble)    
                    
                gsindy.get_multi_sub_series_3D(sol_org_list, t, num_series=100, window_per=.7) ### to get theta_list, sol_deriv_list
                gsindy.basis_identification(remove_per=.2, plot_dist=False) ##True
                
                if num==1:
                    split_basis = [True]
                else:
                    split_basis = [True, False]
                    
                for split_basis in [True, False]:
                    # Xi_final = gsindy.prediction(sol_org_list, t)
                    Xi_final = gsindy.prediction_(sol_org_list, t, split_basis=split_basis)
                    
                    model_set.append(Xi_final)
                    
                    all_basis = gsindy.all_basis
                    diff_basis = gsindy.diff_basis
                    same_basis = gsindy.same_basis

                    diff0_basis_list.append(monomial_name[diff_basis[0]])
                    diff1_basis_list.append(monomial_name[diff_basis[1]])
                    diff2_basis_list.append(monomial_name[diff_basis[2]])
                    same0_basis_list.append(monomial_name[same_basis[0]])
                    same1_basis_list.append(monomial_name[same_basis[1]])
                    same2_basis_list.append(monomial_name[same_basis[2]])
                    parameter_list.append([threshold_sindy, threshold_group, threshold_similarity])
                    
                    if print_results:
                        print(f'################### [GSINDy] threshold_sindy: {threshold_sindy} ################')
                        print(f'################### [GSINDy] threshold_group: {threshold_group} ################')
                        print(f'################### [GSINDy] threshold_similarity: {threshold_similarity} ################')
    
                        print('*'*50)
                        print(f'real0: {real_list[0]}')
                        print(f'feature 0 with different basis {monomial_name[diff_basis[0]]}: \n {Xi_final[:,0,all_basis[0]]} \n {monomial_name[all_basis[0]]}')
                        print(f'real1: {real_list[1]}')
                        print(f'feature 1 with different basis {monomial_name[diff_basis[1]]}: \n {Xi_final[:,1,all_basis[1]]} \n {monomial_name[all_basis[1]]}')
                        print(f'real2: {real_list[2]}')
                        print(f'feature 2 with different basis {monomial_name[diff_basis[2]]}: \n {Xi_final[:,2,all_basis[2]]} \n {monomial_name[all_basis[2]]}')

    return model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, same2_basis_list, parameter_list

#%% 
###########################
##### model selection #####
###########################
def model_selection_gsindy_3d(x0, t, a, real_list, monomial, monomial_name, model_set,\
                              sol_org_list, same0_basis_list, same1_basis_list, same2_basis_list, parameter_list):
    
    from ModelSelection import ModelSelection
    from scipy.integrate import odeint
    num_traj = len(sol_org_list)

    if monomial.__name__ == 'monomial_poly':
        basis_functions = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
            lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5])
    if monomial.__name__ == 'monomial_all':
        basis_functions = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: y**4, \
            lambda x,y: 1/x, lambda x,y: 1/y, 
            lambda x,y: np.sin(x), lambda x,y: np.sin(y),lambda x,y: np.cos(x), lambda x,y: np.cos(y),\
            lambda x,y: np.exp(x), lambda x,y: np.exp(y)])
    
    if monomial.__name__ == 'monomial_lorenz':
        basis_functions = np.array([lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
            lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
            lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
            lambda x,y,z: x**2*y, lambda x,y,z: x*y**2, lambda x,y,z: x*z**2, lambda x,y,z: y**2*z, lambda x,y,z: y*z**2,  \
            lambda x,y,z: x**4,  lambda x,y,z: y**4,  lambda x,y,z: z**4, \
            lambda x,y,z: 1/x, lambda x,y,z: 1/y, lambda x,y,z: 1/z, \
            lambda x,y,z: np.exp(x), lambda x,y,z: np.exp(y), lambda x,y,z: np.exp(z), \
            lambda x,y,z: np.sin(x), lambda x,y,z: np.sin(y), lambda x,y,z: np.sin(z), \
            lambda x,y,z: np.cos(x), lambda x,y,z: np.cos(y), lambda x,y,z: np.cos(z)])
        
        
    def func_simulation(x, t, param, basis):
        mask0 = param[0]!=0
        mask1 = param[1]!=0
        mask2 = param[2]!=0

        x1, x2, x3 = x
        dx1dt = 0
        for par,f in zip(param[0][mask0], basis[mask0]):
            dx1dt = dx1dt+par*f(x1,x2,x3)
        
        dx2dt = 0
        for par,f in zip(param[1][mask1], basis[mask1]):
            dx2dt = dx2dt+par*f(x1,x2,x3)
            
        dx3dt = 0
        for par,f in zip(param[2][mask2], basis[mask2]):
            dx3dt = dx3dt+par*f(x1,x2,x3)
            
        dxdt = [dx1dt, dx2dt, dx3dt]
        return dxdt
    
    t_steps = t.shape[0]
    ms = ModelSelection(model_set, t_steps)
    ms.compute_k_gsindy()
    
    for model_id, Xi in enumerate(model_set):
        sse_sum = 0
        for j in range(num_traj):
            # _, sol_deriv_, _ = get_deriv(sol_org_list[j], t, deriv_spline)
            # sse_sum += ms.compute_SSE(monomial(sol_org_list[j])@Xi[j].T, sol_deriv_)
            
            if np.abs(Xi[j]).sum()>1e3:
                sse_sum += 1e5
                continue
            
            args = (Xi[j], basis_functions)
            ##### simulations #####
            simulation = odeint(func_simulation, x0[j], t, args=args)
            # SSE(sol_org_list[j], simulation)
            
            sse_sum += ms.compute_SSE(sol_org_list[j], simulation)
            
        ms.set_model_SSE(model_id, sse_sum/num_traj)
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
    Xi_best = model_set[best_BIC_model]
    
    print('*'*25+'real'+'*'*25)
    print(f'real a: {a}')
    print(f'real0: {real_list[0]}')
    print(f'real1: {real_list[1]}')
    print(f'real2: {real_list[2]}')
    print('*'*25+'pred'+'*'*25)
    for i in range(num_traj):
        print('*'*8+f'traj {i+1}'+'*'*8)
        dx1dt = "x'="
        dx2dt = "y'="
        dx3dt = "z'="
        for j,pa in enumerate(Xi_best[i,0]):
            if pa!=0:
                dx1dt = dx1dt+f' + {pa:.4f}{monomial_name[j]}'
        for j,pa in enumerate(Xi_best[i,1]):
            if pa!=0:
                dx2dt = dx2dt+f' + {pa:.4f}{monomial_name[j]}'
        for j,pa in enumerate(Xi_best[i,2]):
            if pa!=0:
                dx3dt = dx3dt+f' + {pa:.4f}{monomial_name[j]}'
                    
        print(dx1dt)
        print(dx2dt)
        print(dx3dt)

    
    print('*'*50)
    print(f'threshold: {parameter_list[best_BIC_model]}')
    print(f'same basis for feature 0: {same0_basis_list[best_BIC_model]}')
    print(f'same basis for feature 1: {same1_basis_list[best_BIC_model]}')
    print(f'same basis for feature 2: {same2_basis_list[best_BIC_model]}')

    return ms, best_BIC_model, parameter_list