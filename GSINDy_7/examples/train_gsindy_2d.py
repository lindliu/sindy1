#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:39:49 2023

@author: dliu
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
    plot_mult_traj(sol_org_list, t, a, real_list)
    return t, x0, a, sol_org_list, num_traj, num_feature

def plot_mult_traj(sol_org_list, t, a, real_list):
    fig, ax = plt.subplots(1,1,figsize=[6,3])
    for i in range(len(sol_org_list)):
        ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
    ax.legend()
    ax.text(1, .95, f'${real_list[0]}$', fontsize=12)
    ax.text(1, .8, f'${real_list[1]}$', fontsize=12)

#%% 
########################################
############# group sindy ##############
########################################
def fit_gsindy_2d(sol_org_list, num_traj, num_feature, t, num, real_list, \
                  basis, precision, alpha, opt, deriv_spline, ensemble, print_results=True):
    basis_functions_name_list = basis['names']
    
    model_set = []
    threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    threshold_group_list = [1e-3, 1e-2]
    threshold_similarity_list = [1e-3, 1e-2] if num!=1 else [1e-1]
    # threshold_sindy_list = [5e-3, 1e-2]
    # threshold_group_list = [1e-4, 1e-3]
    # threshold_similarity_list = [1e-3, 1e-2]
    parameter_list = []
    diff0_basis_list, diff1_basis_list = [], []
    same0_basis_list, same1_basis_list = [], []
    for threshold_sindy in threshold_sindy_list:
        for threshold_group in threshold_group_list:
            for threshold_similarity in threshold_similarity_list:
                ### initilization
                gsindy = GSINDy(basis = basis, 
                                num_traj = num_traj, 
                                num_feature = num_feature, 
                                threshold_sindy = threshold_sindy, 
                                threshold_group=threshold_group, 
                                threshold_similarity=threshold_similarity, 
                                precision = precision, 
                                alpha=alpha,
                                deriv_spline=deriv_spline,
                                max_iter = 20, 
                                optimizer=opt, 
                                ensemble=ensemble)    
                ### get sub-series for each trajectory
                gsindy.get_multi_sub_series_2D(sol_org_list, t, num_series=100, window_per=.7) ### to get theta_list, sol_deriv_list
                ### basis identification for each trajectory
                gsindy.basis_identification(remove_per=.2, plot_dist=False) ##True
                
                if num==1:
                    split_basis = [True]
                else:
                    split_basis = [True, False]
                    
                for split_basis_ in split_basis:
                    Xi_final = gsindy.prediction(sol_org_list, t, split_basis=split_basis_)
                    
                    model_set.append(Xi_final)
                    
                    all_basis = gsindy.all_basis
                    diff_basis = gsindy.diff_basis
                    same_basis = gsindy.same_basis
                    
                    diff0_basis_list.append(basis_functions_name_list[0][diff_basis[0]])
                    diff1_basis_list.append(basis_functions_name_list[1][diff_basis[1]])
                    same0_basis_list.append(basis_functions_name_list[0][same_basis[0]])
                    same1_basis_list.append(basis_functions_name_list[1][same_basis[1]])
                    parameter_list.append([threshold_sindy, threshold_group, threshold_similarity])
                    
                    if print_results:
                        print(f'################### [GSINDy] threshold_sindy: {threshold_sindy} ################')
                        print(f'################### [GSINDy] threshold_group: {threshold_group} ################')
                        print(f'################### [GSINDy] threshold_similarity: {threshold_similarity} ################')
                        print('*'*50)
                        print(f'real0: {real_list[0]}')
                        print(f'feature 0 with different basis {basis_functions_name_list[0][diff_basis[0]]}: \n {Xi_final[:,0,all_basis[0]]} \n {basis_functions_name_list[0][all_basis[0]]}')
                        print(f'real1: {real_list[1]}')
                        print(f'feature 1 with different basis {basis_functions_name_list[1][diff_basis[1]]}: \n {Xi_final[:,1,all_basis[1]]} \n {basis_functions_name_list[1][all_basis[1]]}')


    return model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list

#%% 
###########################
##### model selection #####
###########################

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

def model_selection_gsindy_2d(x0, t, a, real_list, basis, model_set, sol_org_list, \
                              same0_basis_list=None, same1_basis_list=None, parameter_list=None):
    from ModelSelection import ModelSelection
    from scipy.integrate import odeint
    num_traj = len(sol_org_list)
    
    basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
            

    
    t_steps = t.shape[0]
    ms = ModelSelection(model_set, t_steps)
    ms.compute_k_gsindy()
    
    for model_id, Xi in enumerate(model_set):
        sse_sum = 0
        for j in range(num_traj):
            
            if np.abs(Xi[j]).sum()>1e3:
                sse_sum += 1e5
                continue
            
            args = (Xi[j], basis_functions_list)
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
    print('*'*25+'pred'+'*'*25)
    for i in range(num_traj):
        print('*'*8+f'traj {i+1}'+'*'*8)
        dx1dt = "x'="
        dx2dt = "y'="
        for j,pa in enumerate(Xi_best[i,0]):
            if pa!=0:
                dx1dt = dx1dt+f' + {pa:.4f}{basis_functions_name_list[0][j]}'
        for j,pa in enumerate(Xi_best[i,1]):
            if pa!=0:
                dx2dt = dx2dt+f' + {pa:.4f}{basis_functions_name_list[1][j]}'
                    
        print(dx1dt)
        print(dx2dt)
    
    if parameter_list is not None:
        print('*'*50)
        print(f'threshold: {parameter_list[best_BIC_model]}')
        print(f'same basis for feature 0: {same0_basis_list[best_BIC_model]}')
        print(f'same basis for feature 1: {same1_basis_list[best_BIC_model]}')
        return ms, best_BIC_model, parameter_list
    
    else:
        return ms, best_BIC_model