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
from GSINDy import *

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()

#%% 
########################################
############# group sindy ##############
########################################
def fit_gsindy_2d(sol_org_list, num_traj, t, num, real_list, \
                  basis, precision, alpha, opt, deriv_spline, ensemble, 
                  threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1], \
                  threshold_group_list = [1e-3, 1e-2],\
                  threshold_similarity_list = [1e-3, 1e-2], \
                  print_results=True):
    
    num_feature = 2
    basis_functions_name_list = basis['names']
    
    model_set = []
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
    
    
    
import os
from glob import glob
from utils import data_generator

def process_gsindy_one_2D(func, x0_list, a_list, t, real_list, basis, basis_type, num_traj=6, num_feature=2):
    basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
    
    num_basis = len(basis_functions_list[0])
    
    #### results from gsindy one by one
    max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in glob(os.path.join(f'results/coeff/gsindy_one_{basis_type}*.npy'))])
    n_split = max_split-2+1   ##4
    coeff_gsindy_one = np.zeros([n_split, num_traj, num_feature, num_basis])
    for j in range(num_traj):
        for k in range(n_split):
            path_ = glob(os.path.join(f'results/coeff/gsindy_one_{basis_type}_{k+2}_{j}.npy'))[0]
            coeff_gsindy_one[k,j,:,:] = np.load(path_)
            
    save_path = f'results/gsindy_one_by_one_{basis_type}_final.txt'
    open(save_path, 'w').close()
    for i in range(num_traj):
        model_set = coeff_gsindy_one[:,[i],:,:]
        x0 = [x0_list[i]]
        a = [a_list[i]]
        t_, x0, a, sol_, num_traj, num_feature = data_generator(func, x0, t, a, real_list)

        ms, best_BIC_model = model_selection_gsindy_2d(x0, t_, a, real_list, basis, model_set, sol_)
        
        print(f'best trajectory {i} split {best_BIC_model+2}')
        # print(f'{coeff_gsindy_one[best_BIC_model,i,:,:]}')
        
        coef = coeff_gsindy_one[best_BIC_model,i,:,:]
        np.save(f'results/coeff/gsindy_{basis_type}_{i}.npy', coef)
    
    
        mask0 = coef[0,:]!=0
        mask1= coef[1,:]!=0
        with open(save_path, "a") as file2:
            file2.writelines(['*'*15, f'result of trajectory {i} ', '*'*15, '\n'])
            file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
            file2.write(f'basis of feature 0: {basis_functions_name_list[0][mask0]} \n')
            file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
            file2.write(f'basis of feature 1: {basis_functions_name_list[1][mask1]} \n\n')
