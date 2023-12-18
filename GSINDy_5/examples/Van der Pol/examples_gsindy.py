#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:43:31 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, func8,\
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

dt = .05 
t = np.arange(0,20,dt)
num = 6

# ################## 1 variable ####################
x0_list = [[1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5]]
a_list = [(.5,), (.3,), (.4,),(.2,), (.35,), (.6,)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]

func = func5
real0 = "x'=5*(x - y- a*x^3)"
real1 = "y'=.2*x"    
####################################################


if __name__ == "__main__":
    #%% generate data
    from utils import get_multi_sol
    sol_org_list = get_multi_sol(func, x0, t, a)
    
    if num==1:    
        ll = t.shape[0]//num_split
        idx_init = list(range(0,ll*num_split,ll))
        
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
    ax.text(1, .95, f'${real0}$', fontsize=12)
    ax.text(1, .8, f'${real1}$', fontsize=12)

    #%% 
    ########################################
    ############# group sindy ##############
    ########################################
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
                print(f'################### [GSINDy] threshold_sindy: {threshold_sindy} ################')
                print(f'################### [GSINDy] threshold_group: {threshold_group} ################')
                print(f'################### [GSINDy] threshold_similarity: {threshold_similarity} ################')

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
                    
                gsindy.get_multi_sub_series(sol_org_list, t, num_series=100, window_per=.7) ### to get theta_list, sol_deriv_list
                gsindy.basis_identification(remove_per=.2, plot_dist=False) ##True
                
                for split_basis in [True, False]:
                    # Xi_final = gsindy.prediction(sol_org_list, t)
                    Xi_final = gsindy.prediction_(sol_org_list, t, split_basis=split_basis)
                    
                    model_set.append(Xi_final)
                    
                    all_basis = gsindy.all_basis
                    diff_basis = gsindy.diff_basis
                    same_basis = gsindy.same_basis
                    np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})
                    print('*'*50)
                    print(f'real0: {real0}')
                    print(f'feature 0 with different basis {monomial_name[diff_basis[0]]}: \n {Xi_final[:,0,all_basis[0]]} \n {monomial_name[all_basis[0]]}')
                    print(f'real1: {real1}')
                    print(f'feature 1 with different basis {monomial_name[diff_basis[1]]}: \n {Xi_final[:,1,all_basis[1]]} \n {monomial_name[all_basis[1]]}')
                    
                    diff0_basis_list.append(monomial_name[diff_basis[0]])
                    diff1_basis_list.append(monomial_name[diff_basis[1]])
                    same0_basis_list.append(monomial_name[same_basis[0]])
                    same1_basis_list.append(monomial_name[same_basis[1]])
                    parameter_list.append([threshold_sindy, threshold_group, threshold_similarity])
                

    #%% 
    ###########################
    ##### model selection #####
    ###########################
    
    from ModelSelection import ModelSelection
    from scipy.integrate import odeint
    
    if monomial.__name__ == 'monomial_poly':
        basis_functions = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
            lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5])
    elif monomial.__name__ == 'monomial_all':
        basis_functions = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: y**4, \
            lambda x,y: 1/x, lambda x,y: 1/y, 
            lambda x,y: np.sin(x), lambda x,y: np.sin(y),lambda x,y: np.cos(x), lambda x,y: np.cos(y),\
            lambda x,y: np.exp(x), lambda x,y: np.exp(y)])
    
    def func_simulation(x, t, param, basis):
        mask0 = param[0]!=0
        mask1 = param[1]!=0
        
        basis[mask0]
        basis[mask1]
        
        x1, x2 = x
        dx1dt = 0
        for par,f in zip(param[0][mask0], basis[mask0]):
            dx1dt = dx1dt+par*f(x1,x2)
        
        dx2dt = 0
        for par,f in zip(param[1][mask1], basis[mask1]):
            dx2dt = dx2dt+par*f(x1,x2)
            
        dxdt = [dx1dt, dx2dt]
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
    print(f'real0: {real0}')
    print(f'real1: {real1}')
    print('*'*25+'pred'+'*'*25)
    for i in range(num_traj):
        print('*'*8+f'traj {i+1}'+'*'*8)
        dx1dt = "x'="
        dx2dt = "y'="
        for j,pa in enumerate(Xi_best[i,0]):
            if pa!=0:
                dx1dt = dx1dt+f' + {pa:.2f}{monomial_name[j]}'
        for j,pa in enumerate(Xi_best[i,1]):
            if pa!=0:
                dx2dt = dx2dt+f' + {pa:.2f}{monomial_name[j]}'
                    
        print(dx1dt)
        print(dx2dt)
    
    print('*'*50)
    print(f'threshold: {parameter_list[best_BIC_model]}')
    print(f'same basis for feature 0: {same0_basis_list[best_BIC_model]}')
    print(f'same basis for feature 1: {same1_basis_list[best_BIC_model]}')
    
