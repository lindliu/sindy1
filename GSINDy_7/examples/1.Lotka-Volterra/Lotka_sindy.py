#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:57:56 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func4, func4_
from train_sindy_2d import fit_sindy_2d, model_selection_pysindy_2d, model_selection_coeff_2d
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
basis_type = 'poly'


if basis_type == 'mix':
    basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
    basis_functions_name_list = [basis_functions_name_mix0, basis_functions_name_mix1]    ### corresponding names of the basis functions
        
if basis_type == 'poly':
    basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5]              ### basis functions for each feature
    basis_functions_name_list = [basis_functions_name_poly_5, basis_functions_name_poly_5]    ### corresponding names of the basis functions

basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}


dt = .1
t = np.arange(0,8,dt)

#################### 2 variable ####################
x0 = [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
a = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5), (1.2,-1.5), (1.3,-1.)]

func = func4_
real0 = "x'=a*x + b*xy"
real1 = "y'=b*y + a*xy" 
real_list = [real0, real1]
####################################################

# #################### 1 variable ####################
# x0 = [[3, 1], [3, 1]]
# a = [(.7,), (1,)]#,(.5,),(.6,)]

# func = func4
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"
# real_list = [real0, real1]
# ####################################################


if __name__ == "__main__":
    
    
    from utils import ode_solver, get_deriv, data_generator
    # t, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list)
    
    threshold_sindy_list =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    model_best_list = []
    for idx in range(len(a)):
        ### get trajectory
        sol_, t_ = ode_solver(func, x0[idx], t, a[idx])
        _, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)
        
        if opt in ['SQTL', 'LASSO', 'SR3']:
            ### sindy
            model_set = fit_sindy_2d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)

            ### model selection
            model_best, best_BIC_model = model_selection_pysindy_2d(model_set, sol_, x0[idx], t_, precision)
            model_best = model_best.coefficients()
            
        elif opt == 'Manually':
            ### sindy
            model_set = fit_sindy_2d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)
            ### model selection
            ms, best_BIC_model = model_selection_coeff_2d(model_set, sol_, x0[idx], t_, a[idx], real_list, basis)
            model_best = model_set[best_BIC_model]
        
        model_best_list.append(model_best)







    basis_functions_name_list_ = [np.array([f(1,1) for f in basis['names'][0]]), \
                                 np.array([f(1,1) for f in basis['names'][1]])]
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/coeff', exist_ok=True)
    save_path = f'results/sindy_all_{basis_type}.txt'
    open(save_path, 'w').close()
    
    for idx in range(len(model_best_list)):
        # coef = model_best_list[idx].coefficients()
        coef = model_best_list[idx]
        np.save(f'results/coeff/sindy_{basis_type}_{idx}.npy', coef)

        mask0 = np.abs(coef[0]) > precision
        mask1 = np.abs(coef[1]) > precision
        with open(save_path, "a") as file2:
            file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
            file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
            file2.write(f'basis of feature 0: {basis_functions_name_list_[0][mask0]} \n')
            file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
            file2.write(f'basis of feature 1: {basis_functions_name_list_[1][mask1]} \n\n')
