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
from constants import get_basis_functions
import constants
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


########## hyper parameters ###########
ensemble = constants.ensemble
precision = constants.precision
deriv_spline = constants.deriv_spline
alpha = constants.alpha

########## function variable ###########
t = constants.t
x0_list = constants.x0_list
a_list = constants.a_list

func = constants.func
real_list = constants.real_list

########## basis functions and optimizer ###########
basis_type = constants.basis_type
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=False)
basis_functions_list = basis['functions']
basis_functions_name_list = basis['names']


if __name__ == "__main__":
    
    from train_sindy_2d import fit_sindy_2d, model_selection_pysindy_2d, model_selection_coeff_2d
    from utils import ode_solver, get_deriv
    
    threshold_sindy_list =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    model_best_list = []
    for idx in range(len(a_list)):
        x0 = x0_list[idx]
        a = a_list[idx]
        
        ### get trajectory
        sol_, t_ = ode_solver(func, x0, t, a)
        _, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)
        
        if opt in ['SQTL', 'LASSO', 'SR3']:
            ### sindy
            model_set = fit_sindy_2d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)
            ### model selection
            ms, best_BIC_model = model_selection_pysindy_2d(model_set, sol_, x0, t_, precision)
            model_best = model_set[best_BIC_model].coefficients()
            
        elif opt == 'Manually':
            ### sindy
            model_set = fit_sindy_2d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)
            ### model selection
            ms, best_BIC_model = model_selection_coeff_2d(model_set, sol_, x0, t_, a, real_list, basis)
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
