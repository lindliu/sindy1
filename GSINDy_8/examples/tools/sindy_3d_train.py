#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:46:18 2023

@author: dliu
"""


from sindy_3d import fit_sindy_3d, model_selection_pysindy_3d, model_selection_coeff_3d
from utils import ode_solver, get_deriv
import os
import numpy as np

def sindy_3d_train(func, t, x0_list, a_list, real_list, basis_type, basis, precision, alpha, opt, deriv_spline, ensemble, path_base='results'):
    threshold_sindy_list =  [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    model_best_list = []
    for idx in range(len(a_list)):
        x0 = x0_list[idx]
        a = a_list[idx]
        
        ### get trajectory
        sol_, t_ = ode_solver(func, x0, t, a)
        _, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)
        
        if opt in ['SQTL', 'LASSO', 'SR3']:
            ### sindy
            model_set = fit_sindy_3d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)
            ### model selection
            ms, best_BIC_model = model_selection_pysindy_3d(model_set, sol_, x0, t_, precision)
            model_best = model_set[best_BIC_model].coefficients()
            
        elif opt == 'Manually':
            ### sindy
            model_set = fit_sindy_3d(sol_, sol_deriv_, t_, real_list, basis, precision, alpha, opt, \
                                     deriv_spline, ensemble, threshold_sindy_list)
            ### model selection
            ms, best_BIC_model = model_selection_coeff_3d(model_set, sol_, x0, t_, a, real_list, basis)
            model_best = model_set[best_BIC_model]
        
        model_best_list.append(model_best)



    
    basis_functions_name_list_ = [np.array([f(1,1,1) for f in basis['names'][0]]), \
                                 np.array([f(1,1,1) for f in basis['names'][1]]), \
                                 np.array([f(1,1,1) for f in basis['names'][2]])]

    os.makedirs(path_base, exist_ok=True)
    os.makedirs(os.path.join(path_base, 'coeff'), exist_ok=True)
    save_path = os.path.join(path_base, f'sindy_all_{basis_type}.txt')
    open(save_path, 'w').close()
    
    for idx in range(len(model_best_list)):
        # coef = model_best_list[idx].coefficients()
        coef = model_best_list[idx]
        np.save(os.path.join(path_base, 'coeff/sindy_{basis_type}_{idx}.npy'), coef)

        mask0 = np.abs(coef[0]) > precision
        mask1 = np.abs(coef[1]) > precision
        mask2 = np.abs(coef[2]) > precision

        with open(save_path, "a") as file2:
            file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
            file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
            file2.write(f'basis of feature 0: {basis_functions_name_list_[0][mask0]} \n')
            file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
            file2.write(f'basis of feature 1: {basis_functions_name_list_[1][mask1]} \n\n')
            file2.write(f'coef of feature 2: {coef[2,:][mask2]} \n')
            file2.write(f'basis of feature 2: {basis_functions_name_list_[2][mask2]} \n\n')
