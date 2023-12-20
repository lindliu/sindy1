#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:41:15 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func4, func4_
from train_gsindy_2d import data_generator, fit_gsindy_2d, model_selection_gsindy_2d
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
basis_type = 'mix'



if basis_type == 'mix':
    basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
    basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_mix0]), \
                                 np.array([f(1,1) for f in basis_functions_name_mix1])]     ### corresponding names of the basis functions
if basis_type == 'poly':
    basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5]              ### basis functions for each feature
    basis_functions_name_list = [basis_functions_name_poly_5, basis_functions_name_poly_5]    ### corresponding names of the basis functions
basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}



### time
dt = .1 
t = np.arange(0,8,dt)

#################### 2 variable ####################
x0_list = [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
a_list = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5), (1.2,-1.5), (1.3,-1.)]

func = func4_
real0 = "x'=a*x + b*xy"
real1 = "y'=b*y + a*xy"
real_list = [real0, real1]
####################################################

# #################### 1 variable ####################
# x0_list = [[3, 1], [3, 1]]
# a_list = [(.7,), (1,)]#,(.5,),(.6,)]

# func = func4
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"
# real_list = [real0, real1]
# ####################################################



if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/coeff', exist_ok=True)

    for num in [1, len(a_list)]:
        
        if num==1:
            save_path = f'results/gsindy_one_by_one_{basis_type}.txt'
            open(save_path, 'w').close()
            
            max_split = 5
            for num_split in range(2, max_split+1):
                for idx in range(len(a_list)):
                    x0 = [x0_list[idx]]
                    a = [a_list[idx]]
                    
                    t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
                    
                    model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                                fit_gsindy_2d(sol_org_list, num_traj, num_feature, t_, num, real_list, \
                                                              basis, precision, alpha, opt, deriv_spline, \
                                                              ensemble, print_results=False)
                    
                    ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                            x0, t_, a, real_list, basis, model_set, \
                                                            sol_org_list, same0_basis_list, same1_basis_list, parameter_list)
    
                    coef = model_set[best_BIC_model].mean(0)
                    np.save(f'results/coeff/gsindy_one_{basis_type}_{num_split}_{idx}.npy', coef)
                    
                    mask0 = coef[0,:]!=0
                    mask1= coef[1,:]!=0
                    with open(save_path, "a") as file1:
                        file1.writelines(['*'*15, f'result of trajectory {idx} split into {num_split} pieces', '*'*15, '\n'])
                        file1.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
                        file1.write(f'basis of feature 0: {basis_functions_name_list[0][mask0]} \n')
                        file1.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
                        file1.write(f'basis of feature 1: {basis_functions_name_list[1][mask1]} \n\n')
                        
        
        if num>1:
            save_path = f'results/gsindy_all_{basis_type}.txt'
            open(save_path, 'w').close()
            
            x0 = x0_list[:num]
            a = a_list[:num]
            
            t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num)
            
            model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                        fit_gsindy_2d(sol_org_list, num_traj, num_feature, t_, num, real_list, \
                                                      basis, precision, alpha, opt, deriv_spline, \
                                                      ensemble, print_results=False)
            
            ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                    x0, t_, a, real_list, basis, model_set, \
                                                    sol_org_list, same0_basis_list, same1_basis_list, parameter_list)

            coef = model_set[best_BIC_model] ### num_traj, num_feature, num_basis
            np.save(f'results/coeff/gsindy_all_{basis_type}_{num}.npy', coef)

            mask0 = coef[:,0,:]!=0
            mask1= coef[:,1,:]!=0
            
            for idx in range(num_traj):
                with open(save_path, "a") as file2:
                    file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
                    file2.write(f'coef of feature 0: {coef[idx,0,:][mask0[idx]]} \n')
                    file2.write(f'basis of feature 0: {basis_functions_name_list[0][mask0[idx]]} \n')
                    file2.write(f'coef of feature 1: {coef[idx,1,:][mask1[idx]]} \n')
                    file2.write(f'basis of feature 1: {basis_functions_name_list[1][mask1[idx]]} \n\n')