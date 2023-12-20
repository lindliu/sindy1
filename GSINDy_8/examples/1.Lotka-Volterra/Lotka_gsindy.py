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
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=True)
basis_functions_list = basis['functions']
basis_functions_name_list = basis['names']



if __name__ == "__main__":
    
    from train_gsindy_2d import fit_gsindy_2d, model_selection_gsindy_2d, process_gsindy_one_2D
    from utils import data_generator
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/coeff', exist_ok=True)

    for num in [1, len(a_list)]:
        threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        threshold_group_list = [1e-3, 1e-2]
        threshold_similarity_list = [1e-3, 1e-2] if num!=1 else [1e-1]
        
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
                                                fit_gsindy_2d(sol_org_list, num_traj, t_, num, real_list, \
                                                              basis, precision, alpha, opt, deriv_spline, \
                                                              ensemble, 
                                                              threshold_sindy_list = threshold_sindy_list, \
                                                              threshold_group_list = threshold_group_list,\
                                                              threshold_similarity_list = threshold_similarity_list, 
                                                              print_results=False)
                    
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
                      
                        
        process_gsindy_one_2D(func, x0_list, a_list, t, real_list, basis, basis_type, num_traj=len(a_list), num_feature=2)
        
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