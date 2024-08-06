#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:21:31 2023

@author: dliu
"""


from gsindy_2d import fit_gsindy_2d, model_selection_gsindy_2d, process_gsindy_one_2D
from utils import data_generator
import os
import numpy as np

def gsindy_2d_train(func, t, x0_list, a_list, real_list, suffix, basis, precision, alpha, opt, deriv_spline, ensemble, path_base='results', \
                    threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1], \
                    threshold_group_list = [1e-3, 1e-2], \
                    threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]):
    
    os.makedirs(path_base, exist_ok=True)
    os.makedirs(os.path.join(path_base, 'coeff'), exist_ok=True)

    # basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
    
    for num in [len(a_list), 1]:
        threshold_similarity_list_ = threshold_similarity_list[0] if num!=1 else threshold_similarity_list[1]
        
        if num==1:
            save_path = os.path.join(path_base, f'gsindy_one_by_one_{suffix}.txt')
            open(save_path, 'w').close()
            
            max_split = 5
            for num_split in range(2, max_split+1):
                for idx in range(len(a_list)):
                    x0 = [x0_list[idx]]
                    a = [a_list[idx]]
                    
                    t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
                    # ## add noise
                    # for i in range(len(sol_org_list)):
                    #     sol_org_list[i] = sol_org_list[i] + np.random.randn(*sol_org_list[i].shape)*sol_org_list[i]*.01

                    model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                                fit_gsindy_2d(sol_org_list, num_traj, t_, num, real_list, \
                                                              basis, alpha, opt, deriv_spline, ensemble, \
                                                              threshold_sindy_list = threshold_sindy_list, \
                                                              threshold_group_list = threshold_group_list,\
                                                              threshold_similarity_list = threshold_similarity_list_, 
                                                              print_results=False)
                    
                    ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                            x0, t_, a, real_list, basis, model_set, \
                                                            sol_org_list, same0_basis_list, same1_basis_list, parameter_list)
    
                    coef = model_set[best_BIC_model].mean(0)
                    np.save(os.path.join(path_base, f'coeff/gsindy_one_{suffix}_{num_split}_{idx}.npy'), coef)
                    
                    
                    mask0 = np.abs(coef[0,:]) > precision
                    mask1 = np.abs(coef[1,:]) > precision
                    with open(save_path, "a") as file1:
                        file1.writelines(['*'*15, f'result of trajectory {idx} split into {num_split} pieces', '*'*15, '\n'])
                        file1.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
                        file1.write(f'basis of feature 0: {basis_functions_name_list[0][mask0]} \n')
                        file1.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
                        file1.write(f'basis of feature 1: {basis_functions_name_list[1][mask1]} \n\n')
                      
                        
            process_gsindy_one_2D(func, x0_list, a_list, t, real_list, basis, suffix, num_traj=len(a_list), num_feature=2)
        
        if num>1:
            save_path = os.path.join(path_base, f'gsindy_all_{suffix}.txt')
            open(save_path, 'w').close()
            
            x0 = x0_list[:num]
            a = a_list[:num]
            
            t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num)
            # ## add noise
            # for i in range(len(sol_org_list)):
            #     sol_org_list[i] = sol_org_list[i] + np.random.randn(*sol_org_list[i].shape)*sol_org_list[i]*.01

            model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                        fit_gsindy_2d(sol_org_list, num_traj, t_, num, real_list, \
                                                      basis, alpha, opt, deriv_spline, ensemble, \
                                                      threshold_sindy_list = threshold_sindy_list, \
                                                      threshold_group_list = threshold_group_list,\
                                                      threshold_similarity_list = threshold_similarity_list_, 
                                                      print_results=False)
            
            ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                    x0, t_, a, real_list, basis, model_set, \
                                                    sol_org_list, same0_basis_list, same1_basis_list, parameter_list)

            coef = model_set[best_BIC_model] ### num_traj, num_feature, num_basis
            np.save(os.path.join(path_base, f'coeff/gsindy_all_{suffix}_{num}.npy'), coef)

            mask0 = np.abs(coef[:,0,:]) > precision
            mask1 = np.abs(coef[:,1,:]) > precision
            for idx in range(num_traj):
                with open(save_path, "a") as file2:
                    file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
                    file2.write(f'coef of feature 0: {coef[idx,0,:][mask0[idx]]} \n')
                    file2.write(f'basis of feature 0: {basis_functions_name_list[0][mask0[idx]]} \n')
                    file2.write(f'coef of feature 1: {coef[idx,1,:][mask1[idx]]} \n')
                    file2.write(f'basis of feature 1: {basis_functions_name_list[1][mask1[idx]]} \n\n')