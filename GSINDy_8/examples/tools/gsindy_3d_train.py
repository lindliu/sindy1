#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:21:31 2023

@author: dliu
"""


from gsindy_3d import fit_gsindy_3d, model_selection_gsindy_3d, process_gsindy_one_3d
from utils import data_generator
import os
import numpy as np

def gsindy_3d_train(func, t, x0_list, a_list, real_list, suffix, basis, precision, alpha, opt, deriv_spline, ensemble, path_base='results', \
                    threshold_sindy_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0], \
                    threshold_group_list = [1e-3, 1e-2], \
                    threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]):
    
    os.makedirs(path_base, exist_ok=True)
    os.makedirs(os.path.join(path_base, 'coeff'), exist_ok=True)

    # basis_functions_list = basis['functions']
    basis_functions_name_list = basis['names']
    
    for num in [1, len(a_list)]:
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
                    
                    model_set, diff0_basis_list, diff1_basis_list, diff2_basis_list, same0_basis_list, same1_basis_list, same2_basis_list, parameter_list = \
                                                fit_gsindy_3d(sol_org_list, num_traj, t_, num, real_list, \
                                                              basis, precision, alpha, opt, deriv_spline, ensemble, \
                                                              threshold_sindy_list = threshold_sindy_list, \
                                                              threshold_group_list = threshold_group_list,\
                                                              threshold_similarity_list = threshold_similarity_list_, 
                                                              print_results=False)
                    
                    ms, best_BIC_model, parameter_list = model_selection_gsindy_3d( \
                                                            x0, t_, a, real_list, basis, model_set, \
                                                            sol_org_list, same0_basis_list, same1_basis_list, same2_basis_list, parameter_list)
    
                    coef = model_set[best_BIC_model].mean(0)
                    np.save(os.path.join(path_base, f'coeff/gsindy_one_{suffix}_{num_split}_{idx}.npy'), coef)
                    
                    mask0 = coef[0,:]!=0
                    mask1 = coef[1,:]!=0
                    mask2 = coef[2,:]!=0
                    with open(save_path, "a") as file1:
                        file1.writelines(['*'*15, f'result of trajectory {idx} split into {num_split} pieces', '*'*15, '\n'])
                        file1.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
                        file1.write(f'basis of feature 0: {basis_functions_name_list[0][mask0]} \n')
                        file1.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
                        file1.write(f'basis of feature 1: {basis_functions_name_list[1][mask1]} \n')
                        file1.write(f'coef of feature 2: {coef[2,:][mask2]} \n')
                        file1.write(f'basis of feature 2: {basis_functions_name_list[2][mask2]} \n\n')
                    
                        
            process_gsindy_one_3d(func, x0_list, a_list, t, real_list, basis, suffix, num_traj=len(a_list), num_feature=3)
        
        if num>1:
            save_path = os.path.join(path_base, f'gsindy_all_{suffix}.txt')
            open(save_path, 'w').close()
            
            x0 = x0_list[:num]
            a = a_list[:num]
            
            t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num)
            
            model_set, diff0_basis_list, diff1_basis_list, diff2_basis_list, same0_basis_list, same1_basis_list, same2_basis_list, parameter_list = \
                                        fit_gsindy_3d(sol_org_list, num_traj, t_, num, real_list, \
                                                      basis, precision, alpha, opt, deriv_spline, ensemble, \
                                                      threshold_sindy_list = threshold_sindy_list, \
                                                      threshold_group_list = threshold_group_list,\
                                                      threshold_similarity_list = threshold_similarity_list_, 
                                                      print_results=False)
            
            ms, best_BIC_model, parameter_list = model_selection_gsindy_3d( \
                                                    x0, t_, a, real_list, basis, model_set, sol_org_list, \
                                                    same0_basis_list, same1_basis_list, same2_basis_list, parameter_list)

            coef = model_set[best_BIC_model] ### num_traj, num_feature, num_basis
            np.save(os.path.join(path_base, f'coeff/gsindy_all_{suffix}_{num}.npy'), coef)

            mask0 = coef[:,0,:]!=0
            mask1 = coef[:,1,:]!=0
            mask2 = coef[:,2,:]!=0
            for idx in range(num_traj):
                with open(save_path, "a") as file2:
                    file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
                    file2.write(f'coef of feature 0: {coef[idx,0,:][mask0[idx]]} \n')
                    file2.write(f'basis of feature 0: {basis_functions_name_list[0][mask0[idx]]} \n')
                    file2.write(f'coef of feature 1: {coef[idx,1,:][mask1[idx]]} \n')
                    file2.write(f'basis of feature 1: {basis_functions_name_list[1][mask1[idx]]} \n')
                    file2.write(f'coef of feature 1: {coef[idx,2,:][mask2[idx]]} \n')
                    file2.write(f'basis of feature 1: {basis_functions_name_list[2][mask2[idx]]} \n\n')