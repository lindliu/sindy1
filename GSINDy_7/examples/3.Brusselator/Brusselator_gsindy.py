#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:43:31 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func8, monomial_poly, monomial_poly_name, monomial_all, monomial_all_name
from utils import monomial_all_0, monomial_all_1, monomial_all_0_name, monomial_all_1_name
from train_gsindy_2d import data_generator, fit_gsindy_2d, model_selection_gsindy_2d

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = [monomial_all]
# monomial_name = [monomial_all_name]
monomial = [monomial_poly, monomial_poly]
monomial_name = [monomial_poly_name, monomial_poly_name]
# monomial = [monomial_all_0, monomial_all_1]
# monomial_name = [monomial_all_0_name, monomial_all_1_name]
num_split = None

dt = .1    
t = np.arange(0,20,dt)
num = 6

##################### 2 variable ####################
x0_list = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
a_list = [(1, 3), (.8, 2.5), (.6, 2.6), (.9, 2.8), (.5, 2.8), (.7, 2.8)]

if num==1:
    idx = 0 #calculate 0,1,2,3,4,5 one by one
    x0 = [x0_list[idx]]
    a = [a_list[idx]]
    num_split = 4
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]

func = func8
real0 = "x'=a-4x+x^2y"
real1 = "y'=bx-x^2y"
real_list = [real0, real1]
######################################################





if __name__ == "__main__":
    
    # t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
    
    # model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
    #                             fit_gsindy_2d(sol_org_list, num_traj, num_feature, t_, num, real_list, \
    #                                           monomial, monomial_name, precision, alpha, opt, deriv_spline, \
    #                                           ensemble, print_results=False)
    
    # ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
    #                                         x0, t_, a, real_list, monomial, monomial_name, model_set, \
    #                                         sol_org_list, same0_basis_list, same1_basis_list, parameter_list)
    
    # if num==1:
    #     coef = model_set[best_BIC_model].mean(0)
    #     mask0 = coef[0,:]!=0
    #     mask1= coef[1,:]!=0
    #     print('*'*75)
    #     print('*'*15, f'result of trajectory {idx} split into {num_split} pieces', '*'*15)
    #     print('*'*75)
    #     print('coef of feature 0: ', coef[0,:][mask0])
    #     print('basis of feature 0: ', monomial_name[mask0])
    #     print('coef of feature 1: ', coef[1,:][mask1])
    #     print('basis of feature 1: ', monomial_name[mask1])
    
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/coeff', exist_ok=True)

    for num in [1, len(a_list)]:
        
        if num==1:
            save_path = f'results/gsindy_one_by_one_{monomial[0].__name__}.txt'
            open(save_path, 'w').close()
            
            max_split = 5
            for num_split in range(2, max_split+1):
                for idx in range(len(a_list)):
                    x0 = [x0_list[idx]]
                    a = [a_list[idx]]
                    
                    t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
                    
                    model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                                fit_gsindy_2d(sol_org_list, num_traj, num_feature, t_, num, real_list, \
                                                              monomial, monomial_name, precision, alpha, opt, deriv_spline, \
                                                              ensemble, print_results=False)
                    
                    ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                            x0, t_, a, real_list, monomial, monomial_name, model_set, \
                                                            sol_org_list, same0_basis_list, same1_basis_list, parameter_list)
    
                    coef = model_set[best_BIC_model].mean(0)
                    np.save(f'results/coeff/gsindy_one_{monomial[0].__name__}_{num_split}_{idx}.npy', coef)
                    
                    mask0 = coef[0,:]!=0
                    mask1= coef[1,:]!=0
                    with open(save_path, "a") as file1:
                        file1.writelines(['*'*15, f'result of trajectory {idx} split into {num_split} pieces', '*'*15, '\n'])
                        file1.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
                        file1.write(f'basis of feature 0: {monomial_name[0][mask0]} \n')
                        file1.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
                        file1.write(f'basis of feature 1: {monomial_name[1][mask1]} \n\n')
                        
        
        if num>1:
            save_path = f'results/gsindy_all_{monomial[0].__name__}.txt'
            open(save_path, 'w').close()
            
            x0 = x0_list[:num]
            a = a_list[:num]
            
            t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list, num, num_split)
            
            model_set, diff0_basis_list, diff1_basis_list, same0_basis_list, same1_basis_list, parameter_list = \
                                        fit_gsindy_2d(sol_org_list, num_traj, num_feature, t_, num, real_list, \
                                                      monomial, monomial_name, precision, alpha, opt, deriv_spline, \
                                                      ensemble, print_results=False)
            
            ms, best_BIC_model, parameter_list = model_selection_gsindy_2d( \
                                                    x0, t_, a, real_list, monomial, monomial_name, model_set, \
                                                    sol_org_list, same0_basis_list, same1_basis_list, parameter_list)

            coef = model_set[best_BIC_model] ### num_traj, num_feature, num_basis
            np.save(f'results/coeff/gsindy_all_{monomial[0].__name__}_{num}.npy', coef)

            mask0 = coef[:,0,:]!=0
            mask1= coef[:,1,:]!=0
            
            for idx in range(num_traj):
                with open(save_path, "a") as file2:
                    file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
                    file2.write(f'coef of feature 0: {coef[idx,0,:][mask0[idx]]} \n')
                    file2.write(f'basis of feature 0: {monomial_name[0][mask0[idx]]} \n')
                    file2.write(f'coef of feature 1: {coef[idx,1,:][mask1[idx]]} \n')
                    file2.write(f'basis of feature 1: {monomial_name[1][mask1[idx]]} \n\n')