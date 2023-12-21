#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:29:29 2023

@author: dliu
"""

import sys
sys.path.insert(1, './Exp1_Lotka_Volterra')
sys.path.insert(1, './Exp2_Modified_Lotka_Volterra')
sys.path.insert(1, './Exp3_Brusselator')
sys.path.insert(1, './Exp4_Van_der_Pol')
sys.path.insert(1, './Exp5_Lorenz')
sys.path.insert(1, '../GSINDy')

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})

path_Exp1 = os.path.join(os.getcwd(), 'Exp1_Lotka_Volterra/results/')
path_Exp2 = os.path.join(os.getcwd(), 'Exp2_Modified_Lotka_Volterra/results/')
path_Exp3 = os.path.join(os.getcwd(), 'Exp3_Brusselator/results/')
path_Exp4 = os.path.join(os.getcwd(), 'Exp4_Van_der_Pol/results/')
path_Exp5 = os.path.join(os.getcwd(), 'Exp5_Lorenz/results/')


exp_idx = 4#2,3,4,5,6

if exp_idx == 1:
    import Lotka_constants as constants
    from Lotka_constants import get_basis_functions
    
    func_name = 'Exp1_Lotka_Volterra'
    path_base = path_Exp1
    num_traj, num_feature = 6, 2
    
    basis_type = 'poly' ##'mix'##
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    
    basis_function_name = basis['names']
    num_basis = basis_function_name[0].shape[0]
    a_list = constants.a_list

    ##### true results
    coeff_true_ = np.array(a_list)
    coeff_true = np.zeros([num_traj, num_feature, num_basis])
    coeff_true[:,0,[1,4]] = coeff_true_[:,[0,1]]
    coeff_true[:,1,[2,4]] = coeff_true_[:,[1,0]]
    
elif exp_idx == 2:
    import M_Lotka_constants as constants
    from M_Lotka_constants import get_basis_functions
    
    func_name = 'Exp2_Modified_Lotka_Volterra'
    path_base = path_Exp2
    num_traj, num_feature = 6, 2
    
    basis_type = 'poly' ##'mix'##
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    
    basis_function_name = basis['names']
    num_basis = basis_function_name[0].shape[0]
    a_list = constants.a_list

    ##### true results
    coeff_true_ = np.array(a_list)
    coeff_true = np.zeros([num_traj, num_feature, num_basis])
    coeff_true[:,0,[2,3,6]] = coeff_true_[:,[1,0,2]]
    coeff_true[:,1,[2,7,9]] = coeff_true_[:,[0,1,2]]

    coeff_true[:,0,[8]] = [-1]
    coeff_true[:,1,[1]] = [1]
    
elif exp_idx == 3:
    import Brusselator_constants as constants
    from Brusselator_constants import get_basis_functions
    
    func_name = 'Exp3_Brusselator'
    path_base = path_Exp3
    num_traj, num_feature = 6, 2
    
    basis_type = 'poly' ##'mix'##
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    
    basis_function_name = basis['names']
    num_basis = basis_function_name[0].shape[0]
    a_list = constants.a_list

    ##### true results
    coeff_true_ = np.array(a_list)
    coeff_true = np.zeros([num_traj, num_feature, num_basis])
    coeff_true[:,0,[0]] = coeff_true_[:,[0]]
    coeff_true[:,1,[1]] = coeff_true_[:,[1]]

    coeff_true[:,0,[1,7]] = [-4, 1]
    coeff_true[:,1,[7]] = -1
    
elif exp_idx == 4:
    import Van_constants as constants
    from Van_constants import get_basis_functions
    
    func_name = 'Exp4_Van_der_Pol'
    path_base = path_Exp4
    num_traj, num_feature = 6, 2
    
    basis_type = 'poly' ##'mix'##
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    
    basis_function_name = basis['names']
    num_basis = basis_function_name[0].shape[0]
    a_list = constants.a_list

    ##### true results
    coeff_true_ = np.array(a_list)
    coeff_true = np.zeros([num_traj, num_feature, num_basis])
    coeff_true[:,0,[6]] = 5*coeff_true_[:,[0]]
    coeff_true[:,1,[1]] = coeff_true_[:,[1]]
    
    coeff_true[:,0,[1,2]] = [5,-5]
    
elif exp_idx == 5:
    import Lorenz_constants as constants
    from Lorenz_constants import get_basis_functions
    
    func_name = 'Exp5_Lorenz'
    path_base = path_Exp5
    num_traj, num_feature = 6, 3
    
    basis_type = 'poly' ##'mix'##
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    
    basis_function_name = basis['names']
    num_basis = basis_function_name[0].shape[0]
    a_list = constants.a_list

    ##### true results
    coeff_true_ = np.array(a_list)
    coeff_true = np.zeros([num_traj, num_feature, num_basis])
    coeff_true[:,0,[1,2]] = coeff_true_[:,[0,1]]
    coeff_true[:,1,[1]] = coeff_true_[:,[2]]
    coeff_true[:,2,[3]] = coeff_true_[:,[3]]
    
    coeff_true[:,1,[2,8]] = [-1,-1]
    coeff_true[:,2,[7]] = [1]
    

assert num_traj == len(a_list)
if __name__ == "__main__":
    
    #### results from sindy
    coeff_sindy_all = np.zeros([num_traj, num_feature, num_basis])
    for i in range(num_traj):
        path_ = os.path.join(path_base, f'coeff/sindy_{suffix}_{i}.npy')
        coeff_sindy_all[i,:,:] = np.load(path_)
        
    #### results from gsindy one by one
    coeff_gsindy_one = np.zeros([num_traj, num_feature, num_basis])
    for i in range(num_traj):
        path_ = os.path.join(path_base, f'coeff/gsindy_{suffix}_{i}.npy')
        coeff_gsindy_one[i,:,:] = np.load(path_)
                
    #### results from gsindy all together
    path_gsindy_all = glob(os.path.join(path_base, f'coeff/gsindy_all_{suffix}*.npy'))
    coeff_gsindy_all = np.load(path_gsindy_all[0])
    



    os.makedirs('figures', exist_ok=True)
    basis_idx = np.arange(num_basis)
    for i in range(num_traj):
        fig, ax = plt.subplots(3,2,figsize=[10,8])
        fig.suptitle(f'{func_name} {suffix} with trajectory {i}')
        ax[0,0].scatter(basis_idx, coeff_true[i,0,:], c='b', alpha=.3)
        ax[0,0].scatter(basis_idx, coeff_sindy_all[i,0,:], c='r', alpha=.3)
        ax[0,0].set_title('0th feature: True vs SINDy')
        ax[0,1].scatter(basis_idx, coeff_true[i,1,:], c='b', alpha=.3)
        ax[0,1].scatter(basis_idx, coeff_sindy_all[i,1,:], c='r', alpha=.3)
        ax[0,1].set_title('1th feature: True vs SINDy')
    
        ax[1,0].scatter(basis_idx, coeff_true[i,0,:], c='b', alpha=.3)
        ax[1,0].scatter(basis_idx, coeff_gsindy_one[i,0,:], c='r', alpha=.3)
        ax[1,0].set_title('0th feature: True vs GS-SINDy one')
        ax[1,1].scatter(basis_idx, coeff_true[i,1,:], c='b', alpha=.3)
        ax[1,1].scatter(basis_idx, coeff_gsindy_one[i,1,:], c='r', alpha=.3)
        ax[1,1].set_title('1th feature: True vs GS-SINDy one')
    
        ax[2,0].scatter(basis_idx, coeff_true[i,0,:], c='b', alpha=.3)
        ax[2,0].scatter(basis_idx, coeff_gsindy_all[i,0,:], c='r', alpha=.3)
        ax[2,0].set_title('0th feature: True vs GS-SINDy all')
        ax[2,1].scatter(basis_idx, coeff_true[i,1,:], c='b', alpha=.3)
        ax[2,1].scatter(basis_idx, coeff_gsindy_all[i,1,:], c='r', alpha=.3)
        ax[2,1].set_title('1th feature: True vs GS-SINDy all')
        
        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(f'figures/{func_name} {suffix} with trajectory {i}', dpi=200)
        
        

    coeff_true

    # os.makedirs(path_base, exist_ok=True)
    # os.makedirs(os.path.join(path_base, 'coeff'), exist_ok=True)
    # save_path = os.path.join(path_base, f'sindy_all_{suffix}.txt')
    # open(save_path, 'w').close()
    
    # for idx in range(len(model_best_list)):
    #     # coef = model_best_list[idx].coefficients()
    #     coef = model_best_list[idx]
    #     np.save(os.path.join(path_base, f'coeff/sindy_{suffix}_{idx}.npy'), coef)

    #     mask0 = np.abs(coef[0]) > precision
    #     mask1 = np.abs(coef[1]) > precision
    #     with open(save_path, "a") as file2:
    #         file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
    #         file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
    #         file2.write(f'basis of feature 0: {basis_functions_name_list_[0][mask0]} \n')
    #         file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
    #         file2.write(f'basis of feature 1: {basis_functions_name_list_[1][mask1]} \n\n')
    