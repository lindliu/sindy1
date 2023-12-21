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

path_Exp1 = os.path.join(os.getcwd(), 'Exp1_Lotka_Volterra/results/')
path_Exp2 = os.path.join(os.getcwd(), 'Exp2_Modified_Lotka_Volterra/results/')
path_Exp3 = os.path.join(os.getcwd(), 'Exp3_Brusselator/results/')
path_Exp4 = os.path.join(os.getcwd(), 'Exp4_Van_der_Pol/results/')
path_Exp5 = os.path.join(os.getcwd(), 'Exp5_Lorenz/results/')


exp_idx = 5 ###1,2,3,4,5

if exp_idx == 1:
    import Lotka_constants as constants
    from Lotka_constants import get_basis_functions
    
    func_name = 'Exp1_Lotka_Volterra'
    path_data = path_Exp1
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
    path_data = path_Exp2
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
    path_data = path_Exp3
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
    path_data = path_Exp4
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
    path_data = path_Exp5
    num_traj, num_feature = 6, 3
    
    basis_type = 'poly' ##'mix_same' ## 'mix_diff' ##
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
        path_ = os.path.join(path_data, f'coeff/sindy_{suffix}_{i}.npy')
        coeff_sindy_all[i,:,:] = np.load(path_)
        
    #### results from gsindy one by one
    coeff_gsindy_one = np.zeros([num_traj, num_feature, num_basis])
    for i in range(num_traj):
        path_ = os.path.join(path_data, f'coeff/gsindy_{suffix}_{i}.npy')
        coeff_gsindy_one[i,:,:] = np.load(path_)
                
    #### results from gsindy all together
    path_gsindy_all = glob(os.path.join(path_data, f'coeff/gsindy_all_{suffix}*.npy'))
    coeff_gsindy_all = np.load(path_gsindy_all[0])
    


    # #### results from gsindy one by one
    # max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in \
    #                  glob(os.path.join(path_data, f'coeff/gsindy_one_{suffix}*.npy'))])
    # n_split = max_split-2+1   ##4
    # coeff_gsindy_one_by_one = np.zeros([n_split, num_traj, num_feature, num_basis])
    # for j in range(num_traj):
    #     for k in range(n_split):
    #         path_ = glob(os.path.join(path_data, f'coeff/gsindy_one_{suffix}_{k+2}_{j}.npy'))[0]
    #         coeff_gsindy_one_by_one[k,j,:,:] = np.load(path_)
            
            
            
    basis_function_name = np.array(basis_function_name)
    path_base = os.path.join(f'Results/{func_name}')
    os.makedirs(path_base, exist_ok=True)
    
    
    ### plot real coefficients vs predicted 
    os.makedirs(os.path.join(path_base, 'figures'), exist_ok=True)
    basis_idx = np.arange(num_basis)
    for i in range(num_traj):
        fig, ax = plt.subplots(3,num_feature,figsize=[10,8])
        fig.suptitle(f'{func_name} {suffix} with trajectory {i}')
        for j in range(num_feature):
            ax[0,j].scatter(basis_idx, coeff_true[i,j,:], c='b', alpha=.3)
            ax[0,j].scatter(basis_idx, coeff_sindy_all[i,j,:], c='r', alpha=.3)
            ax[0,j].set_title(f'{j}th feature: True vs SINDy')
            
            ax[1,j].scatter(basis_idx, coeff_true[i,j,:], c='b', alpha=.3)
            ax[1,j].scatter(basis_idx, coeff_gsindy_one[i,j,:], c='r', alpha=.3)
            ax[1,j].set_title(f'{j}th feature: True vs GS-SINDy one')
        
            ax[2,j].scatter(basis_idx, coeff_true[i,j,:], c='b', alpha=.3)
            ax[2,j].scatter(basis_idx, coeff_gsindy_all[i,j,:], c='r', alpha=.3)
            ax[2,j].set_title(f'{j}th feature: True vs GS-SINDy all')
            ax[2,j].set_xlabel('basis functions index')
            
        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(os.path.join(path_base, f'figures/{func_name} {suffix} with trajectory {i}'), dpi=200)
        
        




    ### record all true and predicted coefficients
    save_path = os.path.join(path_base, f'{func_name} {suffix} coefficients all.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file:
        np.set_printoptions(formatter={'float': lambda x: "{:.4f}".format(x)})
        for i in range(num_traj):
            for j in range(num_feature):
                mask = (np.c_[coeff_sindy_all[:,j,:].any(0), coeff_gsindy_one[:,j,:].any(0), coeff_gsindy_all[:,j,:].any(0)]).any(1)

                file.writelines(['*'*15, f' trajectory {i} feature {j} ', '*'*15, '\n'])
                file.writelines([f'{basis_function_name[j][mask]}\n'])
                file.writelines(['*'*15, '  coeff_sindy_all ', '*'*15, '\n'])
                file.writelines([f'{coeff_sindy_all[i,j,:][mask]}\n'])
                file.writelines(['*'*15, ' coeff_gsindy_one ', '*'*15, '\n'])
                file.writelines([f'{coeff_gsindy_one[i,j,:][mask]}\n'])
                file.writelines(['*'*15, ' coeff_gsindy_all ', '*'*15, '\n'])
                file.writelines([f'{coeff_gsindy_all[i,j,:][mask]}\n'])
                file.writelines(['*'*15, ' coeff_true ', '*'*15, '\n'])
                file.writelines([f'{coeff_true[i,j,:][mask]}\n\n'])
            
        
        
        
        
    ### record true and predicted coefficients of true basis functions
    save_path = os.path.join(path_base, f'{func_name} {suffix} coefficients.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file:
        np.set_printoptions(formatter={'float': lambda x: "{:.4f}".format(x)})
        for i in range(num_traj):
            mask_true = coeff_true[i]!=0
            
            file.writelines(['*'*15, f' trajectory {i} ', '*'*15, '\n'])
            file.writelines([f'{basis_function_name[mask_true]}\n'])
            file.writelines(['*'*15, '  coeff_sindy_all ', '*'*15, '\n'])
            file.writelines([f'{coeff_sindy_all[i][mask_true]}\n'])
            file.writelines(['*'*15, ' coeff_gsindy_one ', '*'*15, '\n'])
            file.writelines([f'{coeff_gsindy_one[i][mask_true]}\n'])
            file.writelines(['*'*15, ' coeff_gsindy_all ', '*'*15, '\n'])
            file.writelines([f'{coeff_gsindy_all[i][mask_true]}\n'])
            file.writelines(['*'*15, ' coeff_true ', '*'*15, '\n'])
            file.writelines([f'{coeff_true[i][mask_true]}\n\n'])
            
            
            






    ### record metrics: rmse precision and recal
    save_path = os.path.join(path_base, f'{func_name} {suffix} rmse_precision_recall.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file1:
        rmse_sindy = np.linalg.norm(coeff_true-coeff_sindy_all, axis=(1,2)) / np.linalg.norm(coeff_true, axis=(1,2))
        rmse_gsindy_one = np.linalg.norm(coeff_true-coeff_gsindy_one, axis=(1,2)) / np.linalg.norm(coeff_true, axis=(1,2))
        rmse_gsindy_all = np.linalg.norm(coeff_true-coeff_gsindy_all, axis=(1,2)) / np.linalg.norm(coeff_true, axis=(1,2))

        np.set_printoptions(formatter={'float': lambda x: "{:.2e}".format(x)})
        file1.writelines(['*'*15, ' rmse_sindy ', '*'*15, '\n'])
        file1.write(f'{rmse_sindy} \n')
        file1.writelines(['*'*15, ' rmse_gsindy_one ', '*'*15, '\n'])
        file1.write(f'{rmse_gsindy_one} \n')
        file1.writelines(['*'*15, ' rmse_gsindy_all ', '*'*15, '\n'])
        file1.write(f'{rmse_gsindy_all} \n\n')
        
        
        mp_sindy      = ((coeff_true*coeff_sindy_all)!=0).sum(axis=(1,2)) /  (coeff_sindy_all!=0).sum(axis=(1,2))
        mp_gsindy_one = ((coeff_true*coeff_gsindy_one)!=0).sum(axis=(1,2)) / (coeff_gsindy_one!=0).sum(axis=(1,2))
        mp_gsindy_all = ((coeff_true*coeff_gsindy_all)!=0).sum(axis=(1,2)) / (coeff_gsindy_all!=0).sum(axis=(1,2))

        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
        file1.writelines(['*'*15, ' metric precision sindy ', '*'*15, '\n'])
        file1.write(f'{mp_sindy} \n')
        file1.writelines(['*'*15, ' metric precision gsindy one ', '*'*15, '\n'])
        file1.write(f'{mp_gsindy_one} \n')
        file1.writelines(['*'*15, ' metric precision gsindy all ', '*'*15, '\n'])
        file1.write(f'{mp_gsindy_all} \n\n')
        
        
        mr_sindy      = ((coeff_true*coeff_sindy_all)!=0).sum(axis=(1,2)) / (coeff_true!=0).sum(axis=(1,2))
        mr_gsindy_one = ((coeff_true*coeff_gsindy_one)!=0).sum(axis=(1,2)) / (coeff_true!=0).sum(axis=(1,2))
        mr_gsindy_all = ((coeff_true*coeff_gsindy_all)!=0).sum(axis=(1,2)) / (coeff_true!=0).sum(axis=(1,2))
        
        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
        file1.writelines(['*'*15, ' metric recall sindy ', '*'*15, '\n'])
        file1.write(f'{mr_sindy} \n')
        file1.writelines(['*'*15, ' metric recall gsindy one ', '*'*15, '\n'])
        file1.write(f'{mr_gsindy_one} \n')
        file1.writelines(['*'*15, ' metric recall gsindy all ', '*'*15, '\n'])
        file1.write(f'{mr_gsindy_all} \n\n')
        