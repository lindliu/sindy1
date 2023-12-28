#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:29:29 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../Exp1_Lotka_Volterra')
sys.path.insert(1, '../Exp2_Modified_Lotka_Volterra')
sys.path.insert(1, '../Exp3_Brusselator')
sys.path.insert(1, '../Exp4_Van_der_Pol')
sys.path.insert(1, '../Exp5_Lorenz')
sys.path.insert(1, '../../GSINDy')

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

path_Exp1 = os.path.join(os.getcwd(), '../Exp1_Lotka_Volterra/results/')
path_Exp2 = os.path.join(os.getcwd(), '../Exp2_Modified_Lotka_Volterra/results/')
path_Exp3 = os.path.join(os.getcwd(), '../Exp3_Brusselator/results/')
path_Exp4 = os.path.join(os.getcwd(), '../Exp4_Van_der_Pol/results/')
path_Exp5 = os.path.join(os.getcwd(), '../Exp5_Lorenz/results/')


exp_idx = 5 ###1,2,3,4,5

if exp_idx == 1:
    import Lotka_constants as constants
    from Lotka_constants import get_basis_functions
    
    func_name = 'Exp1_Lotka_Volterra'
    path_data = path_Exp1
    num_traj, num_feature = 6, 2
    a_list = constants.a_list
    
    ########## poly basis functions ############
    basis_type = 'poly'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_poly = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_poly = np.array(basis['names'])
    num_basis_poly = basis_function_name_poly.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    coeff_true_poly[:,0,[1,4]] = coeff_true_[:,[0,1]]
    coeff_true_poly[:,1,[2,4]] = coeff_true_[:,[1,0]]
    
    ########## mix basis functions ############
    basis_type = 'mix'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_mix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_mix = np.array(basis['names'])
    num_basis_mix = basis_function_name_mix.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    coeff_true_mix[:,0,[1,4]] = coeff_true_[:,[0,1]]
    coeff_true_mix[:,1,[2,4]] = coeff_true_[:,[1,0]]
    
elif exp_idx == 2:
    import M_Lotka_constants as constants
    from M_Lotka_constants import get_basis_functions
    
    func_name = 'Exp2_Modified_Lotka_Volterra'
    path_data = path_Exp2
    num_traj, num_feature = 6, 2
    a_list = constants.a_list
    
    ########## poly basis functions ############
    basis_type = 'poly'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_poly = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_poly = np.array(basis['names'])
    num_basis_poly = basis_function_name_poly.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    coeff_true_poly[:,0,[2,3,6]] = coeff_true_[:,[1,0,2]]
    coeff_true_poly[:,1,[2,7,9]] = coeff_true_[:,[0,1,2]]

    coeff_true_poly[:,0,[8]] = [-1]
    coeff_true_poly[:,1,[1]] = [1]
    
    ########## mix basis functions ############
    basis_type = 'mix'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_mix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_mix = np.array(basis['names'])
    num_basis_mix = basis_function_name_mix.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    coeff_true_mix[:,0,[2,3,6]] = coeff_true_[:,[1,0,2]]
    coeff_true_mix[:,1,[2,7,9]] = coeff_true_[:,[0,1,2]]

    coeff_true_mix[:,0,[8]] = [-1]
    coeff_true_mix[:,1,[1]] = [1]
    
elif exp_idx == 3:
    import Brusselator_constants as constants
    from Brusselator_constants import get_basis_functions
    
    func_name = 'Exp3_Brusselator'
    path_data = path_Exp3
    num_traj, num_feature = 6, 2
    a_list = constants.a_list
    
    ########## poly basis functions ############
    basis_type = 'poly'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_poly = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_poly = np.array(basis['names'])
    num_basis_poly = basis_function_name_poly.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    coeff_true_poly[:,0,[0]] = coeff_true_[:,[0]]
    coeff_true_poly[:,1,[1]] = coeff_true_[:,[1]]

    coeff_true_poly[:,0,[1,7]] = [-4, 1]
    coeff_true_poly[:,1,[7]] = -1
    
    ########## mix basis functions ############
    basis_type = 'mix'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_mix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_mix = np.array(basis['names'])
    num_basis_mix = basis_function_name_mix.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    coeff_true_mix[:,0,[0]] = coeff_true_[:,[0]]
    coeff_true_mix[:,1,[1]] = coeff_true_[:,[1]]

    coeff_true_mix[:,0,[1,7]] = [-4, 1]
    coeff_true_mix[:,1,[7]] = -1
    
elif exp_idx == 4:
    import Van_constants as constants
    from Van_constants import get_basis_functions
    
    func_name = 'Exp4_Van_der_Pol'
    path_data = path_Exp4
    num_traj, num_feature = 6, 2
    a_list = constants.a_list
    
    ########## poly basis functions ############
    basis_type = 'poly'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_poly = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_poly = np.array(basis['names'])
    num_basis_poly = basis_function_name_poly.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    coeff_true_poly[:,0,[6]] = 5*coeff_true_[:,[0]]
    coeff_true_poly[:,1,[1]] = coeff_true_[:,[1]]
    
    coeff_true_poly[:,0,[1,2]] = [5,-5]
    
    ########## mix basis functions ############
    basis_type = 'mix'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_mix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_mix = np.array(basis['names'])
    num_basis_mix = basis_function_name_mix.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    coeff_true_mix[:,0,[6]] = 5*coeff_true_[:,[0]]
    coeff_true_mix[:,1,[1]] = coeff_true_[:,[1]]
    
    coeff_true_mix[:,0,[1,2]] = [5,-5]
    
elif exp_idx == 5:
    import Lorenz_constants as constants
    from Lorenz_constants import get_basis_functions
    
    func_name = 'Exp5_Lorenz'
    path_data = path_Exp5
    num_traj, num_feature = 6, 3
    a_list = constants.a_list
    
    ########## poly basis functions ############
    basis_type = 'poly'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_poly = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_poly = np.array(basis['names'])
    num_basis_poly = basis_function_name_poly.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    coeff_true_poly[:,0,[1,2]] = coeff_true_[:,[0,1]]
    coeff_true_poly[:,1,[1]] = coeff_true_[:,[2]]
    coeff_true_poly[:,2,[3]] = coeff_true_[:,[3]]
    
    coeff_true_poly[:,1,[2,8]] = [-1,-1]
    coeff_true_poly[:,2,[7]] = [1]
    
    ########## mix basis functions ############
    basis_type = 'mix_diff'
    basis, opt = get_basis_functions(basis_type, GSINDY=True)
    suffix_mix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
    basis_function_name_mix = np.array(basis['names'])
    num_basis_mix = basis_function_name_mix.shape[1]

    ### true results
    coeff_true_ = np.array(a_list)
    coeff_true_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    coeff_true_mix[:,0,[1,2]] = coeff_true_[:,[0,1]]
    coeff_true_mix[:,1,[1]] = coeff_true_[:,[2]]
    coeff_true_mix[:,2,[3]] = coeff_true_[:,[3]]
    
    coeff_true_mix[:,1,[2,8]] = [-1,-1]
    coeff_true_mix[:,2,[7]] = [1]
    

assert num_traj == len(a_list)
if __name__ == "__main__":
    
    #### results from sindy
    coeff_sindy_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    for i in range(num_traj):
        path_ = os.path.join(path_data, f'coeff/sindy_{suffix_poly}_{i}.npy')
        coeff_sindy_poly[i,:,:] = np.load(path_)
        
    coeff_sindy_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    for i in range(num_traj):
        path_ = os.path.join(path_data, f'coeff/sindy_{suffix_mix}_{i}.npy')
        coeff_sindy_mix[i,:,:] = np.load(path_)
            
            
    #### results from gsindy one by one
    coeff_gsindy_one_poly = np.zeros([num_traj, num_feature, num_basis_poly])
    for i in range(num_traj):
        path_ = os.path.join(path_data, f'coeff/gsindy_{suffix_poly}_{i}.npy')
        coeff_gsindy_one_poly[i,:,:] = np.load(path_)
        
    coeff_gsindy_one_mix = np.zeros([num_traj, num_feature, num_basis_mix])
    for i in range(num_traj):
        path_ = os.path.join(path_data, f'coeff/gsindy_{suffix_mix}_{i}.npy')
        coeff_gsindy_one_mix[i,:,:] = np.load(path_)
            
        
    #### results from gsindy all together
    path_gsindy_all_poly = glob(os.path.join(path_data, f'coeff/gsindy_all_{suffix_poly}*.npy'))
    coeff_gsindy_all_poly = np.load(path_gsindy_all_poly[0])
    
    path_gsindy_all_mix = glob(os.path.join(path_data, f'coeff/gsindy_all_{suffix_mix}*.npy'))
    coeff_gsindy_all_mix = np.load(path_gsindy_all_mix[0])


    # #### results from gsindy one by one
    # max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in \
    #                  glob(os.path.join(path_data, f'coeff/gsindy_one_{suffix}*.npy'))])
    # n_split = max_split-2+1   ##4
    # coeff_gsindy_one_by_one = np.zeros([n_split, num_traj, num_feature, num_basis])
    # for j in range(num_traj):
    #     for k in range(n_split):
    #         path_ = glob(os.path.join(path_data, f'coeff/gsindy_one_{suffix}_{k+2}_{j}.npy'))[0]
    #         coeff_gsindy_one_by_one[k,j,:,:] = np.load(path_)
            
    
    
        
        
        
    # coeff_sindy_poly
    # coeff_gsindy_one_poly
    # coeff_gsindy_all_poly
    # coeff_true_poly
    
    
    # coeff_sindy_mix
    # coeff_gsindy_one_mix
    # coeff_gsindy_all_mix
    # coeff_true_mix
    
    
    directory = 'Results'
    path_base = os.path.join(f'{directory}/{func_name}')
    os.makedirs(path_base, exist_ok=True)
    
    
    bound = 1e-3#constants.precision
    coeff_sindy_poly[np.abs(coeff_sindy_poly)<bound] = 0
    coeff_gsindy_one_poly[np.abs(coeff_gsindy_one_poly)<bound] = 0
    coeff_gsindy_all_poly[np.abs(coeff_gsindy_all_poly)<bound] = 0 
    
    coeff_sindy_mix[np.abs(coeff_sindy_mix)<bound] = 0
    coeff_gsindy_one_mix[np.abs(coeff_gsindy_one_mix)<bound] = 0
    coeff_gsindy_all_mix[np.abs(coeff_gsindy_all_mix)<bound] = 0 

    
    
    
    
    
    
    ######################################################
    ############# generate rmse mp mr table###############
    ######################################################
    def get_rmse_mp_mr(real, pred):
        ## get rmse, precision and recall
        ## dimension of both inputs: num_traj, num_feature, num_basis
        rmse = np.linalg.norm(real-pred, axis=(1,2)) / np.linalg.norm(real, axis=(1,2))
        mp = ((real*pred)!=0).sum(axis=(1,2)) / (pred!=0).sum(axis=(1,2))
        mr = ((real*pred)!=0).sum(axis=(1,2)) / (real!=0).sum(axis=(1,2))
        
        return rmse, mp, mr
    
    rmse_sindy_poly, mp_sindy_poly, mr_sindy_poly = get_rmse_mp_mr(coeff_true_poly, coeff_sindy_poly)
    rmse_gsindy_one_poly, mp_gsindy_one_poly, mr_gsindy_one_poly = get_rmse_mp_mr(coeff_true_poly, coeff_gsindy_one_poly)
    rmse_gsindy_all_poly, mp_gsindy_all_poly, mr_gsindy_all_poly = get_rmse_mp_mr(coeff_true_poly, coeff_gsindy_all_poly)

    rmse_sindy_mix, mp_sindy_mix, mr_sindy_mix = get_rmse_mp_mr(coeff_true_mix, coeff_sindy_mix)
    rmse_gsindy_one_mix, mp_gsindy_one_mix, mr_gsindy_one_mix = get_rmse_mp_mr(coeff_true_mix, coeff_gsindy_one_mix)
    rmse_gsindy_all_mix, mp_gsindy_all_mix, mr_gsindy_all_mix = get_rmse_mp_mr(coeff_true_mix, coeff_gsindy_all_mix)

    
    table_sindy = np.zeros([3, num_traj*2+1], dtype=object)
    table_sindy[:,0] = ['RMSE', 'Mp', 'Mr']
    table_sindy[0,1:7] = rmse_sindy_poly
    table_sindy[1,1:7] = mp_sindy_poly
    table_sindy[2,1:7] = mr_sindy_poly
    table_sindy[0,7:] = rmse_sindy_mix
    table_sindy[1,7:] = mp_sindy_mix
    table_sindy[2,7:] = mr_sindy_mix
    
    table_gindy_one = np.zeros([3, num_traj*2+1], dtype=object)
    table_gindy_one[:,0] = ['RMSE', 'Mp', 'Mr']
    table_gindy_one[0,1:7] = rmse_gsindy_one_poly
    table_gindy_one[1,1:7] = mp_gsindy_one_poly
    table_gindy_one[2,1:7] = mr_gsindy_one_poly
    table_gindy_one[0,7:] = rmse_gsindy_one_mix
    table_gindy_one[1,7:] = mp_gsindy_one_mix
    table_gindy_one[2,7:] = mr_gsindy_one_mix
        
    table_gindy_all = np.zeros([3, num_traj*2+1], dtype=object)
    table_gindy_all[:,0] = ['RMSE', 'Mp', 'Mr']
    table_gindy_all[0,1:7] = rmse_gsindy_all_poly
    table_gindy_all[1,1:7] = mp_gsindy_all_poly
    table_gindy_all[2,1:7] = mr_gsindy_all_poly
    table_gindy_all[0,7:] = rmse_gsindy_all_mix
    table_gindy_all[1,7:] = mp_gsindy_all_mix
    table_gindy_all[2,7:] = mr_gsindy_all_mix
    
    
    def get_latex_line(input_list, print_type='.2f'):
        
        line = ['&']
        line.append('$'+input_list[0]+'$')
        line.append('&')
        for ele in input_list[1:]:
            if print_type[-1]=='f':
                line.append(f'{ele:{print_type}}'.rstrip('0'))
            else:
                ss = f'{ele:{print_type}}'
                if ss[-2]=='0':
                    line.append(ss[:-2]+ss[-1])
                else:
                    line.append(ss)
            line.append('&')
        line = line[:-1]
        line.append('\\\ \n')
        return line
    
    ### record metrics: rmse precision and recal
    save_path = os.path.join(path_base, f'{func_name} metrics table.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file:
        file.write(f'the coefficients will keep if its absolute value >= {bound} \n')
        file.writelines(['columns: traj1(poly, mix), traj2(poly, mix), traj3(poly, mix), traj4(poly, mix), traj5(poly, mix), traj6(poly, mix)', '\n'])
        
        
        file.writelines(['\n', '*'*15, ' sindy ', '*'*15, '\n'])
        line_ = get_latex_line(table_sindy[0], '.2e')
        file.writelines(line_)
        for i in range(1,3):
            line_ = get_latex_line(table_sindy[i])
            file.writelines(line_)
            
            
        file.writelines(['\n', '*'*15, ' gsindy one ', '*'*15, '\n'])
        line_ = get_latex_line(table_gindy_one[0], '.2e')
        file.writelines(line_)
        for i in range(1,3):
            line_ = get_latex_line(table_gindy_one[i])
            file.writelines(line_)
            
            
        file.writelines(['\n', '*'*15, ' gsindy all ', '*'*15, '\n'])
        line_ = get_latex_line(table_gindy_all[0], '.2e')
        file.writelines(line_)
        for i in range(1,3):
            line_ = get_latex_line(table_gindy_all[i])
            file.writelines(line_)
            
            
        file.writelines(['\n\n\n', '*'*15, ' Polynomial basis functions ', '*'*15, '\n'])
        file.writelines(['*'*15, ' sindy: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_sindy_poly.mean():.2e}, {mp_sindy_poly.mean():.2f}, {mr_sindy_poly.mean():.2f} ')
        file.writelines(['\n', '*'*15, ' gsindy_one: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_gsindy_one_poly.mean():.2e}, {mp_gsindy_one_poly.mean():.2f}, {mr_gsindy_one_poly.mean():.2f} ')
        file.writelines(['\n', '*'*15, ' gsindy_all: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_gsindy_all_poly.mean():.2e}, {mp_gsindy_all_poly.mean():.2f}, {mr_gsindy_all_poly.mean():.2f} ')
        
        file.writelines(['\n\n\n', '*'*15, ' Mixed basis functions ', '*'*15, '\n'])
        file.writelines(['*'*15, ' sindy: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_sindy_mix.mean():.2e}, {mp_sindy_mix.mean():.2f}, {mr_sindy_mix.mean():.2f} ')
        file.writelines(['\n', '*'*15, ' gsindy_one: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_gsindy_one_mix.mean():.2e}, {mp_gsindy_one_mix.mean():.2f}, {mr_gsindy_one_mix.mean():.2f} ')
        file.writelines(['\n', '*'*15, ' gsindy_all: mean of rmse, precision, recall ', '*'*15, '\n'])
        file.write(f'{rmse_gsindy_all_mix.mean():.2e}, {mp_gsindy_all_mix.mean():.2f}, {mr_gsindy_all_mix.mean():.2f} ')
            
        
        
        
        
        os.makedirs(os.path.join(f'{directory}/average'), exist_ok=True)
        Mean_poly = np.array([[rmse_sindy_poly.mean(), mp_sindy_poly.mean(), mr_sindy_poly.mean()],
                              [rmse_gsindy_one_poly.mean(), mp_gsindy_one_poly.mean(), mr_gsindy_one_poly.mean()],
                              [rmse_gsindy_all_poly.mean(), mp_gsindy_all_poly.mean(), mr_gsindy_all_poly.mean()]])
                    
        Mean_mix = np.array([[rmse_sindy_mix.mean(), mp_sindy_mix.mean(), mr_sindy_mix.mean()],
                              [rmse_gsindy_one_mix.mean(), mp_gsindy_one_mix.mean(), mr_gsindy_one_mix.mean()],
                              [rmse_gsindy_all_mix.mean(), mp_gsindy_all_mix.mean(), mr_gsindy_all_mix.mean()]])
        
        np.save(os.path.join(f'{directory}/average', f'mean_poly_{func_name}.npy'), Mean_poly)
        np.save(os.path.join(f'{directory}/average', f'mean_mix_{func_name}.npy'), Mean_mix)

        
        
        
    
    ########################################################
    ############# generate coefficients table###############
    ########################################################
    
    table_coeff_poly = np.zeros([num_feature, num_basis_poly, 4*num_traj+1], dtype=object)
    for idx_feature in range(num_feature):
        table_coeff_poly[idx_feature,:,0] = basis_function_name_poly[idx_feature]
        
        table_coeff_poly[idx_feature,:,1::4] = coeff_true_poly[:,idx_feature,:].T
        table_coeff_poly[idx_feature,:,2::4] = coeff_gsindy_all_poly[:,idx_feature,:].T
        table_coeff_poly[idx_feature,:,3::4] = coeff_gsindy_one_poly[:,idx_feature,:].T
        table_coeff_poly[idx_feature,:,4::4] = coeff_sindy_poly[:,idx_feature,:].T
    
    
    
    table_coeff_mix = np.zeros([num_feature, num_basis_mix, 4*num_traj+1], dtype=object)
    for idx_feature in range(num_feature):
        table_coeff_mix[idx_feature,:,0] = basis_function_name_mix[idx_feature]
        
        table_coeff_mix[idx_feature,:,1::4] = coeff_true_mix[:,idx_feature,:].T
        table_coeff_mix[idx_feature,:,2::4] = coeff_gsindy_all_mix[:,idx_feature,:].T
        table_coeff_mix[idx_feature,:,3::4] = coeff_gsindy_one_mix[:,idx_feature,:].T
        table_coeff_mix[idx_feature,:,4::4] = coeff_sindy_mix[:,idx_feature,:].T
    
    
    def get_latex_line(input_list, print_type='.4f'):
        line = ['&']
        line.append('$'+input_list[0]+'$')
        line.append('&')
        for ele in input_list[1:]:
            if ele==0:
                line.append(' ')
                line.append('&')
            else:
                line.append(f'{ele:{print_type}}'.rstrip('0'))
                line.append('&')
        line = line[:-1]
        line.append('\\\ \n')
        return line
    
    ### record all coefficients
    save_path = os.path.join(path_base, f'{func_name} coeff {suffix_poly} table.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file:
        file.write(f'the coefficients will keep if its absolute value >= {constants.precision} \n')
        file.writelines(['columns: traj1(poly, mix), traj2(poly, mix), traj3(poly, mix), traj4(poly, mix), traj5(poly, mix), traj6(poly, mix)', '\n'])
        file.writelines(['for each trajectory: true, gsindy all, gsindy one, sindy', '\n'])
        
        for j in range(num_feature):
            file.writelines(['\n', '*'*15, f' feature {j} ', '*'*15, '\n'])
            for i in range(table_coeff_poly.shape[1]):
                line_ = get_latex_line(table_coeff_poly[j,i], '.3f')
                file.writelines(line_)
            
    
    ### record all coefficients
    save_path = os.path.join(path_base, f'{func_name} coeff {suffix_mix} table.txt')
    open(save_path, 'w').close()
    
    with open(save_path, "a") as file:
        file.write(f'the coefficients will keep if its absolute value >= {constants.precision} \n')
        file.writelines(['columns: traj1(poly, mix), traj2(poly, mix), traj3(poly, mix), traj4(poly, mix), traj5(poly, mix), traj6(poly, mix)', '\n'])
        file.writelines(['for each trajectory: true, gsindy all, gsindy one, sindy', '\n'])
        
        for j in range(num_feature):
            file.writelines(['\n', '*'*15, f' feature {j} ', '*'*15, '\n'])
            for i in range(table_coeff_mix.shape[1]):
                line_ = get_latex_line(table_coeff_mix[j,i], '.3f')
                file.writelines(line_)
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    ### plot real coefficients vs predicted 
    os.makedirs(os.path.join(path_base, 'figures'), exist_ok=True)
    for i in range(num_traj):
        fig, ax = plt.subplots(3,num_feature,figsize=[10,8])
        fig.suptitle(f'{func_name} {suffix_poly} with trajectory {i}')
        basis_idx = np.arange(num_basis_poly)
        for j in range(num_feature):
            ax[0,j].scatter(basis_idx, coeff_true_poly[i,j,:], c='b', alpha=.3)
            ax[0,j].scatter(basis_idx, coeff_sindy_poly[i,j,:], c='r', alpha=.3)
            ax[0,j].set_title(f'{j}th feature: True vs SINDy')
            
            ax[1,j].scatter(basis_idx, coeff_true_poly[i,j,:], c='b', alpha=.3)
            ax[1,j].scatter(basis_idx, coeff_gsindy_one_poly[i,j,:], c='r', alpha=.3)
            ax[1,j].set_title(f'{j}th feature: True vs GS-SINDy one')
        
            ax[2,j].scatter(basis_idx, coeff_true_poly[i,j,:], c='b', alpha=.3)
            ax[2,j].scatter(basis_idx, coeff_gsindy_all_poly[i,j,:], c='r', alpha=.3)
            ax[2,j].set_title(f'{j}th feature: True vs GS-SINDy all')
            ax[2,j].set_xlabel('basis functions index')
            
        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(os.path.join(path_base, f'figures/{func_name} {suffix_poly} with trajectory {i}'), dpi=200)
        
        
        fig, ax = plt.subplots(3,num_feature,figsize=[10,8])
        fig.suptitle(f'{func_name} {suffix_mix} with trajectory {i}')
        basis_idx = np.arange(num_basis_mix)
        for j in range(num_feature):
            ax[0,j].scatter(basis_idx, coeff_true_mix[i,j,:], c='b', alpha=.3)
            ax[0,j].scatter(basis_idx, coeff_sindy_mix[i,j,:], c='r', alpha=.3)
            ax[0,j].set_title(f'{j}th feature: True vs SINDy')
            
            ax[1,j].scatter(basis_idx, coeff_true_mix[i,j,:], c='b', alpha=.3)
            ax[1,j].scatter(basis_idx, coeff_gsindy_one_mix[i,j,:], c='r', alpha=.3)
            ax[1,j].set_title(f'{j}th feature: True vs GS-SINDy one')
        
            ax[2,j].scatter(basis_idx, coeff_true_mix[i,j,:], c='b', alpha=.3)
            ax[2,j].scatter(basis_idx, coeff_gsindy_all_mix[i,j,:], c='r', alpha=.3)
            ax[2,j].set_title(f'{j}th feature: True vs GS-SINDy all')
            ax[2,j].set_xlabel('basis functions index')
            
        fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        fig.savefig(os.path.join(path_base, f'figures/{func_name} {suffix_mix} with trajectory {i}'), dpi=100)
        
        
        