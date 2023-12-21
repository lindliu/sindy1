#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:15:54 2023

@author: do0236li
"""
import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import os
import numpy as np
import matplotlib.pyplot as plt
from M_Lotka_constants import get_basis_functions
import M_Lotka_constants as constants
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


path_base = os.path.join(os.getcwd(), 'results')

if __name__ == "__main__":
    from glob import glob
    
    #### results from gsindy all together
    path_gsindy_all = glob(os.path.join(path_base, f'coeff/gsindy_all_{basis_type}*.npy'))
    coeff_gsindy_all = np.load(path_gsindy_all[0])
    num_traj, num_feature, num_basis = coeff_gsindy_all.shape
    
    
    #### results from gsindy one by one
    max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in \
                     glob(os.path.join(path_base, f'coeff/gsindy_one_{basis_type}*.npy'))])
    n_split = max_split-2+1   ##4
    coeff_gsindy_one = np.zeros([n_split, num_traj, num_feature, num_basis])
    for j in range(num_traj):
        for k in range(n_split):
            path_ = glob(os.path.join(path_base, f'coeff/gsindy_one_{basis_type}_{k+2}_{j}.npy'))[0]
            coeff_gsindy_one[k,j,:,:] = np.load(path_)
            
            
    #### results from sindy
    path_sindy_all = glob(os.path.join(path_base, f'coeff/sindy_{basis_type}*.npy'))
    coeff_sindy_all = np.zeros([num_traj, num_feature, num_basis])
    for i, path_ in enumerate(path_sindy_all):
        coeff_sindy_all[i,:,:] = np.load(path_)
    
    
    ##### true results
    if basis_type=='poly':
        coeff_true_ = np.array([(.2, -.6, -.5), (.4, -.8, -.7), (.2, -.6, -1), (.4, -.8, -1), (.4, -1, -1), (.6, -1, -1)])
        coeff_true = np.zeros([num_traj, num_feature, num_basis])
        coeff_true[:,0,[2,3,6]] = coeff_true_[:,[1,0,2]]
        coeff_true[:,1,[2,7,9]] = coeff_true_[:,[0,1,2]]
    
        coeff_true[:,0,[8]] = [-1]
        coeff_true[:,1,[1]] = [1]
    
        # real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
        # real1 = "y'=x + a*y + b*x^2y + c*y^3"
    
    
    basis_idx = np.arange(num_basis)
    ##### true coefficients vs gsindy all coefficients
    fig, ax = plt.subplots(1,2,figsize=[8,5])
    fig.suptitle("True(red) vs GSINDy(blue) all", fontsize=16)
    ax = ax.flatten()
    for i in range(num_feature):
        for j in range(num_traj):
            ax[i].scatter(basis_idx, coeff_true[j,i,:], c='b', alpha=.5)
            ax[i].scatter(basis_idx, coeff_gsindy_all[j,i,:], c='r', alpha=.2)
        ax[i].set_title(f'feature {i}')
    fig.tight_layout()
    
    
    ##### true coefficients vs sindy coefficients
    fig, ax = plt.subplots(num_traj,2,figsize=[10,15])
    fig.suptitle("True(red) vs GSINDy(blue) one vs SINDy(green)", fontsize=16)
    ax = ax.flatten()
    for j in range(num_traj):
        for i in range(num_feature):
            ax[2*j+i].scatter(basis_idx, coeff_true[j,i,:], c='b', alpha=.5)
            ax[2*j+i].scatter(basis_idx, coeff_sindy_all[j,i,:], c='g', alpha=.5)
            for k in range(n_split):
                ax[2*j+i].scatter(basis_idx, coeff_gsindy_one[k,j,i,:], c='r', alpha=.2)
        
        ax[2*j+i].set_title(f'feature {i}')
    fig.tight_layout()
        
    
    
    
    
    
    
    
    
    
    
    
    

