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
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from utils import monomial_poly, monomial_poly_name, monomial_all, monomial_all_name


#### results from gsindy all together
path_gsindy_all = glob(os.path.join('results/coeff/gsindy_all*.npy'))
coeff_gsindy_all = np.load(path_gsindy_all[0])
num_traj, num_feature, num_basis = coeff_gsindy_all.shape

#### results from gsindy one by one
max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in glob(os.path.join('results/coeff/gsindy_one_*.npy'))])
n_split = max_split-2+1   ##4
coeff_gsindy_one = np.zeros([n_split, num_traj, num_feature, num_basis])
for j in range(num_traj):
    for k in range(n_split):
        path_ = glob(os.path.join(f'results/coeff/gsindy_one_{k+2}_{j}.npy'))[0]
        coeff_gsindy_one[k,j,:,:] = np.load(path_)

#### results from sindy
path_sindy_all = glob(os.path.join('results/coeff/sindy*.npy'))
coeff_sindy_all = np.zeros([num_traj, num_feature, num_basis])
for i, path_ in enumerate(path_sindy_all):
    coeff_sindy_all[i,:,:] = np.load(path_)


##### true results
coeff_true_ = np.array([(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5), (1.2,-1.5), (1.3,-1.)])
coeff_true = np.zeros([num_traj, num_feature, num_basis])
coeff_true[:,0,[1,4]] = coeff_true_[:,[0,1]]
coeff_true[:,1,[2,4]] = coeff_true_[:,[1,0]]


basis_idx = np.arange(num_basis)
monomial_name = monomial_poly_name
assert monomial_name.shape[0]==num_basis, 'number of basis functions does not match'



#####
fig, ax = plt.subplots(1,2,figsize=[8,5])
fig.suptitle("True(red) vs GSINDy(blue) all", fontsize=16)
ax = ax.flatten()
for i in range(num_feature):
    for j in range(num_traj):
        ax[i].scatter(basis_idx, coeff_true[j,i,:], c='b', alpha=.5)
        ax[i].scatter(basis_idx, coeff_gsindy_all[j,i,:], c='r', alpha=.2)
    ax[i].set_title(f'feature {i}')
fig.tight_layout()


##### 
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
    












