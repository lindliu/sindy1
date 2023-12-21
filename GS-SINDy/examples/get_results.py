#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:29:29 2023

@author: dliu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, './Exp1_Lotka_Volterra')
import Lotka_constants
from Lotka_constants import get_basis_functions

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
suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'

import os
from glob import glob

path_Exp1 = os.path.join(os.getcwd(), 'Exp1_Lotka_Volterra/results/coeff')
path_Exp2 = os.path.join(os.getcwd(), 'Exp2_Modified_Lotka_Volterra/results/coeff')
path_Exp3 = os.path.join(os.getcwd(), 'Exp3_Brusselator/results/coeff')
path_Exp4 = os.path.join(os.getcwd(), 'Exp4_Van_der_Pol/results/coeff')
path_Exp5 = os.path.join(os.getcwd(), 'Exp5_Lorenz/results/coeff')

path_base = path_Exp1



    
#### results from gsindy all together
path_gsindy_all = glob(os.path.join(path_base, f'coeff/gsindy_all_{suffix}*.npy'))
coeff_gsindy_all = np.load(path_gsindy_all[0])
num_traj, num_feature, num_basis = coeff_gsindy_all.shape


#### results from gsindy one by one
max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in \
                 glob(os.path.join(path_base, f'coeff/gsindy_one_{suffix}*.npy'))])
n_split = max_split-2+1   ##4
coeff_gsindy_one = np.zeros([n_split, num_traj, num_feature, num_basis])
for j in range(num_traj):
    for k in range(n_split):
        path_ = glob(os.path.join(path_base, f'coeff/gsindy_one_{suffix}_{k+2}_{j}.npy'))[0]
        coeff_gsindy_one[k,j,:,:] = np.load(path_)
        
        
#### results from sindy
path_sindy_all = glob(os.path.join(path_base, f'coeff/sindy_{suffix}*.npy'))
coeff_sindy_all = np.zeros([num_traj, num_feature, num_basis])
for i, path_ in enumerate(path_sindy_all):
    coeff_sindy_all[i,:,:] = np.load(path_)
    
    
    