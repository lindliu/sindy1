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
sys.path.insert(1, '../GSINDy')

import Lotka_constants
from Lotka_constants import get_basis_functions

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})



basis_type = 'poly' ##'mix'##
basis, opt = get_basis_functions(basis_type, GSINDY=True)
suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'

basis_function_name = basis['names']
import os
from glob import glob

path_Exp1 = os.path.join(os.getcwd(), 'Exp1_Lotka_Volterra/results/')
path_Exp2 = os.path.join(os.getcwd(), 'Exp2_Modified_Lotka_Volterra/results/')
path_Exp3 = os.path.join(os.getcwd(), 'Exp3_Brusselator/results/')
path_Exp4 = os.path.join(os.getcwd(), 'Exp4_Van_der_Pol/results/')
path_Exp5 = os.path.join(os.getcwd(), 'Exp5_Lorenz/results/')

path_base = path_Exp1


x0_list = Lotka_constants.x0_list
a_list = Lotka_constants.a_list

    
#### results from gsindy all together
path_gsindy_all = glob(os.path.join(path_base, f'coeff/gsindy_all_{suffix}*.npy'))
coeff_gsindy_all = np.load(path_gsindy_all[0])
num_traj, num_feature, num_basis = coeff_gsindy_all.shape


#### results from gsindy one by one
coeff_gsindy_one = np.zeros([num_traj, num_feature, num_basis])
for i in range(num_traj):
    path_ = os.path.join(path_base, f'coeff/gsindy_{suffix}_{i}.npy')
    coeff_gsindy_one[i,:,:] = np.load(path_)
        
        
#### results from sindy
coeff_sindy_all = np.zeros([num_traj, num_feature, num_basis])
for i in range(num_traj):
    path_ = os.path.join(path_base, f'coeff/sindy_{suffix}_{i}.npy')
    coeff_sindy_all[i,:,:] = np.load(path_)
    
    
    