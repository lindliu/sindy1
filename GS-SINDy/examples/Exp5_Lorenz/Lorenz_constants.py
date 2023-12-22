#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func_Lorenz
from utils import basis_functions_mix0_3d, basis_functions_mix1_3d, basis_functions_mix2_3d, \
                basis_functions_name_mix0_3d, basis_functions_name_mix1_3d, basis_functions_name_mix2_3d, \
                basis_functions_3d, basis_functions_name_3d,\
                basis_functions_poly_3d, basis_functions_poly_name_3d

########## hyper parameters ###########
ensemble = False
precision = 1e-4
deriv_spline = True#False#
alpha = .05
threshold_sindy_list = [1e-2, 5e-2, 1e-1, 5e-1, 1e0]
threshold_group_list = [1e-3, 1e-2]
threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]

########## function variable ###########
dt = .05
t = np.arange(0,10,dt)
x0_list = [[-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27]]
a_list = [(-10,10,28,-2.67), (-9,9,30,-2), (-8.5,8.5,28,-2.67), (-10,10,27,-2.67), (-10,10,27,-3), (-10,10,29,-2)]

func = func_Lorenz
real0 = "x'=by+ax"
real1 = "y'=x(c-z)-y" 
real2 = "z'=xy + dz" 
real_list = [real0, real1, real2]


########## basis functions and optimizer ###########
basis_type = 'poly' ##'mix_same' ## 'mix_diff' ##

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix_diff':
            basis_functions_list = [basis_functions_mix0_3d, basis_functions_mix1_3d, basis_functions_mix2_3d]          ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_name_mix0_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_mix1_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_mix2_3d])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
        if basis_type == 'mix_same':
            basis_functions_list = [basis_functions_3d, basis_functions_3d, basis_functions_3d]          ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_name_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_3d])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
        
        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_3d, basis_functions_poly_3d, basis_functions_poly_3d]          ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_poly_name_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_poly_name_3d]), \
                                         np.array([f(1,1,1) for f in basis_functions_poly_name_3d])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
    else:
        if basis_type == 'mix_diff':
            basis_functions_list = [basis_functions_mix0_3d, basis_functions_mix1_3d, basis_functions_mix2_3d]          ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_mix0_3d, basis_functions_name_mix1_3d, basis_functions_name_mix2_3d]    ### corresponding names of the basis functions
            opt = 'Manually' ## only 'Manually' works

        if basis_type == 'mix_same':
            basis_functions_list = [basis_functions_3d, basis_functions_3d, basis_functions_3d]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_3d, basis_functions_name_3d, basis_functions_name_3d]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_3d, basis_functions_poly_3d, basis_functions_poly_3d]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_poly_name_3d, basis_functions_poly_name_3d, basis_functions_poly_name_3d]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
    basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}
    
    return basis, opt

