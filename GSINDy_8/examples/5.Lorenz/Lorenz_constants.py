#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func9
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5, basis_functions_Lorenz, basis_functions_name_Lorenz

########## hyper parameters ###########
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05


########## function variable ###########
dt = .05
t = np.arange(0,10,dt)
x0_list = [[-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27]]
a_list = [(-10,10,28,-2.67), (-9,9,30,-2), (-8.5,8.5,28,-2.67), (-10,10,27,-2.67), (-10,10,27,-3), (-10,10,29,-2)]

func = func9
real0 = "x'=dy+ax)"
real1 = "y'=x(b-z)-y" 
real2 = "z'=xy + cz" 
real_list = [real0, real1, real2]


########## basis functions and optimizer ###########
basis_type = 'mix_Lorenz' ##'poly_Lorenz' ###'mix_diff_Lorenz'

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix_diff_Lorenz':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_name_mix0]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_mix1])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
        if basis_type == 'poly_Lorenz':
            basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5, basis_functions_poly_5]          ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_name_poly_5]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_poly_5]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_poly_5])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
        if basis_type == 'mix_Lorenz':
            basis_functions_list = [basis_functions_Lorenz, basis_functions_Lorenz, basis_functions_Lorenz]              ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1,1) for f in basis_functions_name_Lorenz]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_Lorenz]), \
                                         np.array([f(1,1,1) for f in basis_functions_name_Lorenz])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
    else:
        if basis_type == 'mix_diff_Lorenz':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_mix0, basis_functions_name_mix1]    ### corresponding names of the basis functions
            opt = 'Manually' ## only 'Manually' works

        if basis_type == 'poly_Lorenz':
            basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5, basis_functions_poly_5]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_poly_5, basis_functions_name_poly_5, basis_functions_name_poly_5]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
        if basis_type == 'mix_Lorenz':
            basis_functions_list = [basis_functions_Lorenz, basis_functions_Lorenz, basis_functions_Lorenz]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_Lorenz, basis_functions_name_Lorenz, basis_functions_name_Lorenz]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
    basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}
    
    return basis, opt

