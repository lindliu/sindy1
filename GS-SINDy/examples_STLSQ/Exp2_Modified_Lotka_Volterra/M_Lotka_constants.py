#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func_M_Lotka_Voltera
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5, basis_functions_poly_4, basis_functions_name_poly_4

########## hyper parameters ###########
ensemble = False
precision = 1e-4
deriv_spline = True#False#
alpha = .05
threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
threshold_group_list = [1e-3, 1e-2]
threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]

########## function variable ###########
dt = .05      
t = np.arange(0,10,dt)
x0_list = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
a_list = [(1, 1, -1), (.2, .8, -.7), (0, 1, -1), (-.1, 1, -1), (1, .7, -1), (.6, 1, -.8)]

func = func_M_Lotka_Voltera
real0 = "x' = a*x - b*y + c*x(x^2 + y^2)"
real1 = "y' = b*x + a*y + c*y(x^2 + y^2)"
real_list = [real0, real1]


########## basis functions and optimizer ###########
basis_type = 'mix'##'poly' ##

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_mix0]), \
                                         np.array([f(1,1) for f in basis_functions_name_mix1])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_4, basis_functions_poly_4]              ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_poly_4]), \
                                         np.array([f(1,1) for f in basis_functions_name_poly_4])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
    else:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_mix0, basis_functions_name_mix1]    ### corresponding names of the basis functions
            opt = 'Manually' ## only 'Manually' works

        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_4, basis_functions_poly_4]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_poly_4, basis_functions_name_poly_4]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']

    basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}
    
    return basis, opt

