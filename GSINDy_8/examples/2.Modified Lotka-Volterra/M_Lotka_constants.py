#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func3__
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5

########## hyper parameters ###########
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05


########## function variable ###########
dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,10,dt)
x0_list = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
a_list = [(.2, -.6, -.5), (.4, -.8, -.7), (.2, -.6, -1), (.4, -.8, -1), (.4, -1, -1), (.6, -1, -1)]

func = func3__
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
real_list = [real0, real1]


########## basis functions and optimizer ###########
basis_type = 'poly' ##'mix'

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_mix0]), \
                                         np.array([f(1,1) for f in basis_functions_name_mix1])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5]              ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_poly_5]), \
                                         np.array([f(1,1) for f in basis_functions_name_poly_5])]     ### corresponding names of the basis functions
            opt = 'SQTL'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
    else:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1]                           ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_mix0, basis_functions_name_mix1]    ### corresponding names of the basis functions
            opt = 'Manually' ## only 'Manually' works

        if basis_type == 'poly':
            basis_functions_list = [basis_functions_poly_5, basis_functions_poly_5]              ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_poly_5, basis_functions_name_poly_5]    ### corresponding names of the basis functions
            opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']

    basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}
    
    return basis, opt

