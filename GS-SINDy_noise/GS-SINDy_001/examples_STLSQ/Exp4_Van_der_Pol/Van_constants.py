#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func_Van_der_Pol
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5, \
        basis_functions_mix1_, basis_functions_name_mix1_
np.random.seed(42)

########## hyper parameters ###########
ensemble = True
precision = 1e-4
deriv_spline = True#False#
alpha = .05
threshold_sindy_list = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
threshold_group_list = [1e-3, 1e-2]
threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]
             
noise_var = 0.01
       
########## function variable ###########
dt = .05
t = np.arange(0,20,dt)
x0_list = [[1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5]]
a_list = [(-.5, .2), (-.3, .2), (-.4, .3),(-.2, .3), (-.35, .4), (-.6, .4)]

func = func_Van_der_Pol
real0 = "x'=5*(x - y + a*x^3)"
real1 = "y'=b*x"    
real_list = [real0, real1]


########## basis functions and optimizer ###########
basis_type = 'mix' ## 'poly' ##

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_mix0, basis_functions_mix1_]                           ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_mix0]), \
                                         np.array([f(1,1) for f in basis_functions_name_mix1_])]     ### corresponding names of the basis functions
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

