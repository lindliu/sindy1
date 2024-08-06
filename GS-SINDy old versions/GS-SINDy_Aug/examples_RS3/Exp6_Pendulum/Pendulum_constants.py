#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:28:19 2023

@author: dliu
"""

import numpy as np
from utils import func_Pendulum
from utils import basis_functions_trig, basis_functions_name_trig

########## hyper parameters ###########
ensemble = False
precision = 1e-4  ### if precision is not bigger than threshold_sindy, does not influence results by Maunually optimizer
deriv_spline = True#False#
alpha = .05
threshold_sindy_list = [5e-3, 1e-2, 5e-2, 1e-1]
threshold_group_list = [1e-3, 1e-2]
threshold_similarity_list = [[1e-3, 1e-2], [1e-1]]   
# first list is for multi series. 
# if only one series, we will split it into several pieces, 
# but we know it comes from the same series, 
# so the coefficient should be the same, so threshold should be big


########## function variable ###########
dt = .1
t = np.arange(0,5,dt)
x0_list = [[np.pi-.1, 0.01], [np.pi-.1, 0.01], [np.pi-.1, 0.01], [np.pi-.1, 0.01], [np.pi-.1, 0.01], [np.pi-.1, 0.01]]
a_list = [(-.25,-5), (-.2,-4.5), (-.25,-5.5), (-.23,-4), (-.3,-4), (-.2,-4.8)]

func = func_Pendulum
real0 = "x'=1*y"
real1 = "y'=a*y + b*sin(x)" 
real_list = [real0, real1]


########## basis functions and optimizer ###########
basis_type = 'mix'

def get_basis_functions(basis_type, GSINDY=True):
    if GSINDY:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_trig, basis_functions_trig]              ### basis functions for each feature
            basis_functions_name_list = [np.array([f(1,1) for f in basis_functions_name_trig]), \
                                         np.array([f(1,1) for f in basis_functions_name_trig])]     ### corresponding names of the basis functions
            opt = 'SR3'  ##['Manually', 'SQTL', 'LASSO', 'SR3']
            
    else:
        if basis_type == 'mix':
            basis_functions_list = [basis_functions_trig, basis_functions_trig]                ### basis functions for each feature
            basis_functions_name_list = [basis_functions_name_trig, basis_functions_name_trig]    ### corresponding names of the basis functions
            opt = 'SR3' ##['Manually', 'SQTL', 'LASSO', 'SR3']

    basis = {'functions': basis_functions_list, 'names': basis_functions_name_list}
    
    return basis, opt

