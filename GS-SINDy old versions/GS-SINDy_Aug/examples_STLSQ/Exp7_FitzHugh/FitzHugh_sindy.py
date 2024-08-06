#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:57:56 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../tools')

import os
import numpy as np
import matplotlib.pyplot as plt
from FitzHugh_constants import get_basis_functions
import FitzHugh_constants as constants
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


########## hyper parameters ###########
ensemble = constants.ensemble
precision = constants.precision
deriv_spline = constants.deriv_spline
alpha = constants.alpha
threshold_sindy_list = constants.threshold_sindy_list

########## function variable ###########
t = constants.t
x0_list = constants.x0_list
a_list = constants.a_list

func = constants.func
real_list = constants.real_list

########## basis functions and optimizer ###########
basis_type = constants.basis_type
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=False)


path_base = os.path.join(os.getcwd(), 'results')
suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'
suffix = f'{suffix}_e' if ensemble else suffix
if __name__ == "__main__":
    
    from sindy_2d_train import sindy_2d_train
    sindy_2d_train(func, t, x0_list, a_list, real_list, suffix, basis, precision, \
                   alpha, opt, deriv_spline, ensemble, path_base, threshold_sindy_list)