#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:41:15 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import os
import numpy as np
import matplotlib.pyplot as plt
from Lorenz_constants import get_basis_functions
import Lorenz_constants as constants
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


path_base = os.path.join(os.getcwd(), 'results')

if __name__ == "__main__":
    
    from gsindy_3d_train import gsindy_3d_train
    gsindy_3d_train(func, t, x0_list, a_list, real_list, basis_type, basis, precision, alpha, opt, deriv_spline, ensemble, path_base)
    