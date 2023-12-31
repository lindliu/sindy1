#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:05:01 2023

@author: dliu
"""


import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func4, func4_, monomial_poly, monomial_poly_name, monomial_all, monomial_all_name
from train_sindy_mult_2d import fit_sindy_mult_2d


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name


dt = .1 
t = np.arange(0,8,dt)
num = 1

#################### 2 variable ####################
x0 = [[3, 1]]
a = [(.7,-.8)]
num_split = 2

func = func4_
real0 = "x'=a*x + b*xy"
real1 = "y'=b*y + a*xy" 
real_list = [real0, real1]
##################################################

# #################### 1 variable ####################
# x0 = [[3, 1]]
# a = [(.7,)]
# num_split = 2

# func = func4
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"
# real_list = [real0, real1]
# ##################################################

if __name__ == "__main__":
    fit_sindy_mult_2d(func, x0, t, a, num, num_split, real_list, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)
    