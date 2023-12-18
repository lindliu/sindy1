#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:55:57 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, func8, func9,\
                func12_, func3_, func4_, func3__, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name, \
                monomial_all, monomial_all_name
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
num = None

dt = .1     
t = np.arange(0,20,dt)

# ################## 2 variable ####################
x0 = [[.4, 1]]
a = [(1, 3)]
num_split = 3 

func = func8
real0 = "x'=a-4x+x^2y"
real1 = "y'=bx-x^2y"
####################################################



if __name__ == "__main__":
    fit_sindy_mult_2d(func, x0, t, a, num, num_split, real0, real1, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)
    