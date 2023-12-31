#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:55:57 2023

@author: dliu
"""


import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func5, monomial_poly, monomial_poly_name, monomial_all, monomial_all_name
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


dt = .05   
t = np.arange(0,20,dt)
num = 5

################## 1 variable ####################
x0 = [[1.5, -.5]]
a = [(-.5,.2)]
num_split = 3 

func = func5
real0 = "x'=5*(x - y- a*x^3)"
real1 = "y'=b*x"
real_list = [real0, real1]
##################################################


if __name__ == "__main__":
    fit_sindy_mult_2d(func, x0, t, a, num, num_split, real_list, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)
    