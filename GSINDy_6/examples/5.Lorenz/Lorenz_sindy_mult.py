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
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, func9, \
                monomial_poly, monomial_trig, monomial_lorenz, monomial_lorenz_name, \
                monomial_all, monomial_all_name
from train_sindy_mult_3d import fit_sindy_mult_3d

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_lorenz
monomial_name = monomial_lorenz_name

dt = .05
t = np.arange(0,10,dt)
num = 1

#################### 3 variable ####################
x0 = [[-8, 8, 27]]
a = [(10,28,-2.67)]
num_split = 2

func = func9
real0 = "x'=a(y-x)"
real1 = "y'=x(b-z)-y" 
real2 = "z'=xy + cz" 
real_list = [real0, real1, real2]
# ##################################################


if __name__ == "__main__":
    fit_sindy_mult_3d(func, x0, t, a, num, num_split, real_list, monomial, monomial_name, \
           precision, alpha, opt, deriv_spline, ensemble)