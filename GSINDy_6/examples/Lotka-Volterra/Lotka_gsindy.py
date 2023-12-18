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

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, func8, func9, \
                func12_, func3_, func4_, func3__, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name, \
                monomial_all, monomial_all_name
from train_gsindy_2d import fit_gsindy_2d

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name
num_split = None

dt = .1 
t = np.arange(0,8,dt)
num = 6

#################### 2 variable ####################
x0_list = [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
a_list = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5), (1.2,-1.5), (1.3,-1.)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]

func = func4_
real0 = "x'=a*x + b*xy"
real1 = "y'=b*y + a*xy" 
####################################################

# #################### 1 variable ####################
# x0_list = [[3, 1], [3, 1]]
# a_list = [(.7,), (1,)]#,(.5,),(.6,)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 2
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func4
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"
# ####################################################



if __name__ == "__main__":
    fit_gsindy_2d(func, x0, t, a, num, num_split, real0, real1, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)