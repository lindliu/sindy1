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
from utils import func1, func2, func3, func4, func5, func6, func7, func8, func9, \
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

dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,10,dt)

################## 3 variable ####################
x0 = [[.4, 1]]
a = [(.2,-.6,-.5)]
num_split = 3 
    
func = func3__
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
##################################################

# ################## 2 variable ####################
# x0 = [[.4, 1]]
# a = [(.2,-.6)]
# num_split = 3 

# func = func3_
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
####################################################


# ################## 1 variable ####################
# x0 = [[.4, 1]]
# a = [(.2,)]
# num_split = 3 
        
# func = func3
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"   
####################################################



if __name__ == "__main__":
    fit_sindy_mult_2d(func, x0, t, a, num, num_split, real0, real1, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)
    