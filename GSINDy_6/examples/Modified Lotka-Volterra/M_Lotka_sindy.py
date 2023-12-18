#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:43:45 2023

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
from train_sindy_2d import fit_sindy_2d


opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name

dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,10,dt)

################## 3 variable ####################
x0 = [[.4, 1], [.4, 1], [.4, 1]]
a = [(.2, -.6, -.5), (.4, -.8, -.7), (.6, -1, -1)]
    
func = func3__
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
####################################################

# ################## 2 variable ####################
# x0 = [[.4, 1], [.4, 1], [.4, 1]]
# a = [(.2, -.6), (.4, -.8), (.6, -1)]

# func = func3_
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
###################################################

# ################## 1 variable ####################
# x0 = [[.4, 1], [.4, 1], [.4, 1]]
# a = [(.2,), (.4,), (.6,)]
        
# func = func3
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"   
####################################################


if __name__ == "__main__":
    fit_sindy_2d(func, x0, t, a, real0, real1, monomial, monomial_name, \
               precision, alpha, opt, deriv_spline, ensemble)
