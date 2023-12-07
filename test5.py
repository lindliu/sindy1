#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:49:29 2023

@author: do0236li
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import ridge_regression, LinearRegression, SGDRegressor
from func import func1, func2, func3, func4, func5, func6, func7, \
    get_sol_deriv, monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name

from func import func12_, func3_, func4_
deriv_spline = True#False#
threshold_tol = 1e-3

# alpha = .05
# dt = .1   ## 0,3
# t = np.arange(0,1.5,dt)
# x0 = [.5, 1]
# a = [(.16, .25), (.3, .4), (.3, .5)]
# func = func12_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + b*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_similarity = 1e-3
# threshold_tol = 1e-2

alpha = .05
dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,12,dt)
x0 = [.5, 1]
a = [(.2, -.6), (.4, -.8), (.6, -1)]
func = func3_
monomial = monomial_poly
monomial_name = monomial_poly_name
real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y - y^3"
threshold_sindy=1e-2
threshold_similarity = 1e-2

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,6,dt)
# x0 = [4, 1]
# a = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5)]
# func = func4_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x + b*xy"
# real1 = "y'=b*y + a*xy"
# threshold_sindy=1e-2
# threshold_similarity = 1e-3


################### 1 variable ####################
# alpha = .05
# dt = .05   ## 0,3
# t = np.arange(0,2.5,dt)
# x0 = [.2, 1]
# a = [(.12,), (.16,), (.2,)]
# func = func1
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + x^2"
# real1 = "y'=-y"
# threshold_sindy=5e-2
# threshold_similarity = 1e-3

# alpha = .05
# dt = .1
# t = np.arange(0,2,dt)
# x0 = [.5, 1]
# a = [(.25,), (.3,), (.5,)]
# # a = [(.1,), (.25,), (.3,), (.5,)]
# func = func2
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=.2 + a*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_similarity = 1e-3

# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,12,dt)
# x0 = [.5, 1]
# a = [(.2,), (.4,), (.6,)]
# func = func3
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"    
# threshold_sindy=1e-2
# threshold_similarity = 1e-2

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [4, 1]
# a = [(.7,), (1,)]#,(.5,),(.6,)]
# func = func4
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"    
# threshold_sindy=1e-2
# threshold_similarity = 1e-3

# alpha = .05
# dt = .1
# t = np.arange(0,3.3,dt)
# x0 = [np.pi-.1, 0]
# a = [(-.25,), (-.35,)]
# func = func6
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=a*y-5sin(x)"
# threshold_sindy=1e-2
# threshold_similarity = 1e-3

# alpha = .05
# dt = .1
# t = np.arange(0,6,dt)
# x0 = [np.pi-.1, 0]
# # a = [-5, -6]
# a = [(-.15,), (-1,), (-2,), (-5,)]
# func = func7
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=-0.25*y+a*sin(x)"
# threshold_sindy=1e-2
# threshold_similarity = 1e-3
