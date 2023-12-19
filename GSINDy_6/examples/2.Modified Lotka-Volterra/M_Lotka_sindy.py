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
x0 = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
a = [(.2, -.6, -.5), (.4, -.8, -.7), (.2, -.6, -1), (.4, -.8, -1), (.4, -1, -1), (.6, -1, -1)]
    
func = func3__
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
real_list = [real0, real1]
####################################################

# ################## 2 variable ####################
# x0 = [[.4, 1], [.4, 1], [.4, 1]]
# a = [(.2, -.6), (.4, -.8), (.6, -1)]

# func = func3_
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
# real_list = [real0, real1]
###################################################

# ################## 1 variable ####################
# x0 = [[.4, 1], [.4, 1], [.4, 1]]
# a = [(.2,), (.4,), (.6,)]
        
# func = func3
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"   
# real_list = [real0, real1]
####################################################


if __name__ == "__main__":
    model_best_list = fit_sindy_2d(func, x0, t, a, real_list, monomial, monomial_name, \
                                   precision, alpha, opt, deriv_spline, ensemble)

    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/coeff', exist_ok=True)
    save_path = 'results/sindy_all.txt'
    open(save_path, 'w').close()
    
    for idx in range(len(model_best_list)):
        coef = model_best_list[idx].coefficients()
        np.save(f'results/coeff/sindy_{idx}.npy', coef)

        mask0 = np.abs(coef[0]) > precision
        mask1 = np.abs(coef[1]) > precision
        with open(save_path, "a") as file2:
            file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
            file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
            file2.write(f'basis of feature 0: {monomial_name[mask0]} \n')
            file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
            file2.write(f'basis of feature 1: {monomial_name[mask1]} \n\n')

        