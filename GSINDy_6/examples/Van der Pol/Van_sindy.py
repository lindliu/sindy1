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

dt = .1    
t = np.arange(0,20,dt)
num = 6

# ################## 2 variable ####################
x0 = [[1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5]]
a = [(-.5, .2), (-.3, .2), (-.4, .3),(-.2, .3), (-.35, .4), (-.6, .4)]

func = func5
real0 = "x'=5*(x - y- a*x^3)"
real1 = "y'=b*x"    
real_list = [real0, real1]
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

        