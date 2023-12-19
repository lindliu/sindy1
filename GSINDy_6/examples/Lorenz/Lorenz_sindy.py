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
from utils import func9, monomial_poly, monomial_lorenz, monomial_poly_name, monomial_lorenz_name, monomial_all, monomial_all_name
from train_sindy_3d import fit_sindy_3d

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

#################### 3 variable ####################
x0 = [[-8, 8, 27], [-8, 8, 27], [-8, 8, 27], [-8, 8, 27]]
a = [(10,28,8/3), (8,25,3), (9,30,2), (10,26,2)]

func = func9
real0 = "x'=a(y-x)"
real1 = "y'=x(b-z)-y" 
real2 = "z'=xy - cz" 
real_list = [real0, real1, real2]
####################################################


if __name__ == "__main__":
    model_best_list = fit_sindy_3d(func, x0, t, a, real_list, monomial, monomial_name, \
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
        mask2 = np.abs(coef[2]) > precision
        with open(save_path, "a") as file2:
            file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
            file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
            file2.write(f'basis of feature 0: {monomial_name[mask0]} \n')
            file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
            file2.write(f'basis of feature 1: {monomial_name[mask1]} \n')
            file2.write(f'coef of feature 2: {coef[2,:][mask2]} \n')
            file2.write(f'basis of feature 2: {monomial_name[mask2]} \n\n')
        