#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:57:46 2024

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../tools')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
import os
from pysindy.utils.odes import lorenz
# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from Van_constants import get_basis_functions
import Van_constants as constants
from sindy_2d import model_selection_coeff_2d
from utils import ode_solver, get_deriv

np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})


# # Seed the random number generators for reproducibility
# np.random.seed(100)

########## hyper parameters ###########
ensemble = constants.ensemble
precision = constants.precision
deriv_spline = constants.deriv_spline
alpha = constants.alpha
threshold_sindy_list = constants.threshold_sindy_list

########## function variable ###########
dt = constants.dt
t = constants.t
x0_list = constants.x0_list
a_list = constants.a_list

func = constants.func
real_list = constants.real_list

########## basis functions and optimizer ###########
basis_type = constants.basis_type
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=False)


path_base = os.path.join(os.getcwd(), 'results')
suffix = f'{basis_type}_SQTL' if opt=='Manually' else f'{basis_type}_{opt}'

threshold_sindy = 5e-2

# Generate measurement data
i=3

x0 = x0_list[i]
a = a_list[i]
sol_, t_ = ode_solver(func, x0, t, a)
_, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)

t_train = t
u_train = sol_
u_dot = sol_deriv_


print(a)
print(real_list[0])
print(real_list[1])


### pysindy settings
import pysindy_ as ps
from pysindy_.feature_library import GeneralizedLibrary, CustomLibrary

basis_functions_list = basis['functions']
basis_functions_name_list = basis['names']

assert (basis_functions_list[0]==basis_functions_list[1]).all(), 'pysindy does not support different features with different basis functions'

basis_functions = basis_functions_list[0]
basis_functions_name = basis_functions_name_list[0]

lib_custom = CustomLibrary(library_functions=basis_functions, function_names=basis_functions_name)
lib_generalized = GeneralizedLibrary([lib_custom])

# optimizer = ps.SR3(threshold=threshold_sindy, nu=.1)
optimizer = ps.SR3(
    threshold=threshold_sindy, thresholder="l1", max_iter=1000, normalize_columns=True, tol=1e-1
)
# Instantiate and fit the SINDy model with u_dot
model = ps.SINDy(feature_names=["x", "y", "z"], feature_library=lib_generalized)
model.fit(u_train, t=t_train, x_dot=u_dot, quiet=True)
model.print()




model_best_list = []
for idx in range(6):

    x0 = x0_list[idx]
    a = a_list[idx]
    sol_, t_ = ode_solver(func, x0, t, a)
    _, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)
    
    t_train = t
    u_train = sol_
    u_dot = sol_deriv_
    
    # Define weak form ODE library
    # defaults to derivative_order = 0 if not specified,
    # and if spatial_grid is not specified, defaults to None,
    # which allows weak form ODEs.
    ode_lib = ps.WeakPDELibrary(
        library_functions=basis_functions[1:],
        function_names=basis_functions_name[1:],
        spatiotemporal_grid=t_train,
        include_bias=True,
        is_uniform=True,
        K=5000,
        )
        
    diff = ps.SINDyDerivative(kind="spline", s=1e-2)
    
    model_set = []
    for threshold_sindy in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    
        # Instantiate and fit the SINDy model with the integral of u_dot
        opt = ps.STLSQ(threshold=threshold_sindy, alpha=1e-12, normalize_columns=False)
        model = ps.SINDy(feature_library=ode_lib, optimizer=opt, differentiation_method=diff)
    
        model.fit(u_train, quiet=True)
        model_set.append(model.coefficients())
        # model.print(precision=precision)
    
    
    ms, best_BIC_model = model_selection_coeff_2d(model_set, u_train, x0, t_train, a, real_list, basis)
    model_best = model_set[best_BIC_model]
    
    model_best_list.append(model_best)
    
    
    



basis_functions_name_list_ = [np.array([f(1,1) for f in basis['names'][0]]), \
                             np.array([f(1,1) for f in basis['names'][1]])]

os.makedirs(path_base, exist_ok=True)
os.makedirs(os.path.join(path_base, 'coeff'), exist_ok=True)
save_path = os.path.join(path_base, f'wsindy_all_{suffix}.txt')
open(save_path, 'w').close()

for idx in range(len(model_best_list)):
    # coef = model_best_list[idx].coefficients()
    coef = model_best_list[idx]
    np.save(os.path.join(path_base, f'coeff/wsindy_{suffix}_{idx}.npy'), coef)

    mask0 = np.abs(coef[0]) > precision
    mask1 = np.abs(coef[1]) > precision
    with open(save_path, "a") as file2:
        file2.writelines(['*'*15, f'result of trajectory {idx} ', '*'*15, '\n'])
        file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
        file2.write(f'basis of feature 0: {basis_functions_name_list_[0][mask0]} \n')
        file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
        file2.write(f'basis of feature 1: {basis_functions_name_list_[1][mask1]} \n\n')