#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:09:19 2024

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../tools')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

import pysindy_ as ps

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
# np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


ensemble = True

precision = 1e-3
K = 2000
noise_l = 0.1
threshold = 1e-2
step = 3





real = np.array([0,0,0,0,-1,-1,0,0,0,0,0])
order = 3
KDV_1 = np.load('data/KDV_1.npz')
t = np.ravel(KDV_1['t'])[::step]
x = np.ravel(KDV_1['x'])[::step]
u = np.real(KDV_1['usol'])[::step,::step]




ii=0

u_noise = u + noise_l*np.mean(u**2)**.5 * np.random.normal(0,1,u.shape)
u_noise = u_noise[:,:,np.newaxis]


############################# weak form SINDy ##############################
X, T = np.meshgrid(x, t)
XT = np.asarray([X, T]).T

library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names = library_function_names,
    # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
    derivative_order=order,
    spatiotemporal_grid=XT,
    is_uniform=True,
    K=K,
    include_bias=False
)


optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=True)
# optimizer = ps.SR3(threshold=threshold, thresholder="l0", tol=1e-8, normalize_columns=True, max_iter=1000)
# np.random.seed(1)

model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u_noise, ensemble=ensemble)
print(f'{ii}. WSINDy: ')
# print(pde_lib.get_feature_names())
model.print()














# import pickle
# with open('x.pickle', 'rb') as handle:
#     theta = pickle.load(handle)

# with open('y.pickle', 'rb') as handle:
#     y = pickle.load(handle)





from pysindy_.feature_library import Shell_custom_theta
from pysindy_.utils import AxesArray
def Axes_transfer(theta_):
    xp = AxesArray(
        np.empty(
            theta_.shape,
            dtype=np.float64,
            order='C',
        ),
        {'ax_time': 0, 'ax_coord': 1, 'ax_sample': None, 'ax_spatial': []},
    )
    
    num_basis = theta_.shape[1]
    for i in range(num_basis):
        xp[..., i] = theta_[:, i]
    
    theta = [] + [xp]
    
    return theta


X, T = np.meshgrid(x, t)
XT = np.asarray([X, T]).T

library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names = library_function_names,
    # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
    derivative_order=order,
    spatiotemporal_grid=XT,
    is_uniform=True,
    K=K,
    include_bias=False
)

pde_lib.fit(u_noise[:,:,np.newaxis])
theta1 = pde_lib.transform(u_noise[:,:,np.newaxis])#[:,gsindy.all_basis[0]]

from pysindy_.differentiation import FiniteDifference
differentiation_method = FiniteDifference
u_dot = [pde_lib.calc_trajectory(differentiation_method, u_noise, t)][0]


optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=True)
oop = ps.SINDyOptimizer(optimizer)
# oop.fit(theta1, u_dot)
oop.fit(theta1[:,[0,1,2,3,4,5]], u_dot)

# from sklearn.linear_model import LinearRegression
# coef_ = LinearRegression(fit_intercept=False).fit(theta1[:,[4,5,9]], u_dot).coef_
# coef_

Xi_final = np.zeros_like(real,dtype=float)
Xi_final = oop.coef_[0,...]

Xi_final[np.abs(Xi_final)<precision] = 0
print(Xi_final)






# # from typing import Sequence
# # from itertools import product
# # def _zip_like_sequence(x, t):
# #     """Create an iterable like zip(x, t), but works if t is scalar."""
# #     if isinstance(t, Sequence):
# #         return zip(x, t)
# #     else:
# #         return product(x, [t])
# # u_dot = [pde_lib.calc_trajectory(differentiation_method, xi, ti) for xi, ti in _zip_like_sequence([u], t)][0]


# # from typing import List
# # def concat_sample_axis(x_list: List[AxesArray]):
# #     """Concatenate all trajectories and axes used to create samples."""
# #     new_arrs = []
# #     for x in x_list:
# #         sample_axes = (
# #             x.ax_spatial
# #             + ([x.ax_time] if x.ax_time is not None else [])
# #             + ([x.ax_sample] if x.ax_sample is not None else [])
# #         )
# #         new_axes = {"ax_sample": 0, "ax_coord": 1}
# #         n_samples = np.prod([x.shape[ax] for ax in sample_axes])
# #         arr = AxesArray(x.reshape((n_samples, x.shape[x.ax_coord])), new_axes)
# #         new_arrs.append(arr)
# #     return np.concatenate(new_arrs, axis=new_arrs[0].ax_sample)
# # u_dot = concat_sample_axis(u_dot)

# optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=True)
# Theta_ = Axes_transfer(theta1)
# lib_generalized = Shell_custom_theta(theta=Theta_)###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
# model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
# model.fit(np.ones([1]), t=1, x_dot=u_dot) 
# # print('WSINDy: ')
# # print(model.coefficients()[0,...])


# Xi_final = np.zeros_like(real,dtype=float)
# Xi_final = np.array(list(model.coefficients()[0,...]))
# print(Xi_final)
# # Xi_final[gsindy.all_basis[0]] = list(model.coefficients()[0,...])

# # Xi_final = Xi_final.reshape(1,1,Xi_final.shape[0])
# # print(f'{ii}: ', np.linalg.norm(Xi_final-real)/np.linalg.norm(real))



