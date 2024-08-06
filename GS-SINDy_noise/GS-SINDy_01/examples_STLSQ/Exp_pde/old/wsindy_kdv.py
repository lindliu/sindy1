#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:44:01 2024

@author: dliu
"""
# https://pysindy.readthedocs.io/en/latest/examples/10_PDEFIND_examples/example.html
import sys
sys.path.insert(1, '../../../GSINDy')
sys.path.insert(1, '../../..')
sys.path.insert(1, '../../tools')

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

K = 5000
noise_l = .9
ensemble = True
threshold = 1e-2
step = 5

# # Load the data stored in a matlab .mat file
# real = np.array([0,0,0,0,-1,-6,0,0,0,0,0])
# order = 3
# kdV = np.load('data/kdv1.npz')
# t = np.ravel(kdV['t'])[::step]
# x = np.ravel(kdV['x'])[::step]
# u_ = np.real(kdV['usol'])[::step,::step]
# # u = u_
# u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)


# real = np.array([0,0,0,0,-1,-1,0,0,0,0,0])
# order = 3
# kdV = loadmat('data/KdV.mat')
# t = np.ravel(kdV['t'])[::step]
# x = np.ravel(kdV['x'])[::step]
# u_ = np.real(kdV['U_exact'])[0][0][::step,::step]
# u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)



# real = np.array([0,0,0,0,-.5,-5,0,0,0,0,0])
# order = 3
# kdV = np.load('data/kdv2.npz')
# t = np.ravel(kdV['t'])[::step]
# x = np.ravel(kdV['x'])[::step]
# u_ = np.real(kdV['usol'])[::step,::step]
# # u = u_
# u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)



real = np.array([0,0,0,-1,0,-1,-1,0,0,0,0,0,0,0])
# real = np.array([0,0,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0])
order = 4
kdV = loadmat('data/KS.mat')
t = np.ravel(kdV['t'])[::step]
x = np.ravel(kdV['x'])[::step]
u_ = np.real(kdV['U_exact'])[0][0][::step,::step]
u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)



# real = np.array([0,0,0,0,0,-1,0,0,0,0,0])
# order = 3
# kdV = np.load('data/IB.npz')
# t = np.ravel(kdV['t'])[::step]
# x = np.ravel(kdV['x'])[::step]
# u_ = np.real(kdV['usol'])[::step,::step]
# # u = u_
# u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)


# real = np.array([0,0,0,0,0,-.7,0,0,0,0,0])
# order = 3
# kdV = np.load('data/IB2.npz')
# t = np.ravel(kdV['t'])[::step]
# x = np.ravel(kdV['x'])[::step]
# u_ = np.real(kdV['usol'])[::step,::step]
# # u = u_
# u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)



# dt = t[1] - t[0]
# dx = x[1] - x[0]

# # Plot u and u_dot
# plt.figure()
# plt.pcolormesh(t, x, u)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# plt.title(r'$u(x, t)$', fontsize=16)
# plt.figure()
# u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)

# plt.pcolormesh(t, x, u_dot)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# plt.title(r'$\dot{u}(x, t)$', fontsize=16)
# plt.show()


# dx = x[1] - x[0]
# ux = ps.FiniteDifference(d=1, axis=0,
#                          drop_endpoints=False)._differentiate(u, dx)
# uxx = ps.FiniteDifference(d=2, axis=0,
#                           drop_endpoints=False)._differentiate(u, dx)
# uxxx = ps.FiniteDifference(d=3, axis=0,
#                            drop_endpoints=False)._differentiate(u, dx)
# uxxxx = ps.FiniteDifference(d=4, axis=0,
#                             drop_endpoints=False)._differentiate(u, dx)

# # Plot derivative results
# plt.figure(figsize=(18, 4))
# plt.subplot(1, 4, 1)
# plt.pcolormesh(t, x, ux)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# plt.title(r'$u_x(x, t)$', fontsize=16)
# plt.subplot(1, 4, 2)
# plt.pcolormesh(t, x, uxx)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# ax = plt.gca()
# ax.set_yticklabels([])
# plt.title(r'$u_{xx}(x, t)$', fontsize=16)
# plt.subplot(1, 4, 3)
# plt.pcolormesh(t, x, uxxx)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# ax = plt.gca()
# ax.set_yticklabels([])
# plt.title(r'$u_{xxx}(x, t)$', fontsize=16)
# plt.subplot(1, 4, 4)
# plt.pcolormesh(t, x, uxxxx)
# plt.xlabel('t', fontsize=16)
# plt.ylabel('x', fontsize=16)
# ax = plt.gca()
# ax.set_yticklabels([])
# plt.title(r'$u_{xxxx}(x, t)$', fontsize=16)
# plt.show()



u = u.reshape(len(x), len(t), 1)

# # Define PDE library that is quadratic in u,
# # and third-order in spatial derivatives of u.
# library_functions = [lambda x: x, lambda x: x * x]
# library_function_names = [lambda x: x, lambda x: x + x]
# pde_lib = ps.PDELibrary(
#     library_functions=library_functions,
#     function_names = library_function_names,
#     # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
#     derivative_order=order,
#     spatial_grid=x,
#     include_bias=False, 
#     is_uniform=True
# )


# # -6*u*ux - uxxx
# # Fit the model with different optimizers.
# # Using normalize_columns = True to improve performance.
# print('STLSQ model: ')
# optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

# print('SR3 model, L0 norm: ')
# optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,
#                    thresholder='l0', normalize_columns=True)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

# print('SR3 model, L1 norm: ')
# optimizer = ps.SR3(threshold=0.05, max_iter=10000, tol=1e-15,
#                    thresholder='l1', normalize_columns=True)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

# print('SSR model: ')
# optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

# print('SSR (metric = model residual) model: ')
# optimizer = ps.SSR(criteria='model_residual', normalize_columns=True, kappa=5e-3)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

# print('FROLs model: ')
# optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
# model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# model.fit(u, t=dt)
# model.print()

mask = real!=0
coeffs = []
for ii in range(20):
    u = u_ + noise_l*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)
    u = u.reshape(len(x), len(t), 1)
    
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
    
    optimizer = ps.STLSQ(threshold=threshold, alpha=1e-12, normalize_columns=False)
    # optimizer = ps.SR3(
    #     threshold=0.1, thresholder="l0", tol=1e-8, normalize_columns=True, max_iter=1000
    # )
    # np.random.seed(1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u, ensemble=ensemble)
    print(f'{ii}. WSINDy: ')
    # print(pde_lib.get_feature_names())
    # model.print()
    
    # print(np.linalg.norm(model.coefficients()[0,...][mask]-real[mask])/np.linalg.norm(real[mask]))
    ###########################################################################
    
    coeffs.append(model.coefficients()[0,...])



    coeff_error = np.linalg.norm(np.c_[coeffs][:,mask]-real[mask], axis=1)/np.linalg.norm(real[mask])
    print(f"{ii} average coeff error: {coeff_error.mean():.4f}")
    
    zz = (np.c_[coeffs]==0)[:,real==0].sum(1)  ## identified zero term
    non_zz = (np.c_[coeffs]!=0)[:,real!=0].sum(1)  ## identified non-zero term
    rate_avg = ((zz+non_zz)/real.shape[0]).mean()
    print(f"{ii} average success rate: {rate_avg:.4f}")




# u = u_ + .9*np.mean(u_**2)**.5 * np.random.normal(0,1,u_.shape)
# u = u.reshape(len(x), len(t), 1)

# # num_traj, num_series, length_series, num_basis = theta_.shape
# ############################# weak form SINDy ##############################
# from pysindy_.feature_library import Shell_custom_theta
# from pysindy_.utils import AxesArray
# def Axes_transfer(theta_):
#     xp = AxesArray(
#         np.empty(
#             theta_.shape,
#             dtype=np.float64,
#             order='C',
#         ),
#         {'ax_time': 0, 'ax_coord': 1, 'ax_sample': None, 'ax_spatial': []},
#     )
    
#     num_basis = theta_.shape[1]
#     for i in range(num_basis):
#         xp[..., i] = theta_[:, i]
    
#     theta = [] + [xp]
    
#     return theta

# X, T = np.meshgrid(x, t)
# XT = np.asarray([X, T]).T

# library_functions = [lambda x: x, lambda x: x * x]
# library_function_names = [lambda x: x, lambda x: x + x]
# pde_lib = ps.WeakPDELibrary(
#     library_functions=library_functions,
#     function_names = library_function_names,
#     # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
#     derivative_order=order,
#     spatiotemporal_grid=XT,
#     is_uniform=True,
#     K=K,
#     include_bias=False
# )



# pde_lib.fit(u)
# theta1 = pde_lib.transform(u)


# from pysindy_.differentiation import FiniteDifference
# differentiation_method = FiniteDifference

# # from typing import Sequence
# # from itertools import product
# # def _zip_like_sequence(x, t):
# #     """Create an iterable like zip(x, t), but works if t is scalar."""
# #     if isinstance(t, Sequence):
# #         return zip(x, t)
# #     else:
# #         return product(x, [t])
# # u_dot = [pde_lib.calc_trajectory(differentiation_method, xi, ti) for xi, ti in _zip_like_sequence([u], t)]
# u_dot = [pde_lib.calc_trajectory(differentiation_method, u, t)]


# optimizer = ps.STLSQ(threshold=1e-1, alpha=1e-12, normalize_columns=False)
# Theta_ = Axes_transfer(theta1)
# lib_generalized = Shell_custom_theta(theta=Theta_)###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
# model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
# model.fit(np.ones([1]), t=1, x_dot=u_dot[0]) 
# print('WSINDy: ')
# print(model.coefficients()[0,...])
# print(pde_lib.get_feature_names())

# print(np.linalg.norm(model.coefficients()[0,...]-real)/np.linalg.norm(real))




























# def get_wsindy_theta_deriv(x,t,u):
#     from pysindy_.feature_library import Shell_custom_theta
#     from pysindy_.utils import AxesArray
#     def Axes_transfer(theta_):
#         xp = AxesArray(
#             np.empty(
#                 theta_.shape,
#                 dtype=np.float64,
#                 order='C',
#             ),
#             {'ax_time': 0, 'ax_coord': 1, 'ax_sample': None, 'ax_spatial': []},
#         )
        
#         num_basis = theta_.shape[1]
#         for i in range(num_basis):
#             xp[..., i] = theta_[:, i]
        
#         theta = [] + [xp]
        
#         return theta
    
#     X, T = np.meshgrid(x, t)
#     XT = np.asarray([X, T]).T
    
#     library_functions = [lambda x: x, lambda x: x * x]
#     library_function_names = [lambda x: x, lambda x: x + x]
#     pde_lib = ps.WeakPDELibrary(
#         library_functions=library_functions,
#         function_names = library_function_names,
#         # function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
#         derivative_order=order,
#         spatiotemporal_grid=XT,
#         is_uniform=True,
#         K=1000,
#         include_bias=True
#     )
    
    
        
#     pde_lib.fit(u)
#     theta1 = pde_lib.transform(u)
    
    
#     from pysindy_.differentiation import FiniteDifference
#     differentiation_method = FiniteDifference
#     u_dot = [pde_lib.calc_trajectory(differentiation_method, u, t)]
    
#     return theta1, u_dot[0]



# import pynumdiff
# def data_interp(x_new, x, y, deriv_spline=True):
#     from scipy import interpolate
#     f = interpolate.interp1d(x, y, kind='cubic')
    
#     if deriv_spline:
#         fd1 = f._spline.derivative(nu=1)
#         return f(x_new), fd1(x_new)
#     else:
#         y_new = f(x_new)
#         dx_new = x_new[1]-x_new[0]
#         y_hat, dydx_hat = pynumdiff.finite_difference.second_order(y_new, dx_new)
#         return y_new, dydx_hat.reshape([-1,1])
# # data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())


# # f = interpolate.RegularGridInterpolator((x, t), u, method='cubic')

# # tt, xx = np.meshgrid(t_new, x)
# # u_new = f(np.c_[xx.flatten(),tt.flatten()])
# # dx_new = t_new[1]-t_new[0]
# # y_hat, dydx_hat = pynumdiff.finite_difference.second_order(u_new, dx_new)



# u_list = [u, u]
# num_traj = len(u_list)


# num_series=60
# window_per=.7


# length = t[-1]-t[0]
# length_sub = length*window_per

# dt = t[1]-t[0]
# step = (1-window_per)/num_series 

# theta0 = []   ### num_series, length
# sol0_deriv = [] ### num_series, length
# for k in range(num_traj):
#     for i in range(num_series):
#         t_new = np.linspace(length*(i*step), length*(i*step)+length_sub, num=int(length_sub//dt))
#         u0, _ = data_interp(t_new, t, u_list[k].squeeze())
#         u0 = u0[...,np.newaxis]
        
#         theta_, u_dot = get_wsindy_theta_deriv(x,t_new,u0)

#         theta0.append(theta_)
#         sol0_deriv.append(u_dot)



# theta0 = np.c_[theta0]
# sol0_deriv = np.c_[sol0_deriv]

# theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:]) ##num_traj, num_series, length_series, num_basis
# sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:]) ##num_traj, num_series, length_series, 1

# sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)


# theta_list = [theta0]
# sol_deriv_list = [sol0_deriv]
