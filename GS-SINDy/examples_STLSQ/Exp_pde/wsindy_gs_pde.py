#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:03:26 2024

@author: do0236li
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
from GSINDy import GSINDy

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




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

def get_wsindy_theta_deriv(x,t,u):
    
    
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
    
    
        
    pde_lib.fit(u)
    theta1 = pde_lib.transform(u)
    
    
    from pysindy_.differentiation import FiniteDifference
    differentiation_method = FiniteDifference
    u_dot = [pde_lib.calc_trajectory(differentiation_method, u, t)]
    
    return theta1, u_dot[0]



import pynumdiff
def data_interp(x_new, x, y, deriv_spline=True):
    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    
    if deriv_spline:
        fd1 = f._spline.derivative(nu=1)
        return f(x_new), fd1(x_new)
    else:
        y_new = f(x_new)
        dx_new = x_new[1]-x_new[0]
        y_hat, dydx_hat = pynumdiff.finite_difference.second_order(y_new, dx_new)
        return y_new, dydx_hat.reshape([-1,1])
# data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())


# f = interpolate.RegularGridInterpolator((x, t), u, method='cubic')

# tt, xx = np.meshgrid(t_new, x)
# u_new = f(np.c_[xx.flatten(),tt.flatten()])
# dx_new = t_new[1]-t_new[0]
# y_hat, dydx_hat = pynumdiff.finite_difference.second_order(u_new, dx_new)


import scipy
def block_diag(A, B):
    return scipy.linalg.block_diag(A,B)

def block_diag_multi_traj(A):
    if len(A)<=1:
        return A
    
    block = scipy.linalg.block_diag(A[0], A[1])
    for i in range(2, len(A)):
        block = block_diag(block, A[i])
    return block

def plot_distribution(Xi0_group, nth_feature, epoch, basis_functions_name, idx):
    fig, ax = plt.subplots(5,5,figsize=(12,12), constrained_layout=True)
    ax = ax.flatten()
    for i in idx:
        ax[i].hist(list(Xi0_group[:,:,i]), alpha = 0.5, label=basis_functions_name[i])
        ax[i].set_title(basis_functions_name[i])
    fig.suptitle(f'{nth_feature}th feature with iteration:{epoch}', fontsize=20)


import copy
def prediction(gsindy, split_basis=True):
    if split_basis:
        all_basis = copy.deepcopy(gsindy.all_basis)
        diff_basis = copy.deepcopy(gsindy.diff_basis)
        same_basis = copy.deepcopy(gsindy.same_basis)
    else:
        all_basis = copy.deepcopy(gsindy.all_basis)
        diff_basis = copy.deepcopy(gsindy.all_basis)
        same_basis = [[] for _ in range(gsindy.num_feature)]
        
        
    Xi_final = np.zeros([gsindy.num_traj, gsindy.num_feature, gsindy.num_basis])
    for k, (theta_org_, sol_deriv_org_) in enumerate(zip(theta_org_list, sol_deriv_org_list)):
        
        ## if no basis selected
        if not all_basis[k].any():
            continue
        
        block_diff_list = [block_[:,diff_basis[k]] for block_ in theta_org_]
        block_diff = block_diag_multi_traj(block_diff_list)
        
        block_same_list = [block_[:,same_basis[k]] for block_ in theta_org_]
        block_same = np.vstack(block_same_list)
    
        Theta = np.c_[block_diff, block_same]
        dXdt = np.vstack(sol_deriv_org_)
        
        # if gsindy.optimizer=='Manually':
        #     ### using own solver, should be the same in sindy without ensemble case
        #     Xi0_ = SLS(Theta, dXdt, gsindy.threshold_sindy, gsindy.alpha)[...,0]
        # else:
        ### using pysindy solver
        Theta_ = Axes_transfer(Theta)
        lib_generalized = Shell_custom_theta(theta=Theta_)  ###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
        model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=gsindy.optimizer)

        model.fit(np.ones([1]), t=1, x_dot=dXdt, ensemble=gsindy.ensemble, quiet=True) ###first 2 inputs has no meanings

        Xi0_ = model.coefficients()[0,...]
                            
        num_diff_ = diff_basis[k].sum()*gsindy.num_traj
        Xi_final[:,k,diff_basis[k]] = Xi0_[:num_diff_].reshape([gsindy.num_traj,-1])
        Xi_final[:,k,same_basis[k]] = Xi0_[num_diff_:]

    if split_basis:
        gsindy.all_basis = copy.deepcopy(all_basis)
    else:
        gsindy.all_basis = copy.deepcopy(all_basis)
    return Xi_final



def split(t, x, u, ratio=.95):
    cut = int(t.shape[0]*ratio)
    u1 = u[:, :cut]
    u2 = u[:, -cut:]

    return u1, u2, t[:cut]



threshold_group = 1e-2
threshold_similarity = 1e-1


precision = 1e-2
K = 2000
noise_l = 0.2
threshold_sindy = 1e-2
step = 5


# real = np.array([0,0,0,0,0,-1,0,0,0,0,0])
# order = 3
# IB_1 = np.load('data/IB_1.npz')
# t = np.ravel(IB_1['t'])[::step]
# x = np.ravel(IB_1['x'])[::step]
# u = np.real(IB_1['usol'])[::step,::step]


# real = np.array([0,0,0,0,0,-.7,0,0,0,0,0])
# order = 3
# IB_2 = np.load('data/IB_2.npz')
# t = np.ravel(IB_2['t'])[::step]
# x = np.ravel(IB_2['x'])[::step]
# u = np.real(IB_2['usol'])[::step,::step]


real = np.array([0,0,0,0,-1,-1,0,0,0,0,0])
order = 3
KDV_1 = np.load('data/KDV_1.npz')
t = np.ravel(KDV_1['t'])[::step]
x = np.ravel(KDV_1['x'])[::step]
u = np.real(KDV_1['usol'])[::step,::step]


# real = np.array([0,0,0,0,-.7,-1.5,0,0,0,0,0])
# order = 3
# KDV_2 = np.load('data/KDV_2.npz')
# t = np.ravel(KDV_2['t'])[::step]
# x = np.ravel(KDV_2['x'])[::step]
# u = np.real(KDV_2['usol'])[::step,::step]


# real = np.array([0,0,0,-1,0,-1,-1,0,0,0,0,0,0,0])
# order = 4
# KS_1 = np.load('data/KS_1.npz')
# t = np.ravel(KS_1['t'])[::step]
# x = np.ravel(KS_1['x'])[::step]
# u = np.real(KS_1['usol'])[::step,::step]


mask = real!=0
coeffs = []
for ii in range(20):
    
    u_noised = u + noise_l*np.mean(u**2)**.5 * np.random.normal(0,1,u.shape)
    u1, u2, t_ = split(t,x,u_noised)
        
    
    
    u_list = [u1, u2]
    num_traj = len(u_list)

    theta_org1, sol_deriv_org1 = get_wsindy_theta_deriv(x,t_,u1[...,np.newaxis])
    theta_org2, sol_deriv_org2 = get_wsindy_theta_deriv(x,t_,u2[...,np.newaxis])
    
    theta_org_list, sol_deriv_org_list = [[theta_org1,theta_org2]], [[sol_deriv_org1,sol_deriv_org2]]
    
    
    num_series=60
    window_per=.7
    
    
    length = t_[-1]-t_[0]
    length_sub = length*window_per
    
    dt = t_[1]-t_[0]
    step = (1-window_per)/num_series 
    
    theta0 = []   ### num_series, length
    sol0_deriv = [] ### num_series, length
    for k in range(num_traj):
        for i in range(num_series):
            # # t_new = np.linspace(length*(i*step), length*(i*step)+length_sub, num=int(length_sub//dt))
            # # u0, _ = data_interp(t_new, t, u_list[k].squeeze())
            # # u0 = u0[...,np.newaxis]
            
            # t_new = t[i:i+t.shape[0]-num_series]
            # u0 = u_list[k][:,i:i+t.shape[0]-num_series]
            # u0 = u0[...,np.newaxis]
            
            # theta_, u_dot = get_wsindy_theta_deriv(x,t_new,u0)
    
            idxx = np.random.choice(K, int(K*.7), replace=False)
            theta_ = theta_org_list[0][k][idxx]
            u_dot = sol_deriv_org_list[0][k][idxx]
    
            theta0.append(theta_)
            sol0_deriv.append(u_dot)
    
    
    
    theta0 = np.c_[theta0]
    sol0_deriv = np.c_[sol0_deriv]
    
    theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:]) ##num_traj, num_series, length_series, num_basis
    sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:]) ##num_traj, num_series, length_series, 1
    
    sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)
    
    
    theta_list = [theta0]
    sol_deriv_list = [sol0_deriv]
    
    
    
    basis = {'functions': [0], 'names': [1]}
    gsindy = GSINDy(basis = basis, 
                    num_traj = num_traj, 
                    num_feature = 1, 
                    threshold_sindy = threshold_sindy, 
                    threshold_group=threshold_group, 
                    # threshold_similarity=1e-3, 
                    threshold_similarity=threshold_similarity, 
                    # precision = precision, 
                    alpha=1e-12,
                    # deriv_spline=deriv_spline,
                    max_iter = 20, 
                    optimizer='SQTL', 
                    ensemble=False,
                    normalize_columns=True)   
    ### get sub-series for each trajectory
    _, gsindy.num_series, gsindy.length_series, gsindy.num_basis = theta0.shape
    gsindy.idx_basis = np.arange(gsindy.num_basis)
    
    gsindy.theta_list = theta_list
    gsindy.sol_deriv_list = sol_deriv_list
    # gsindy.get_multi_sub_series_2D(sol_org_list, t, num_series=100, window_per=.7) ### to get theta_list, sol_deriv_list
    ### basis identification for each trajectory
    gsindy.basis_identification(remove_per=.2, plot_dist=False) ##True
                    
    all_basis = gsindy.all_basis
    
    
    
    
    
    
    
    
    X, T = np.meshgrid(x, t)
    XT = np.asarray([X, T]).T
    
    # np.random.seed(1)
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
    
    pde_lib.fit(u_noised[:,:,np.newaxis])
    theta1 = pde_lib.transform(u_noised[:,:,np.newaxis])[:,gsindy.all_basis[0]]
    
    
    
    from pysindy_.differentiation import FiniteDifference
    differentiation_method = FiniteDifference
    u_dot = [pde_lib.calc_trajectory(differentiation_method, u_noised, t)][0]
    
    optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=1e-12, normalize_columns=True)
    oop = ps.SINDyOptimizer(optimizer)
    oop.fit(theta1, u_dot)
    # oop.fit(theta1[:,[0,1,2,3,4,5]], u_dot)
    
    # from sklearn.linear_model import LinearRegression
    # coef_ = LinearRegression(fit_intercept=False).fit(theta1[:,[4,5,9]], u_dot).coef_
    # coef_
    
    Xi_final = np.zeros_like(real,dtype=float)
    Xi_final[gsindy.all_basis[0]] = oop.coef_[0,...]
    Xi_final = Xi_final.reshape(1,1,Xi_final.shape[0])






    # from pysindy_.differentiation import FiniteDifference
    # differentiation_method = FiniteDifference
    # u_dot = [pde_lib.calc_trajectory(differentiation_method, u_noised, t)][0]
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


    # optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=1e-12, normalize_columns=True)
    # Theta_ = Axes_transfer(theta1)
    # lib_generalized = Shell_custom_theta(theta=Theta_)###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
    # model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
    # model.fit(np.ones([1]), t=1, x_dot=u_dot) 
    # # print('WSINDy: ')
    # # print(model.coefficients()[0,...])
    # Xi_final = np.zeros_like(real,dtype=float)
    # # Xi_final = np.array(list(model.coefficients()[0,...]))
    # Xi_final[gsindy.all_basis[0]] = list(model.coefficients()[0,...])
    
    # Xi_final = Xi_final.reshape(1,1,Xi_final.shape[0])
    # # print(f'{ii}: ', np.linalg.norm(Xi_final-real)/np.linalg.norm(real))






    

    # theta_org1, sol_deriv_org1 = get_wsindy_theta_deriv(x,t,u_noised[...,np.newaxis])
    # theta_org_list, sol_deriv_org_list = [[theta_org1,theta_org1]], [[sol_deriv_org1,sol_deriv_org1]]
    # Xi_final = prediction(gsindy, split_basis=True)
    # # print(Xi_final)
    # # print(f'{ii}: ', np.linalg.norm(Xi_final[0,0][mask]-real[mask])/np.linalg.norm(real[mask]))
    
    
    
    Xi_final[np.abs(Xi_final)<precision] = 0
    coeffs.append(Xi_final)  
    
    coeff_error = np.linalg.norm(np.c_[coeffs][:,0,0,:][:,mask]-real[mask], axis=1)/np.linalg.norm(real[mask])
    print(f'{ii} average coeff error: ', f'{coeff_error.mean():.4f}')
    
    zz = (np.c_[coeffs][:,0,0,:]==0)[:,real==0].sum(1)  ## identified zero term
    non_zz = (np.c_[coeffs][:,0,0,:]!=0)[:,real!=0].sum(1)  ## identified non-zero term
    rate_avg = ((zz+non_zz)/real.shape[0]).mean()
    print(f'{ii} average success rate: ', f'{rate_avg:.4f}')
    
    
    TP = (np.c_[coeffs][:,0,0,:]!=0)[:,real!=0].sum(1) ## correctly identified nonzero coefficients
    FN = (np.c_[coeffs][:,0,0,:]==0)[:,real!=0].sum(1) ## coefficients falsely identified as zero
    FP = (np.c_[coeffs][:,0,0,:]!=0)[:,real==0].sum(1) ## coefficients falsely identified as nonzero
    TPR = TP/(TP+FN+FP)
    print(f"{ii} average TPR: {TPR.mean():.4f}")

    print('#####')

    
coeff_error = np.linalg.norm(np.c_[coeffs][:,0,0,:][:,mask]-real[mask], axis=1)/np.linalg.norm(real[mask])
print(coeff_error.mean())
# # print(pde_lib.get_feature_names())



zz = (np.c_[coeffs][:,0,0,:]==0)[:,real==0].sum(1)  ## identified zero term
non_zz = (np.c_[coeffs][:,0,0,:]!=0)[:,real!=0].sum(1)  ## identified non-zero term
rate_avg = ((zz+non_zz)/real.shape[0]).mean()
print(f"success rate: {rate_avg:.4f}")