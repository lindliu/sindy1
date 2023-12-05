#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:42:07 2023

@author: dliu
"""


# %%
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import ridge_regression, LinearRegression, SGDRegressor
from func import func1, func2, func3, func4, func5, func6, func7, \
    get_sol_deriv, monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name


MSE = lambda x: (np.array(x)**2).mean()
# MSE = lambda x: np.linalg.norm(x)


def SLS(Theta, DXdt, threshold, alpha=.05):
    n_feature = DXdt.shape[1]
    Xi = ridge_regression(Theta,DXdt, alpha=alpha).T
    # Xi = np.linalg.lstsq(Theta,DXdt, rcond=None)[0]
    # Xi = solve_minnonzero(Theta,DXdt)
    Xi[np.abs(Xi)<threshold] = 0
    # print(Xi)
    for _ in range(20):
        smallinds = np.abs(Xi)<threshold
        Xi[smallinds] = 0
        for ind in range(n_feature):
            
            if Xi[:,ind].sum()==0:
                break
            
            biginds = ~smallinds[:,ind]
            Xi[biginds,ind] = ridge_regression(Theta[:,biginds], DXdt[:,ind], alpha=alpha).T
            # Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],DXdt[:,ind], rcond=None)[0]
            # Xi[biginds,ind] = solve_minnonzero(Theta[:,biginds],DXdt[:,ind])
        
    # reg = LinearRegression(fit_intercept=False)
    # ind_ = np.abs(Xi.T) > 1e-14
    # ind_[:,:num_traj] = True
    # num_basis = ind_.sum()
    # while True:
    #     coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    #     for i in range(ind_.shape[0]):
    #         if np.any(ind_[i]):
    #             coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
                
    #             ind_[i, np.abs(coef[i,:])<threshold_tol] = False
    #             ind_[i,:num_traj] = True
        
    #     if num_basis==ind_.sum():
    #         break
    #     num_basis = ind_.sum()
        
    # Xi = coef.T
    # Xi[np.abs(Xi)<threshold_tol] = 0


    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-14
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T
    Xi[np.abs(Xi)<threshold_tol] = 0

    return Xi

def data_interp(x_new, x, y, deriv_spline=True):
    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    
    if deriv_spline:
        fd1 = f._spline.derivative(nu=1)
        return f(x_new), fd1(x_new)
    else:
        import pynumdiff
        y_new = f(x_new)
        dx_new = x_new[1]-x_new[0]
        y_hat, dydx_hat = pynumdiff.finite_difference.second_order(y_new, dx_new)
        return y_new, dydx_hat.reshape([-1,1])
# data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())

from func import func12_, func3_, func4_
deriv_spline = True#False#
threshold_tol = 1e-2

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


# num_feature = len(x0)
#%%
num_traj = len(a)
sol0_org, theta0_org, sol0_deriv_org = [], [], []
sol1_org, theta1_org, sol1_deriv_org = [], [], []
for i in range(num_traj):
    sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i], deriv_spline)
    theta_ = monomial(sol_)

    sol0_org.append(sol_[:,[0]])
    theta0_org.append(theta_)
    sol0_deriv_org.append(sol_deriv_[:,[0]])
    
    sol1_org.append(sol_[:,[1]])
    theta1_org.append(theta_)
    sol1_deriv_org.append(sol_deriv_[:,[1]])
    
    plt.plot(t_, sol_, 'o', markersize=1, label=f'{a[i]}')
plt.legend()
plt.text(1, .95, f'${real0}$', fontsize=12)
plt.text(1, .8, f'${real1}$', fontsize=12)

theta0_org = np.c_[theta0_org]
theta1_org = np.c_[theta1_org]
sol0_deriv_org = np.vstack(sol0_deriv_org)
sol1_deriv_org = np.vstack(sol1_deriv_org)

theta_org_list = [theta0_org, theta1_org]
sol_deriv_org_list = [sol0_deriv_org, sol1_deriv_org]


#%%
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

def plot(Xi0_group, nth_feature, epoch):
    fig, ax = plt.subplots(5,5,figsize=(12,12), constrained_layout=True)
    ax = ax.flatten()
    for i in idx_basis[idx_activ]:
        # for j in range(num_traj):
        ax[i].hist(list(Xi0_group[:,:,i]), alpha = 0.5, label=monomial_name[i])
        ax[i].set_title(monomial_name[i])
        # ax[i].legend()
    fig.suptitle(f'{nth_feature}th feature with iteration:{epoch}', fontsize=20)

### get data
per = .7
length = t[-1]-t[0]
length_sub = length*per

# threshold_sindy = 5e-2
num_series = int(100*(1-per))*2
theta0, theta1 = [], []   ### num_series, length
sol0_deriv, sol1_deriv = [], [] ### num_series, length
Xi0_list_, Xi1_list_ = [], []
for k in range(num_traj):
    for i in range(num_series):
        sol0_, sol0_deriv_ = data_interp(np.linspace(length*(i*.005),length*(per+i*.005),\
                                                     num=int(length_sub//dt)), t, sol0_org[k].squeeze(), deriv_spline=True)
        sol1_, sol1_deriv_ = data_interp(np.linspace(length*(i*.005),length*(per+i*.005),\
                                                     num=int(length_sub//dt)), t, sol1_org[k].squeeze(), deriv_spline=True)

        theta_ = monomial(np.c_[sol0_,sol1_])
        theta0.append(theta_)
        theta1.append(theta_)
        sol0_deriv.append(sol0_deriv_)
        sol1_deriv.append(sol1_deriv_)

theta0 = np.c_[theta0]
theta1 = np.c_[theta1]
sol0_deriv = np.c_[sol0_deriv]
sol1_deriv = np.c_[sol1_deriv]

theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:])
theta1 = theta1.reshape(num_traj, -1, *theta1.shape[1:])
sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:])
sol1_deriv = sol1_deriv.reshape(num_traj, -1, *sol1_deriv.shape[1:])

sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)
sol1_deriv = np.vstack(sol1_deriv.transpose(0,2,1,3)).transpose(1,0,2)


theta_list = [theta0, theta1]
sol_deriv_list = [sol0_deriv, sol1_deriv]
num_feature = len(theta_list)
assert num_feature==len(x0)

num_traj, num_series, length_series, num_basis = theta0.shape
idx_basis = np.arange(num_basis)


max_iter = 20
all_basis = []
same_basis = []
diff_basis = []
for nth_feature, (theta_, sol_deriv_) in enumerate(zip(theta_list, sol_deriv_list)):
    idx_activ = np.ones([num_basis],dtype=bool)
    
    idx_same_activ = np.zeros_like(idx_activ,dtype=bool)
    idx_diff_activ = np.ones_like(idx_activ,dtype=bool)
    idx_same_activ_pre = copy.deepcopy(idx_same_activ)
    idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)

    for epoch in range(max_iter):
        ### do SINDy over each sub-series
        Xi0_group = np.zeros([num_traj, num_series, num_basis])
        for j in range(num_series):
            block_diff_list = [block_[j][:,idx_diff_activ] for block_ in theta_]
            block_diff = block_diag_multi_traj(block_diff_list)
            
            block_same_list = [block_[j][:,idx_same_activ] for block_ in theta_]
            block_same = np.vstack(block_same_list)
        
            Theta = np.c_[block_diff, block_same]
            dXdt = sol_deriv_[j]
            Xi0_ = SLS(Theta, dXdt, threshold_sindy)[...,0]
            
            num_diff_ = idx_diff_activ.sum()*num_traj
            Xi0_group[:,j,idx_diff_activ] = Xi0_[:num_diff_].reshape([num_traj,-1])
            Xi0_group[:,j,idx_same_activ] = Xi0_[num_diff_:]
        
        plot(Xi0_group, nth_feature, epoch)
        
        
        ### remove part outliers of estimated parameters
        tail = int(num_series*.1)
        Xi0_group = np.sort(Xi0_group, axis=1)[:,tail:-tail,:]
        # idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_sindy)
        idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_tol)
        
        ##### Xi0_group normalization for calculate distance of distributions#####
        Xi0_group[:,:,~idx_activ] = 0
        norm_each_coef = np.linalg.norm(np.vstack(Xi0_group),axis=0)
        Xi0_group[:,:,idx_activ] = Xi0_group[:,:,idx_activ]/norm_each_coef[idx_activ]
          
        
        ##### To find the identical basis
        idx_same_activ = np.logical_and(idx_activ, idx_same_activ)
        idx_diff_activ = copy.deepcopy(idx_activ)
        idx_diff_activ[idx_same_activ] = False
        
        from itertools import combinations
        from scipy.stats import wasserstein_distance
        radius = []
        for k in idx_basis[idx_diff_activ]:
            radius_ = []
            for i,j in combinations(np.arange(num_traj), 2):
                radius_.append(wasserstein_distance(Xi0_group[i,:,k],Xi0_group[j,:,k]))
            dist_ = scipy.cluster.hierarchy.linkage(radius_, method='single')[:,2]
            # radius.append(np.max(dist_))
            # radius.append(np.mean(dist_))
            radius.append(np.median(dist_))
        radius = np.array(radius)
        
        idx_similar = np.where(radius<threshold_similarity) 
        idx_same = idx_basis[idx_diff_activ][idx_similar]
        idx_same_activ[idx_same] = True
        idx_diff_activ[idx_same] = False
        
        assert (np.logical_or(idx_diff_activ, idx_same_activ) == idx_activ).any(0)
        
        if (idx_same_activ==idx_same_activ_pre).all() and (idx_diff_activ==idx_diff_activ_pre).all():
            break
        else:
            idx_same_activ_pre = copy.deepcopy(idx_same_activ)
            idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)
            
    all_basis.append(idx_activ)
    same_basis.append(idx_same_activ)
    diff_basis.append(idx_diff_activ)

# for k in range(21):
#     std = Xi0_group[0,:,k].std()
#     mean = Xi0_group[0,:,k].mean()
#     if mean!=0:
#         print(f'{k} std: {std:.2f}, mean: {mean:.2f}, ratio: {std/mean:.2f}')



#%% get final predicted Xi
Xi_final = np.zeros([num_traj, num_feature, num_basis])
for k, (theta_org_, sol_deriv_org_) in enumerate(zip(theta_org_list, sol_deriv_org_list)):
    block_diff_list = [block_[:,diff_basis[k]] for block_ in theta_org_]
    block_diff = block_diag_multi_traj(block_diff_list)
    
    block_same_list = [block_[:,same_basis[k]] for block_ in theta_org_]
    block_same = np.vstack(block_same_list)

    Theta = np.c_[block_diff, block_same]
    dXdt = sol_deriv_org_
    Xi0_ = SLS(Theta, dXdt, threshold_sindy)[...,0]
    
    num_diff_ = diff_basis[k].sum()*num_traj
    Xi_final[:,k,diff_basis[k]] = Xi0_[:num_diff_].reshape([num_traj,-1])
    Xi_final[:,k,same_basis[k]] = Xi0_[num_diff_:]
    
mask_tol = np.abs(Xi_final.mean(0))>threshold_tol
all_basis[0] = np.logical_and(mask_tol[0],all_basis[0])
all_basis[1] = np.logical_and(mask_tol[1],all_basis[1])

print('*'*50)
print(f'real0: {real0}')
print(f'real1: {real1}')
print(f'feature 0 with different basis {monomial_name[diff_basis[0]]}: \n {Xi_final[:,0,all_basis[0]]} \n {monomial_name[all_basis[0]]}')
print(f'feature 1 with different basis {monomial_name[diff_basis[1]]}: \n {Xi_final[:,1,all_basis[1]]} \n {monomial_name[all_basis[1]]}')




#%% compare to pysindy
import pysindy as ps

if func.__name__ not in ['func6', 'func7']:
    from pysindy.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
    # lib_generalized = PolynomialLibrary(degree=5)
    functions = [lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
            lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5]
    names = [lambda x,y: '1', \
            lambda x,y: 'x', lambda x,y: 'y', \
            lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
            lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
            lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4', \
            lambda x,y: 'x^5', lambda x,y: 'x^4y', lambda x,y: 'x^3y^2', lambda x,y: 'x^2y^3', lambda x,y: 'xy^4', lambda x,y: 'y^5']
    lib_custom = CustomLibrary(library_functions=functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])

    from pysindy.optimizers import STLSQ
    optimizer = STLSQ(threshold=threshold_sindy, alpha=alpha)
    model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
        
else:
    from pysindy.feature_library import FourierLibrary, CustomLibrary
    from pysindy.feature_library import GeneralizedLibrary
    functions = [lambda x,y: 1, lambda x,y: x, lambda x,y: y, lambda x,y: np.sin(x), lambda x,y: np.sin(y), \
                 lambda x,y: np.cos(x), lambda x,y: np.cos(y)]
    names = [lambda x,y: '1', lambda x,y: 'x', lambda x,y: 'y', lambda x,y: 'sin(x)', lambda x,y: 'sin(y)', \
             lambda x,y: 'cos(x)', lambda x,y: 'cos(y)']
    lib_custom = CustomLibrary(library_functions=functions, function_names=names)
    # lib_generalized = GeneralizedLibrary([lib_custom, lib_fourier])
    lib_generalized = GeneralizedLibrary([lib_custom])
    # x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    # lib_generalized.fit(x)
    # lib_generalized.transform(x)
    from pysindy.optimizers import STLSQ
    optimizer = STLSQ(threshold=threshold_sindy, alpha=alpha)
    
    model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)


for i in range(len(a)):
    sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i], deriv_spline)

    model.fit(sol_, t=t_, x_dot=sol_deriv_)#, ensemble=True, quiet=True)
    model.print()
    # model.coefficients()
    
    # theta_ = monomial(sol_)
    # print(SLS(theta_, sol_deriv_, threshold_sindy))
    
