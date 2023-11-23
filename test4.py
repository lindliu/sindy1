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
tol = 1e-3

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
    
    threshold = tol
    
    # reg = LinearRegression(fit_intercept=False)
    # ind_ = np.abs(Xi.T) > 1e-14
    # ind_[:,:num_traj] = True
    # num_basis = ind_.sum()
    # while True:
    #     coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    #     for i in range(ind_.shape[0]):
    #         if np.any(ind_[i]):
    #             coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
                
    #             ind_[i, np.abs(coef[i,:])<threshold] = False
    #             ind_[i,:num_traj] = True
        
    #     if num_basis==ind_.sum():
    #         break
    #     num_basis = ind_.sum()
        
    # Xi = coef.T
    # Xi[np.abs(Xi)<threshold] = 0


    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-14
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T
    Xi[np.abs(Xi)<threshold] = 0

    return Xi


def SLS_multi(theta0_, sol_deriv0, threshold):
    num_traj = len(theta0_)
    Xi_group = np.zeros([theta0_[0].shape[1],num_traj])
    for i in range(num_traj): 
        Xi_ = SLS(theta0_[i], sol_deriv0[i], threshold, alpha=alpha).squeeze()
        Xi_group[:,i] = Xi_
        
    mask = np.linalg.norm(Xi_group,axis=1) > tol*num_traj
    ind = np.zeros(theta0_[0].shape[1], dtype=bool)
    ind[mask] = True
     
    theta0_ = [ele[:,ind] for ele in theta0_]
    
    return ind, theta0_


def get_one_hot_theta(theta0_, sol_deriv0, feature_idx=None):
    if feature_idx is None:
        return np.vstack(theta0_), np.vstack(sol_deriv0)
    
    num_feature = theta0_[0].shape[1]
    i_same = np.ones(num_feature, dtype=bool)

    i_diff = list(feature_idx)
    i_same[i_diff] = False

    num_traj = len(theta0_)
    num_diff = len(i_diff)
    
    Theta_diff, Theta_same = [], []
    for j in range(num_traj): ###each trajectory
        Theta_diff_ = np.zeros([theta0_[0].shape[0], num_traj*num_diff])
        Theta_diff_[:,j*num_diff:(j+1)*num_diff] = theta0_[j][:,i_diff]
        
        Theta_diff.append(Theta_diff_)
        Theta_same.append(theta0_[j][:,i_same])
    
    Theta_diff = np.vstack(Theta_diff)
    Theta_same = np.vstack(Theta_same)
    
    Theta = np.c_[Theta_diff,Theta_same]
    DXdt = np.vstack(sol_deriv0)
    return Theta, DXdt

def select_features_of_multi_trajectory(theta0, sol_deriv0, threshold):
    ### select feature by SLS
    pre = -1
    ind_num0 = np.arange(monomial_num)
    theta0_ = copy.deepcopy(theta0)
    while True:
        ind0_, theta0_ = SLS_multi(theta0_, sol_deriv0, threshold)
        ind_num0 = ind_num0[ind0_]

        cur = len(ind_num0)
        if cur==pre:
            break
        else:
            pre = cur
    return ind_num0, theta0_

from itertools import combinations
def select_diff_feature_idx(theta0_, sol_deriv0, n=1):
    ### select proper feature index by one hot matrix
    num_feature = theta0_[0].shape[1]
    sgd = LinearRegression(fit_intercept=False)
    
    i0, dist = [], []
    # i_min0, i_val0 = None, float('Inf')
    for i in combinations(np.arange(num_feature), n):
        Theta0, DXdt0 = get_one_hot_theta(theta0_, sol_deriv0, i)
        Xi0 = sgd.fit(Theta0, DXdt0).coef_.T
        
        distance0 = MSE(Theta0@Xi0-DXdt0)
        # print(f'L2 distance of {i}: {distance0:3E}')
        dist.append(distance0)
        i0.append(i)
        # if i_val0>distance0:
        #     i_val0 = distance0
        #     i_min0 = i

    # print(f'index {i_min0} has lowest error')
    # return list(i_min0), i_val0, dist

    return list(i0[np.argmin(dist)]), i0, dist

def data_interp(x_new, x, y):
    from scipy import interpolate
    
    f = interpolate.interp1d(x, y, kind='cubic')
    fd1 = f._spline.derivative(nu=1)
    return f(x_new), fd1(x_new)
# data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())

from func import func12_, func3_, func4_

threshold_sindy=threshold0=threshold1=4e-2

# alpha = .01
# dt = .01   ## 0,3
# t = np.arange(0,1.5,dt)
# x0 = [.5, 1]
# a = [(.16, .25), (.3, .4)]
# func = func12_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + b*x^2"
# real1 = "y'=-y"
# threshold0 = 1e-1
# threshold1 = 1e-1

# alpha = .05
# dt = .01      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,8,dt)
# x0 = [.5, 1]
# a = [(.2, -.6), (.4, -.8), (.6, -1)]
# func = func3_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"    
# threshold0 = 1e-1
# threshold1 = 1e-1

# alpha = .1
# dt = .01    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [4, 1]
# a = [(.7,-.8), (1,-1)]
# func = func4_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x + b*xy"
# real1 = "y'=b*y + a*xy"
# # threshold0 = 1e-1
# # threshold1 = 1e-1


# ################### 1 variable ####################
alpha = .05
dt = .005   ## 0,3
t = np.arange(0,1.5,dt)
x0 = [.5, 1]
a = [(.12,), (.16,), (.2,)]
func = func1
monomial = monomial_poly
monomial_name = monomial_poly_name
real0 = "x'=a + x^2"
real1 = "y'=-y"
# threshold0 = 1e-1
# threshold1 = 1e-1

# alpha = .05
# dt = .01
# t = np.arange(0,2,dt)
# x0 = [.5, 1]
# a = [(.25,), (.5,)]
# func = func2
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=.2 + a*x^2"
# real1 = "y'=-y"
# # threshold0 = 1e-1
# # threshold1 = 1e-1

# alpha = .05
# dt = .01      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,8,dt)
# x0 = [.5, 1]
# a = [(.2,), (.4,), (.6,)]
# func = func3
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"    
# # threshold0 = 1e-1
# # threshold1 = 1e-1

# alpha = .05
# dt = .01    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [4, 1]
# a = [(.7,), (1,)]#,(.5,),(.6,)]
# func = func4
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"    
# # threshold0 = 1e-1
# # threshold1 = 1e-1

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
# # threshold0 = 1e-1
# # threshold1 = 1e-1

# alpha = .05
# dt = .1
# t = np.arange(0,5,dt)
# x0 = [np.pi-.1, 0]
# # a = [-5, -6]
# a = [(-.15,), (-1,), (-2,)]
# func = func7
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=-0.25y+a*sin(x)"
# # threshold0 = 1e-1
# # threshold1 = 1e-1



# num_feature = len(x0)
#%%
num_traj = len(a)
sol0, theta0, sol_deriv0 = [], [], []
sol1, theta1, sol_deriv1 = [], [], []
for i in range(num_traj):
    sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i], step=1)
    
    # t_ = np.linspace(0,t[-1])
    # sol0_, sol0_deriv_ = data_interp(t_, t, sol_[:,0])
    # sol1_, sol1_deriv_ = data_interp(t_, t, sol_[:,1])
    # sol_ = np.c_[sol0_,sol1_]
    # sol_deriv_ = np.c_[sol0_deriv_, sol1_deriv_]
    
    theta_ = monomial(sol_)

    sol0.append(sol_[:,[0]])
    theta0.append(theta_)
    sol_deriv0.append(sol_deriv_[:,[0]])
    
    sol1.append(sol_[:,[1]])
    theta1.append(theta_)
    sol_deriv1.append(sol_deriv_[:,[1]])
    
    plt.plot(t_, sol_, 'o', markersize=1)
monomial_num = theta0[0].shape[1]

#%%
##########################
####### feature 0 ########
##########################

### increase threshold until find one basis that has much lower loss than other basis
# ind_num0, theta0_ = select_features_of_multi_trajectory(theta0, sol_deriv0, threshold=threshold1)
# i_min0, i0_list, dist0 = select_diff_feature_idx(theta0_, sol_deriv0, n=1)

i_min0_list, i0_list, n0_list, threshold0_list = [], [], [], []
dist0_list, dist0_list_ = [], []
# for threshold0_ in np.arange(1e-2, threshold0+1e-10, 1e-2):
for threshold0_ in [threshold0]:
    ### increase threshold until find one a clear i_min0
    ind_num0, theta0_ = select_features_of_multi_trajectory(theta0, sol_deriv0, threshold0_)
    for n_ in range(0, min(len(ind_num0)+1, 3)):
        # i_min0, i_val0, dist0 = select_diff_feature_idx(theta0_, sol_deriv0, n=n_)
        i_min0, i0, dist0 = select_diff_feature_idx(theta0_, sol_deriv0, n=n_)

        # dist0_list.append(dist0)
        dist0 = np.sort(dist0)
        if dist0.shape[0]<=1:
            n0_list.append(n_)
            threshold0_list.append(threshold0_)
            i_min0_list.append(ind_num0[i_min0])
            i0_list.extend(i0)
            dist0_list_.extend(dist0)

            dist0_list.append(dist0[0])
            #break
        else:
            n0_list.append(n_)
            threshold0_list.append(threshold0_)
            i_min0_list.append(ind_num0[i_min0])
            i0_list.extend(i0)
            dist0_list_.extend(dist0)
            
            dist_ratio = dist0[0]#/dist0[1]
            dist0_list.append(dist_ratio)
            print(f'dist ratio: {dist_ratio}')
            # if dist_ratio<=1e-2:
            #     break

# idx_dist = [[] for _ in range(monomial_num)]
# for idx, dist in zip(i0_list,dist0_list_):
#     for idx_ in idx:
#         idx_dist[idx_].append(dist)


# for i in range(monomial_num):
#     plt.plot(idx_dist[i],'o', markersize=1, label=i)
# plt.legend()

# for i in range(monomial_num):
#     print(len(idx_dist[i]))




ii0 = np.argmin(dist0_list)
ind_num0, theta0_ = select_features_of_multi_trajectory(theta0, sol_deriv0, threshold0_list[ii0])
i_min0, i0, dist0 = select_diff_feature_idx(theta0_, sol_deriv0, n=n0_list[ii0])
        
Theta0, DXdt0 = get_one_hot_theta(theta0_, sol_deriv0, i_min0)
Xi0 = SLS(Theta0, DXdt0, threshold0, alpha=alpha).squeeze()


num_diff = len(i_min0)

num_feature = theta0_[0].shape[1]
i_same = np.ones(num_feature, dtype=bool)

i_diff = i_min0
i_same[i_diff] = False    
    

Xi0_ = np.zeros([len(ind_num0), num_traj])
for j in range(num_traj):
    Xi0_[i_min0,[j]] = Xi0[j*num_diff:(j+1)*num_diff]
    
    Xi0_[i_same,[j]] = Xi0[num_traj*num_diff:]
    # Xi0_[:,[j]] = np.r_[Xi0[num_traj:i_min0+num_traj], Xi0[[j]], Xi0[i_min0+num_traj:]]
    
    dist_norm = MSE(theta0_[j]@Xi0_[:,[j]]-sol_deriv0[j])
    print(f'feature 0 for trajectory {j}: {dist_norm:3E}')
    

##########################
####### feature 1 ########
##########################

### increase threshold until find one basis that has much lower loss than other basis
# ind_num1, theta1_ = select_features_of_multi_trajectory(theta1, sol_deriv1, threshold=threshold2)
# i_min1, i_val1, dist1 = select_diff_feature_idx(theta1_, sol_deriv1, n=1)
    
i1_list, n1_list, threshold1_list = [], [], []
dist1_list = []
for threshold1_ in np.arange(1e-2, threshold1+1e-10, 1e-2):
# for threshold1_ in [threshold1]:
    ### increase threshold until find one a clear i_min0
    ind_num1, theta1_ = select_features_of_multi_trajectory(theta1, sol_deriv1, threshold1_)
    for n_ in range(0, min(len(ind_num1)+1, 3)):
        i_min1, i1, dist1 = select_diff_feature_idx(theta1_, sol_deriv1, n=n_)
        
        # dist1_list.append(dist1)
        dist1 = np.sort(dist1)
        if dist1.shape[0]<=1:
            n1_list.append(n_)
            threshold1_list.append(threshold1_)
            dist1_list.append(dist1[0])
            break
        
        n1_list.append(n_)
        threshold1_list.append(threshold1_)
        i1_list.append(i1)
        
        dist_ratio = dist1[0]/dist1[1]
        dist1_list.append(dist_ratio)
        print(f'dist ratio: {dist_ratio}')
        # if dist_ratio<=1e-2:
        #     break
ii1 = np.argmin(dist1_list)
ind_num1, theta1_ = select_features_of_multi_trajectory(theta1, sol_deriv1, threshold1_list[ii1])
i_min1, i1, dist1 = select_diff_feature_idx(theta1_, sol_deriv1, n=n1_list[ii1])
        
Theta1, DXdt1 = get_one_hot_theta(theta1_, sol_deriv1, i_min1)   
Xi1 = SLS(Theta1, DXdt1, threshold1, alpha=alpha).squeeze()


num_diff = len(i_min1)

num_feature = theta1_[0].shape[1]
i_same = np.ones(num_feature, dtype=bool)

i_diff = i_min1
i_same[i_diff] = False     

Xi1_ = np.zeros([len(ind_num1), num_traj])
for j in range(num_traj):
    Xi1_[i_min1,[j]] = Xi1[j*num_diff:(j+1)*num_diff]
    
    Xi1_[i_same,[j]] = Xi1[num_traj*num_diff:]
    # Xi1_[:,[i]] = np.r_[Xi1[num_traj:i_min1+num_traj], Xi1[[i]], Xi1[i_min1+num_traj:]]

    dist_norm = MSE(theta1_[i]@Xi1_[:,[i]]-sol_deriv1[i])
    print(f'feature 1 for trajectory {j}: {dist_norm:3E}')


print('*'*50)
# print(f'feature 0: \n {Xi0_} \n {ind_num0}')
# print(f'feature 1: \n {Xi1_} \n {ind_num1}')
print(f'real0: {real0}')
print(f'real1: {real1}')
print(f'feature 0 with n={n0_list[ii0]}: \n {Xi0_[Xi0_.any(axis=1)]} \n {np.array([monomial_name[i] for i in ind_num0[Xi0_.any(axis=1)]])}')
print(f'feature 1 with n={n1_list[ii1]}: \n {Xi1_[Xi1_.any(axis=1)]} \n {np.array([monomial_name[i] for i in ind_num1[Xi1_.any(axis=1)]])}')


### compare to pysindy
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
    for i in range(len(a)):
        sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i])
    
        # t_ = np.linspace(0,t[-1])
        # sol0_, sol0_deriv_ = data_interp(t_, t, sol_[:,0])
        # sol1_, sol1_deriv_ = data_interp(t_, t, sol_[:,1])
        # sol_ = np.c_[sol0_,sol1_]
        # sol_deriv_ = np.c_[sol0_deriv_, sol1_deriv_]
        
        model.fit(sol_, t=t_)#, ensemble=True, quiet=True)
        model.print()
        # model.coefficients()
        
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol_deriv_, 1e-1))
        
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
        sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i])
    
        model.fit(sol_, t=t_)
        model.print()
        
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol_deriv_, 1e-1))
        
        
        