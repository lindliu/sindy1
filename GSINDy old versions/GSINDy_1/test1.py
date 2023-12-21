#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:42:52 2023

@author: dliu
"""



from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ridge_regression, LinearRegression, SGDRegressor
import copy

MSE = lambda x: (np.array(x)**2).mean()
# MSE = lambda x: np.linalg.norm(x)

def SLS(Theta, DXdt, threshold):
    n_feature = DXdt.shape[1]
    Xi = ridge_regression(Theta,DXdt, alpha=0.05).T
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
            Xi[biginds,ind] = ridge_regression(Theta[:,biginds], DXdt[:,ind], alpha=.05).T
            # Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],DXdt[:,ind], rcond=None)[0]
            # Xi[biginds,ind] = solve_minnonzero(Theta[:,biginds],DXdt[:,ind])
    ## print(Xi)
    ## np.mean((Theta@Xi-DXdt)**2)
    
    # reg = LinearRegression(fit_intercept=False)
    # ind_ = np.abs(Xi.T) > 1e-10
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
    # # Xi[np.abs(Xi)<threshold] = 0
    # Xi[np.abs(Xi)<1e-3] = 0


    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-10
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T
    # Xi[np.abs(Xi)<threshold] = 0
    Xi[np.abs(Xi)<1e-3] = 0


    # sgd = LinearRegression(fit_intercept=False)
    # biginds = np.abs(Xi)>1e-10
    # Xi[biginds[:,0],0] = sgd.fit(Theta[:,biginds[:,0]],DXdt[:,0]).coef_
    # Xi[biginds[:,1],1] = sgd.fit(Theta[:,biginds[:,1]],DXdt[:,1]).coef_
    # Xi[np.abs(Xi)<threshold] = 0
    # # print(Xi)
    return Xi

def func1(x, t, a):
    """
    P179, differential equations, dynamical systems, and an introduction to chaos
    """
    x1, x2 = x
    dxdt = [a + x1**2, -x2]
    return dxdt

def func2(x, t, a):
    """
    P179, differential equations, dynamical systems, and an introduction to chaos
    """
    x1, x2 = x
    dxdt = [.2 + a*x1**2, -x2]
    return dxdt

def func3(x, t, a):
    """
    P81, Hopf Bifurcation, differential equations, dynamical systems, and an introduction to chaos
    
    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [a*x1**2-x2-x1*(x1**2+x2**2), x1+a*x2-x2*(x1**2+x2**2)]
    return dxdt

def func4(x, t, a):
    """
    SINDy-SA: prey-predator model
    
    dxdt = c0*x + c1*x*y
    dydt = c2*y + c3*x*y
    
    c0=1, c1=-1, c2=-1, c3=1.
    x0, y0 = 4, 1
    
    [0, c0, 0,  0, c1, 0, ...]
    [0, 0,  c2, 0, c3, 0, ...]
    """
    # c0, c1, c2, c3 = 1, -1, -1, 1
    c1, c2 = -1, -1
    c0 = a
    c3 = a
    x1, x2 = x
    dxdt = [c0*x1 + c1*x1*x2, c2*x2 + c3*x1*x2]
    return dxdt

def func5(x, t, a):
    """
    SINDy-SA: tumor growth model
    
    dxdt = c0*x + c1*x^2
    dydt = -y
    
    c0 = .028, c1=-3.305e-6
    x0, y0 = 245.185, 1
    """
    c0 = a
    c1 = -3.305e-6
    x1, x2 = x
    dxdt = [c0*x1 + c1*x1**2, -x2]
    return dxdt

def func6(x, t, a):
    """
    SINDy-SA: Pendulum motion model
    
    dxdt = 1*y
    dydt = c0*y + c1*sin(x)
    
    c0 = -.25, c1=-5.0
    x0, y0 = pi-.1, 0
    """
    
    c0 = a
    c1 = -5
    x1, x2 = x
    dxdt = [x2, c0*x2+c1*np.sin(x1)]
    return dxdt

def func7(x, t, a):
    """
    SINDy-SA: Pendulum motion model
    
    dxdt = 1*y
    dydt = c0*y + c1*sin(x)
    
    c0 = -.25, c1=-5.0
    x0, y0 = pi-.1, 0
    """
    
    c0 = -.25
    c1 = a
    x1, x2 = x
    dxdt = [x2, c0*x2+c1*np.sin(x1)]
    return dxdt

def monomial_poly(x):
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    return np.c_[np.ones([x1.shape[0],1]), \
                 x1, x2, \
                 x1**2, x1*x2, x2**2, \
                 x1**3, x1**2*x2, x1*x2**2, x2**3, \
                 x1**4, x1**3*x2, x1**2*x2**2, x1*x2**3, x2**4, \
                 x1**5, x1**4*x2, x1**3*x2**2, x1**2*x2**3, x1*x2**4, x2**5]#, np.sin(x1)]

monomial_poly_name = np.array(['1', 'x', 'y', 'x^2', 'xy','y^2','x^3','x^2y','xy^2','y^3',\
                      'x^4','x^3y','x^2y^2','xy^3','y^4','x^5','x^4y','x^3y^2','x^2y^3','xy^4','y^5'])

def monomial_trig(x):
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    return np.c_[np.ones([x1.shape[0],1]), x1, x2, \
                 np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2)]
        
monomial_trig_name = np.array(['1', 'x', 'y', 'sin(x)', 'cos(x)', 'sin(y)', 'cos(y)'])

# def get_sol_deriv(func, x0, t, a, step=1):
#     dt = t[1]-t[0]
#     sol1 = odeint(func, x0, t, args=(a,))
#     sol1 = sol1 + .01*np.random.randn(*sol1.shape)
#     sol1_deriv = (sol1[step:,:]-sol1[:-step,:])/dt

#     sol1 = sol1[:-step, :]
#     t = t[:-step]
#     return sol1, sol1_deriv, t

# def get_sol_deriv(func, x0, t, a, step=1):
#     sol1 = odeint(func, x0, t, args=(a,))
#     # sol1 = sol1 + .01*np.random.randn(*sol1.shape)
    
#     from pysindy.differentiation import FiniteDifference
#     fd = FiniteDifference()
#     sol1_deriv = fd._differentiate(sol1, t)

#     return sol1, sol1_deriv, t

def get_sol_deriv(func, x0, t, a, step=1):
    ###https://github.com/florisvb/PyNumDiff/blob/master/examples/1_basic_tutorial.ipynb
    sol1 = odeint(func, x0, t, args=(a,))
    # sol1 = sol1 + .01*np.random.randn(*sol1.shape)
    
    sol1_deriv = np.zeros_like(sol1)
    dt = t[1]-t[0]
    
    import pynumdiff
    
    for i in range(sol1.shape[1]):
        # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol1[:,i], dt)
        x_hat, dxdt_hat = pynumdiff.finite_difference.second_order(sol1[:,i], dt)
        # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol1[:,i], dt, params=[50], options={'iterate': True})
        # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(sol1[:,i], dt, params=[20], options={'iterate': False})
        # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.friedrichsdiff(sol1[:,i], dt, params=10, options={'iterate': False})
        # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(sol1[:,i], dt, params=[3, 0.09], options={'iterate': False})
        # x_hat, dxdt_hat = pynumdiff.linear_model.spectraldiff(sol1[:,i], dt, params=[0.05])
        # x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(sol1[:,i], dt, params=[2, 10, 10])

        sol1_deriv[:,i] = dxdt_hat

    return sol1, sol1_deriv, t


def SLS_multi(theta0_, sol_deriv0, threshold):
    num_traj = len(theta0_)
    ind = np.zeros(theta0_[0].shape[1], dtype=bool)
    for i in range(num_traj): 
        Xi_ = SLS(theta0_[i], sol_deriv0[i], threshold).squeeze()
        ind = np.logical_or(ind!=0,Xi_!=0)

    theta0_ = [ele[:,ind] for ele in theta0_]
    
    return ind, theta0_

def get_one_hot_theta(theta0_, sol_deriv0, feature_idx=None):
    if feature_idx is None:
        return theta0_, sol_deriv0
    
    i = feature_idx
    num_traj = len(theta0_)
    Theta_ = []
    for j in range(num_traj): ###each trajectory
        head = np.zeros([theta0_[j].shape[0], num_traj])
        head[:,[j]] = theta0_[j][:,[i]]
        
        theta_telda_ = np.c_[head, theta0_[j][:,:i], theta0_[j][:,i+1:]]
        
        Theta_.append(theta_telda_)
    Theta = np.vstack(Theta_)
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

def select_diff_feature_idx(theta0_, sol_deriv0):
    ### select proper feature index by one hot matrix
    num_feature = theta0_[0].shape[1]
    sgd = LinearRegression(fit_intercept=False)
    
    dist = []
    i_min0, i_val0 = None, float('Inf')
    for i in range(num_feature):
        Theta0, DXdt0 = get_one_hot_theta(theta0_, sol_deriv0, i)
        Xi0 = sgd.fit(Theta0, DXdt0).coef_.T
        
        distance0 = MSE(Theta0@Xi0-DXdt0)
        # print(f'L2 distance of {i}: {distance0:3E}')
        dist.append(distance0)
        
        if i_val0>distance0:
            i_val0 = distance0
            i_min0 = i
            
    print(f'index {i_min0} has lowest error')
    return i_min0, dist


# dt = .01   ## 0,3
# t = np.arange(0,1.2,dt)
# x0 = [.5, 1]
# a = [.15, .3, .5]
# func = func1
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + x^2"
# real1 = "y'=-y"
# threshold1 = 1e-1
# threshold2 = 1e-1

# dt = .01
# t = np.arange(0,2,dt)
# x0 = [.5, 1]
# a = [.25, .5]
# func = func2
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=.2 + a*x^2"
# real1 = "y'=-y"
# threshold1 = 1e-1
# threshold2 = 1e-1

# dt = .01      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,8,dt)
# x0 = [.5, 1]
# a = [.2, .4, .6]
# func = func3
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"    
# threshold1 = 1e-1
# threshold2 = 1e-1

# dt = .01    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [4, 1]
# a = [.7, 1]
# func = func4
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"    
# threshold1 = 1e-1
# threshold2 = 1e-1

# dt = .005
# t = np.arange(0,300,dt)
# x0 = [245.185, 100]
# a = [.028, .033]
# func = func5
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# threshold1 = 1e-7
# threshold2 = 1e-7

# dt = .1
# t = np.arange(0,3.2,dt)
# x0 = [np.pi-.1, 0]
# a = [-.25, -.35]
# func = func6
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=a*y-5sin(x)"
# threshold1 = 1e-1
# threshold2 = 1e-1

dt = .1
t = np.arange(0,16,dt)
x0 = [np.pi-.1, 0]
# a = [-5, -6]
a = [-.15, -.2]
func = func7
monomial = monomial_trig
monomial_name = monomial_trig_name
real0 = "x'=y"
real1 = "y'=-0.25y+a*sin(x)"
threshold1 = 1e-1
threshold2 = 1e-1



# num_feature = len(x0)

num_traj = len(a)
sol0, theta0, sol_deriv0 = [], [], []
sol1, theta1, sol_deriv1 = [], [], []
for i in range(num_traj):
    sol_, sol1_deriv_, t_ = get_sol_deriv(func, x0, t, a[i], step=1)
    theta_ = monomial(sol_)

    sol0.append(sol_[:,[0]])
    theta0.append(theta_)
    sol_deriv0.append(sol1_deriv_[:,[0]])
    
    sol1.append(sol_[:,[1]])
    theta1.append(theta_)
    sol_deriv1.append(sol1_deriv_[:,[1]])
    
    plt.plot(t, sol_, 'o', markersize=1, label=a[i])
# plt.legend()
monomial_num = theta0[0].shape[1]

##########################
####### feature 0 ########
##########################

### increase threshold until find one basis that has much lower loss than other basis
dist0_list = []
for threshold1_ in np.arange(1e-2, 1e-1, 1e-2):
    ### increase threshold until find one a clear i_min0
    ind_num0, theta0_ = select_features_of_multi_trajectory(theta0, sol_deriv0, threshold1_)
    i_min0, dist0 = select_diff_feature_idx(theta0_, sol_deriv0)
    
    dist0_list.append(dist0)
    dist0 = np.sort(dist0)
    if dist0.shape[0]<=1:
        break
    
    if dist0[0]/dist0[1]<=1e-2:
        break
Theta0, DXdt0 = get_one_hot_theta(theta0_, sol_deriv0, i_min0)

Xi0 = SLS(Theta0, DXdt0, threshold1)
Xi0_ = np.zeros([len(ind_num0), num_traj])
for i in range(num_traj):
    Xi0_[:,[i]] = np.r_[Xi0[num_traj:i_min0+num_traj], Xi0[[i]], Xi0[i_min0+num_traj:]]
    
    dist_norm = MSE(theta0_[i]@Xi0_[:,[i]]-sol_deriv0[i])
    print(f'L2 for trajectory {i}: {dist_norm:3E}')
    

##########################
####### feature 1 ########
##########################

### increase threshold until find one basis that has much lower loss than other basis
dist1_list = []
for threshold2_ in np.arange(1e-2, 1e-1, 1e-2):
    ### increase threshold until find one a clear i_min0
    ind_num1, theta1_ = select_features_of_multi_trajectory(theta1, sol_deriv1, threshold2_)
    i_min1, dist1 = select_diff_feature_idx(theta1_, sol_deriv1)
    
    dist1_list.append(dist1)
    dist1 = np.sort(dist1)
    if dist1.shape[0]<=1:
        break
    
    if dist1[0]/dist1[1]<=1e-2:
        break
Theta1, DXdt1 = get_one_hot_theta(theta1_, sol_deriv1, i_min1)    

Xi1 = SLS(Theta1, DXdt1, threshold2)
Xi1_ = np.zeros([len(ind_num1), num_traj])
for i in range(num_traj):
    Xi1_[:,[i]] = np.r_[Xi1[num_traj:i_min1+num_traj], Xi1[[i]], Xi1[i_min1+num_traj:]]

    dist_norm = MSE(theta0_[i]@Xi0_[:,[i]]-sol_deriv0[i])
    print(f'L2 for trajectory {i}: {dist_norm:3E}')


print('*'*50)
# print(f'feature 0: \n {Xi0_} \n {ind_num0}')
# print(f'feature 1: \n {Xi1_} \n {ind_num1}')
print(f'real0: {real0}')
print(f'real1: {real1}')
print(f'feature 0: \n {Xi0_[Xi0_.any(axis=1)]} \n {np.array([monomial_name[i] for i in ind_num0[Xi0_.any(axis=1)]])}')
print(f'feature 1: \n {Xi1_[Xi1_.any(axis=1)]} \n {np.array([monomial_name[i] for i in ind_num1[Xi1_.any(axis=1)]])}')


### compare to pysindy
import pysindy as ps

if func.__name__ not in ['func6', 'func7']:
    model = ps.SINDy(feature_names=["x", "y"])
    for i in range(len(a)):
        sol_, sol1_deriv_, t_ = get_sol_deriv(func, x0, t, a[i])
    
        model.fit(sol_, t=t)
        model.print()
        # model.coefficients()
        
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol1_deriv_, 1e-1))
        
else:
    from pysindy.feature_library import FourierLibrary, CustomLibrary
    from pysindy.feature_library import GeneralizedLibrary
    functions = [lambda x : 1, lambda x : x, lambda x: np.sin(x), lambda x: np.cos(x)]
    lib_custom = CustomLibrary(library_functions=functions)
    lib_fourier = FourierLibrary()
    # lib_generalized = GeneralizedLibrary([lib_custom, lib_fourier])
    lib_generalized = GeneralizedLibrary([lib_custom])
    # x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    # lib_generalized.fit(x)
    # lib_generalized.transform(x)
    
    model = ps.SINDy(feature_names=["x", "y"],feature_library=lib_generalized)
    for i in range(len(a)):
        sol_, sol1_deriv_, t_ = get_sol_deriv(func, x0, t, a[i])
    
        model.fit(sol_, t=t_)
        model.print()
        
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol1_deriv_, 1e-1))
        
        
        