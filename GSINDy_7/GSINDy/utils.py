#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 01:32:07 2023

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt


#################################
######## functions ##############
#################################
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

def func12_(x, t, a, b):
    """
    P179, differential equations, dynamical systems, and an introduction to chaos
    """
    x1, x2 = x
    dxdt = [a + b*x1**2, -x2]
    return dxdt

def func3(x, t, a):
    """
    P81, differential equations, dynamical systems, and an introduction to chaos
    Modified Lotka-Voltera
    
    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [a*x1**2-x2-x1*(x1**2+x2**2), x1+a*x2-x2*(x1**2+x2**2)]
    return dxdt

def  func3_(x, t, a, b):
    """
    P81, differential equations, dynamical systems, and an introduction to chaos
    Modified Lotka-Voltera


    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [b*x2+a*x1**2-x1**3-x1*x2**2, x1+a*x2+b*x2*x1**2-x2**3]
    return dxdt

def  func3__(x, t, a, b, c):
    """
    P81, differential equations, dynamical systems, and an introduction to chaos
    Modified Lotka-Voltera
    
    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [b*x2+a*x1**2+c*x1**3-x1*x2**2, x1+a*x2+b*x2*x1**2+c*x2**3]
    return dxdt

def func4(x, t, a):
    """
    classical Lotka-Voltera
    
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

def func4_(x, t, a, b):
    """
    classical Lotka-Voltera
    
    dxdt = c0*x + c1*x*y
    dydt = c2*y + c3*x*y
    
    c0=1, c1=-1, c2=-1, c3=1.
    x0, y0 = 4, 1
    
    [0, c0, 0,  0, c1, 0, ...]
    [0, 0,  c2, 0, c3, 0, ...]
    """
    # c0, c1, c2, c3 = 1, -1, -1, 1
    c1 = b
    c2 = b
    c0 = a
    c3 = a
    x1, x2 = x
    dxdt = [c0*x1 + c1*x1*x2, c2*x2 + c3*x1*x2]
    return dxdt

def func5(x, t, a, b):
    """
    van der pol
    
    dxdt = 5*(x - y + c0*x^3), c0=-1/3
    dydt = c1*x,               c1=0.2
    """
    c0 = a
    c1 = b
    x1, x2 = x
    dxdt = [5*(x1 - x2 + c0*x1**3), c1*x1]
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

def func8(x, t, a, b):
    """
    Brusselator
    
    dxdt = a-4x+x^2y,  a=1
    dydt = bx-x^2y,    b=3
    
    """
    x1, x2 = x
    dxdt = [a-4*x1+x1**2*x2, b*x1-x1**2*x2]
    return dxdt

def func9(x, t, a, d, b, c):
    """
    Lorenz
    
    dxdt = a(y-x),       a=10
    dydt = x(b-z)-y,    b=28
    dzdt = xy - cz,   c=8/3
    """
    x1, x2, x3 = x
    dxdt = [d*x2+a*x1, x1*(b-x3)-x2, x1*x2+c*x3]
    return dxdt



##########################################
######### obtain basis functions #########
##########################################
def get_theta(x, basis_functions):
    """
    
    Parameters
    ----------
    x : numpy
        dimension 0: number of time steps
        dimension 1: number of features
    basis_functions : list
        list of basis functions.

    Returns
    -------
    theta : numpy
        dimension 0: number of time steps
        dimension 1: number of basis functions
        
    """
    num_step, num_feature = x.shape
    num_basis = len(basis_functions)
    theta = np.zeros([num_step, num_basis])
    for i in range(num_step):
        if num_feature==1:
            x1 = x[i,0]
            theta_row = [f(x1) for f in basis_functions]
        if num_feature==2:
            x1 = x[i,0]
            x2 = x[i,1]
            theta_row = [f(x1,x2) for f in basis_functions]
        if num_feature==3:
            x1 = x[i,0]
            x2 = x[i,1]
            x3 = x[i,2]
            theta_row = [f(x1,x2,x3) for f in basis_functions]
            
        theta[i,:] = theta_row
        
    return theta

############ mix basis functions ################
basis_functions_mix0 = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: 1/x, 
            lambda x,y: np.sin(x), lambda x,y: np.cos(x), lambda x,y: np.exp(x)])
    
basis_functions_mix1 = np.array([lambda x,y: 1, \
             lambda x,y: x, lambda x,y: y, \
             lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
             lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
             lambda x,y: y**4, lambda x,y: 1/y, 
             lambda x,y: np.sin(y), lambda x,y: np.cos(y), lambda x,y: np.exp(y)])

basis_functions_name_mix0 = [lambda x,y: '1', \
            lambda x,y: 'x', lambda x,y: 'y', \
            lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
            lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
            lambda x,y: 'x^4', lambda x,y: '1/x', \
            lambda x,y: 'sin(x)', lambda x,y: 'cos(x)', lambda x,y: 'exp(x)']

basis_functions_name_mix1 = [lambda x,y: '1', \
            lambda x,y: 'x', lambda x,y: 'y', \
            lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
            lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
            lambda x,y: 'y^4', lambda x,y: '1/y', \
            lambda x,y: 'sin(y)', lambda x,y: 'cos(y)', lambda x,y: 'exp(y)']
    
    
############ polynomial basis functions ################
basis_functions_poly_5 = np.array([lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
            lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5])
basis_functions_name_poly_5 = [lambda x,y: '1', \
        lambda x,y: 'x', lambda x,y: 'y', \
        lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
        lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
        lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4', \
        lambda x,y: 'x^5', lambda x,y: 'x^4y', lambda x,y: 'x^3y^2', lambda x,y: 'x^2y^3', lambda x,y: 'xy^4', lambda x,y: 'y^5']


############ basis functions for Lorenz ################
basis_functions_Lorenz = np.array([lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: x**2*y, lambda x,y,z: x*y**2, lambda x,y,z: x*z**2, lambda x,y,z: y**2*z, lambda x,y,z: y*z**2,  \
        lambda x,y,z: x**4,  lambda x,y,z: y**4,  lambda x,y,z: z**4, \
        lambda x,y,z: 1/x, lambda x,y,z: 1/y, lambda x,y,z: 1/z, \
        lambda x,y,z: np.exp(x), lambda x,y,z: np.exp(y), lambda x,y,z: np.exp(z), \
        lambda x,y,z: np.sin(x), lambda x,y,z: np.sin(y), lambda x,y,z: np.sin(z), \
        lambda x,y,z: np.cos(x), lambda x,y,z: np.cos(y), lambda x,y,z: np.cos(z)])
basis_functions_name_Lorenz = np.array([lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'xy^2', lambda x,y,z: 'xz^2', lambda x,y,z: 'y^2z', lambda x,y,z: 'yz^2',  \
        lambda x,y,z: 'x^4',  lambda x,y,z: 'y^4',  lambda x,y,z: 'z^4', \
        lambda x,y,z: '1/x', lambda x,y,z: '1/y', lambda x,y,z: '1/z', \
        lambda x,y,z: 'exp(x)', lambda x,y,z: 'exp(y)', lambda x,y,z: 'exp(z)', \
        lambda x,y,z: 'sin(x)', lambda x,y,z: 'sin(y)', lambda x,y,z: 'sin(z)', \
        lambda x,y,z: 'cos(x)', lambda x,y,z: 'cos(y)', lambda x,y,z: 'cos(z)'])
    
    
    
    


    
    
########################################
##### data generation ##################
########################################
def data_generator(func, x0, t, a, real_list, num=None, num_split=None):
    sol_org_list = get_multi_sol(func, x0, t, a)
    
    if num==1:    
        ll = t.shape[0]//num_split
        # idx_init = list(range(0,ll*num_split,ll))
        
        sol_org_list_ = list(sol_org_list[0][:num_split*ll,:].reshape([num_split,ll,-1]))
        t_ = t[:ll]
        x0_ = [list(sub[0]) for sub in sol_org_list_]
        a_ = [a[0] for _ in range(num_split)]

        t, x0, a, sol_org_list = t_, x0_, a_, sol_org_list_

    ### generate data ###
    num_traj = len(a)
    num_feature = len(x0[0])
    
    ### plot data ###
    plot_mult_traj(sol_org_list, t, a, real_list)
    return t, x0, a, sol_org_list, num_traj, num_feature


def plot_mult_traj(sol_org_list, t, a, real_list):
    fig, ax = plt.subplots(1,1,figsize=[6,3])
    for i in range(len(sol_org_list)):
        ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
    ax.legend()
    ax.text(1, .95, f'${real_list[0]}$', fontsize=12)
    ax.text(1, .8, f'${real_list[1]}$', fontsize=12)
    
    
    
from scipy.integrate import odeint
def ode_solver(func, x0, t, a):
    sol = odeint(func, x0, t, args=a)
    return sol, t

def get_multi_sol(func, x0, t, a):
    num_traj = len(a)
    sol_org_list_ = []
    for i in range(num_traj):
        sol_, _ = ode_solver(func, x0[i], t, a[i])
        sol_org_list_.append(sol_)
    
    return sol_org_list_

def get_deriv(sol, t, deriv_spline=True):
    if deriv_spline:
        from scipy import interpolate
        f = interpolate.interp1d(t, sol.T, kind='cubic')
        fd1 = f._spline.derivative(nu=1)
        return f(t).T, fd1(t), t
    else:
        ###https://github.com/florisvb/PyNumDiff/blob/master/examples/1_basic_tutorial.ipynb
        import pynumdiff
        sol_deriv = np.zeros_like(sol)
        dt = t[1]-t[0]
        for i in range(sol.shape[1]):
            # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol1[:,i], dt)
            x_hat, dxdt_hat = pynumdiff.finite_difference.second_order(sol[:,i], dt)
            # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol[:,i], dt, params=[50], options={'iterate': True})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(sol[:,i], dt, params=[20], options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.friedrichsdiff(sol[:,i], dt, params=10, options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(sol[:,i], dt, params=[3, 0.09], options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.linear_model.spectraldiff(sol[:,i], dt, params=[0.05])
            # x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(sol[:,i], dt, params=[2, 10, 10])
    
            sol_deriv[:,i] = dxdt_hat
        
        return sol, sol_deriv, t