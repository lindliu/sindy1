#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 01:32:07 2023

@author: dliu
"""

import numpy as np
import matplotlib.pyplot as plt

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

def func9(x, t, a, b, c):
    """
    Lorenz
    
    dxdt = a(y-x),       a=10
    dydt = x(b-z)-y,    b=28
    dzdt = xy - cz,   c=8/3
    """
    x1, x2, x3 = x
    dxdt = [a*(x2-x1), x1*(b-x3)-x2, x1*x2-c*x3]
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

def monomial_lorenz(x):
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    x3 = x[:,[2]]
    return np.c_[np.ones([x1.shape[0],1]), \
                 x1, x2, x3, \
                 x1**2, x2**2, x3**2, x1*x2, x1*x3, x2*x3, \
                 x1**3, x2**3, x3**3, x1**2*x2, x1*x2**2, x1*x3**2, x2**2*x3, x2*x3**2, \
                 x1**4, x2**4, x3**4, \
                 1/x1, 1/x2, 1/x3, \
                 np.exp(x1), np.exp(x2), np.exp(x3),\
                 np.sin(x1), np.sin(x2), np.sin(x3),\
                 np.cos(x1), np.cos(x2), np.cos(x3)]

monomial_lorenz_name = np.array(['1', 'x', 'y', 'z', 'x^2', 'y^2', 'z^2', 'xy', 'xz', 'yz', \
                                 'x^3', 'y^3', 'z^3', 'x^2y', 'xy^2', 'xz^2', 'y^2z', 'yz^2',  \
                                 'x^4', 'y^4', 'z^4', \
                                 '1/x','1/y','1/z', \
                                 'exp(x)','exp(y)','exp(z)', \
                                 'sin(x)','sin(y)','sin(z)', \
                                 'cos(x)', 'cos(y)', 'cos(z)'])
        
def monomial_trig(x):
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    return np.c_[np.ones([x1.shape[0],1]), x1, x2, \
                 np.sin(x1), np.sin(x2), np.cos(x1), np.cos(x2)]
        
monomial_trig_name = np.array(['1', 'x', 'y', 'sin(x)', 'sin(y)',  'cos(x)', 'cos(y)'])

def monomial_all(x):
    x1 = x[:,[0]]
    x2 = x[:,[1]]
    return np.c_[np.ones([x1.shape[0],1]), \
                 x1, x2, \
                 x1**2, x1*x2, x2**2, \
                 x1**3, x1**2*x2, x1*x2**2, x2**3, \
                 x1**4, x2**4, \
                 1/x1, 1/x2, \
                 np.sin(x1), np.sin(x2), np.cos(x1), np.cos(x2), np.exp(x1), np.exp(x2)]

monomial_all_name = np.array(['1', 'x', 'y', 'x^2', 'xy','y^2','x^3','x^2y','xy^2','y^3',\
                      'x^4','y^4','1/x','1/y',\
                      'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'exp(x)', 'exp(y)'])


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