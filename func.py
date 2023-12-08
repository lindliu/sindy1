#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:51:23 2023

@author: do0236li
"""
import numpy as np

from sklearn.linear_model import ridge_regression, LinearRegression, SGDRegressor

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
    Hopf Bifurcation
    
    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [a*x1**2-x2-x1*(x1**2+x2**2), x1+a*x2-x2*(x1**2+x2**2)]
    return dxdt

def func3_(x, t, a, b):
    """
    P81, differential equations, dynamical systems, and an introduction to chaos
    Hopf Bifurcation
    
    dxdt = a*x^2 - y - x*(x^2 + y^2) =      -y + a*x^2 - x^3 - xy^2
    dydt = x + a*y - y*(x^2 + y^2)   = x + a*y                      - x^2y - y^3
    
    [0, 0, -1, a, 0, 0, -1,  0, -1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, a,  0, 0, 0,  0, -1,  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    x1, x2 = x
    dxdt = [b*x2+a*x1**2-x1**3-x1*x2**2, x1+a*x2+b*x2*x1**2-x2**3]
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

def func4_(x, t, a, b):
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
    c1 = b
    c2 = b
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

    
from scipy.integrate import odeint
def get_sol_deriv(func, x0, t, a, deriv_spline=True):
    ###https://github.com/florisvb/PyNumDiff/blob/master/examples/1_basic_tutorial.ipynb
    sol1 = odeint(func, x0, t, args=a)
    # sol1 = sol1 + .01*np.random.randn(*sol1.shape)
    
    sol1_deriv = np.zeros_like(sol1)
    dt = t[1]-t[0]
    
    if deriv_spline:
        from scipy import interpolate
        f = interpolate.interp1d(t, sol1.T, kind='cubic')
        fd1 = f._spline.derivative(nu=1)
        return f(t).T, fd1(t), t
    else:
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


# def get_sol_deriv(func, x0, t, a, step=1):
#     sol1 = odeint(func, x0, t, args=(a,))
#     # sol1 = sol1 + .01*np.random.randn(*sol1.shape)
    
#     from pysindy.differentiation import FiniteDifference
#     fd = FiniteDifference()
#     sol1_deriv = fd._differentiate(sol1, t)

#     return sol1, sol1_deriv, t

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
                 np.sin(x1), np.sin(x2), np.cos(x1), np.cos(x2)]
        
monomial_trig_name = np.array(['1', 'x', 'y', 'sin(x)', 'sin(y)',  'cos(x)', 'cos(y)'])


