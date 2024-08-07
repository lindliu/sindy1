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
def func_Lotka_Voltera(x, t, a, b):
    """
    classical Lotka-Voltera
    
    dxdt = a*x + b*x*y
    dydt = b*y + a*x*y
    
    a=1, b=-1
    x0, y0 = 4, 1
    """
    x1, x2 = x
    dxdt = [a*x1 + b*x1*x2, 
            b*x2 + a*x1*x2]
    return dxdt

def  func_M_Lotka_Voltera(x, t, a, b, c):
    """
    P81, differential equations, dynamical systems, and an introduction to chaos
    Modified Lotka-Voltera
    
    dxdt =  a*x - b*y + c*x(x^2 + y^2)
    dydt =  b*x + a*y + c*y(x^2 + y^2)
    """
    x1, x2 = x
    dxdt = [a*x1-b*x2+c*x1**3+c*x1*x2**2, 
            b*x1+a*x2+c*x2*x1**2+c*x2**3]
    return dxdt

def func_Brusselator(x, t, a, b):
    """
    Brusselator
    
    dxdt = a-4x+x^2y,  a=1
    dydt = bx-x^2y,    b=3
    
    """
    x1, x2 = x
    dxdt = [a-4*x1+x1**2*x2, 
            b*x1-x1**2*x2]
    return dxdt

def func_Van_der_Pol(x, t, a, b):
    """
    van der pol
    
    dxdt = 5*(x - y + a*x^3), c0=-1/3
    dydt = b*x,               c1=0.2
    """
    x1, x2 = x
    dxdt = [5*(x1 - x2 + a*x1**3), 
            b*x1]
    return dxdt

def func_Lorenz(x, t, a, b, c, d):
    """
    Lorenz
    
    dxdt = by+ax,       a=10
    dydt = x(c-z)-y,    b=28
    dzdt = xy + dz,   c=8/3
    """
    x1, x2, x3 = x
    dxdt = [b*x2+a*x1, 
            x1*(c-x3)-x2, 
            x1*x2+d*x3]
    return dxdt


def func_Pendulum(x, t, a, b):
    """
    SINDy-SA: Pendulum motion model
    
    dxdt = 1*y
    dydt = c0*y + c1*sin(x)
    
    c0 = -.25, c1=-5.0
    x0, y0 = pi-.1, 0
    """
    x1, x2 = x
    dxdt = [x2, a*x2+b*np.sin(x1)]
    return dxdt


def func_FitzHugh(x, t, a, b, c, d):
    """
    P272, differential equations, dynamical systems, and an introduction to chaos
    wikipedia
    FitzHugh–Nagumo model
    
    dxdt = x - y - 1/3*x^3 + d
    dydt = a*x + b*y + c 
    
    0<3/2(1-a)<b<1
    x0, y0 = 1, -1
    """
    
    x1, x2 = x
    dxdt = [x1 - x2 - 1/3*x1**3 + d, 
            a*x1 + b*x2 + c]
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

############ mix basis functions 2d ################
basis_functions_mix0 = np.array([lambda x,y: 1, \
        lambda x,y: x, lambda x,y: y, \
        lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
        lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
        lambda x,y: x**4, lambda x,y: 1/x, 
        lambda x,y: np.sin(x), lambda x,y: np.cos(x), lambda x,y: np.exp(x)])

basis_functions_mix1 = np.array([lambda y,x: 1, \
        lambda y,x: x, lambda y,x: y, \
        lambda y,x: x**2, lambda y,x: x*y, lambda y,x: y**2, \
        lambda y,x: x**3, lambda y,x: x**2*y, lambda y,x: x*y**2, lambda y,x: y**3, \
        lambda y,x: y**4, lambda y,x: 1/y, 
        lambda y,x: np.sin(y), lambda y,x: np.cos(y), lambda y,x: np.exp(y)])

basis_functions_mix1_ = np.array([lambda x,y: 1, \
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
        lambda x,y: 'sin(x)', lambda x,y: 'cos(x)', lambda x,y: 'e^x']

basis_functions_name_mix1 = [lambda y,x: '1', \
        lambda y,x: 'x', lambda y,x: 'y', \
        lambda y,x: 'x^2', lambda y,x: 'xy', lambda y,x: 'y^2', \
        lambda y,x: 'x^3', lambda y,x: 'x^2y', lambda y,x: 'xy^2', lambda y,x: 'y^3', \
        lambda y,x: 'y^4', lambda y,x: '1/y', \
        lambda y,x: 'sin(y)', lambda y,x: 'cos(y)', lambda y,x: 'e^y']

basis_functions_name_mix1_ = [lambda x,y: '1', \
        lambda x,y: 'x', lambda x,y: 'y', \
        lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
        lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
        lambda x,y: 'y^4', lambda x,y: '1/y', \
        lambda x,y: 'sin(y)', lambda x,y: 'cos(y)', lambda x,y: 'e^y']
    
    
############ polynomial basis functions(degree 5) 2d ################
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
############ polynomial basis functions(degree 4) 2d ################
basis_functions_poly_4 = np.array([lambda x,y: 1, \
        lambda x,y: x, lambda x,y: y, \
        lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
        lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
        lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4])

basis_functions_name_poly_4 = [lambda x,y: '1', \
        lambda x,y: 'x', lambda x,y: 'y', \
        lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
        lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
        lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4']
############ polynomial basis functions(degree 3) 2d ################
basis_functions_poly_3 = np.array([lambda x,y: 1, \
        lambda x,y: x, lambda x,y: y, \
        lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
        lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3])

basis_functions_name_poly_3 = [lambda x,y: '1', \
        lambda x,y: 'x', lambda x,y: 'y', \
        lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
        lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3']
        
        
    
############ trigonometric basis functions 2d ################
basis_functions_trig = np.array([lambda x,y: 1, \
        lambda x,y: x, lambda x,y: y, \
        lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
        lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
        lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
        lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5, \
        lambda x,y: np.sin(x), lambda x,y: np.sin(y), lambda x,y: np.cos(x), lambda x,y: np.cos(y), \
        lambda x,y: np.exp(x), lambda x,y: np.exp(y)])

basis_functions_name_trig = [lambda x,y: '1', \
        lambda x,y: 'x', lambda x,y: 'y', \
        lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
        lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
        lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4', \
        lambda x,y: 'x^5', lambda x,y: 'x^4y', lambda x,y: 'x^3y^2', lambda x,y: 'x^2y^3', lambda x,y: 'xy^4', lambda x,y: 'y^5', \
        lambda x,y: 'sin(x)', lambda x,y: 'sin(y)', lambda x,y: 'cos(x)', lambda x,y: 'cos(y)',\
        lambda x,y: 'e^x', lambda x,y: 'e^y']
    
    
    
############ mix basis functions 3d ################
basis_functions_mix0_3d = np.array([
        lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**2*y, lambda x,y,z: x**2*z, \
        lambda x,y,z: y**2*x, lambda x,y,z: y**2*z, \
        lambda x,y,z: z**2*x, lambda x,y,z: z**2*y, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: x**4, lambda x,y,z: 1/x, lambda x,y,z: np.exp(x), lambda x,y,z: np.sin(x), lambda x,y,z: np.cos(x)])
    
basis_functions_mix1_3d = np.array([
        lambda y,x,z: 1, lambda y,x,z: x, lambda y,x,z: y, lambda y,x,z: z,\
        lambda y,x,z: x**2, lambda y,x,z: y**2, lambda y,x,z: z**2, lambda y,x,z: x*y, lambda y,x,z: x*z, lambda y,x,z: y*z, \
        lambda y,x,z: x**2*y, lambda y,x,z: x**2*z, \
        lambda y,x,z: y**2*x, lambda y,x,z: y**2*z, \
        lambda y,x,z: z**2*x, lambda y,x,z: z**2*y, \
        lambda y,x,z: x**3, lambda y,x,z: y**3, lambda y,x,z: z**3,  \
        lambda y,x,z: y**4, lambda y,x,z: 1/y, lambda y,x,z: np.exp(y), lambda y,x,z: np.sin(y), lambda y,x,z: np.cos(y)])
    
basis_functions_mix2_3d = np.array([
        lambda z,x,y: 1, lambda z,x,y: x, lambda z,x,y: y, lambda z,x,y: z,\
        lambda z,x,y: x**2, lambda z,x,y: y**2, lambda z,x,y: z**2, lambda z,x,y: x*y, lambda z,x,y: x*z, lambda z,x,y: y*z, \
        lambda z,x,y: x**2*y, lambda z,x,y: x**2*z, \
        lambda z,x,y: y**2*x, lambda z,x,y: y**2*z, \
        lambda z,x,y: z**2*x, lambda z,x,y: z**2*y, \
        lambda z,x,y: x**3, lambda z,x,y: y**3, lambda z,x,y: z**3,  \
        lambda z,x,y: z**4, lambda z,x,y: 1/z, lambda z,x,y: np.exp(z), lambda z,x,y: np.sin(z), lambda z,x,y: np.cos(z)])

basis_functions_mix1_3d_ = np.array([
        lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**2*y, lambda x,y,z: x**2*z, \
        lambda x,y,z: y**2*x, lambda x,y,z: y**2*z, \
        lambda x,y,z: z**2*x, lambda x,y,z: z**2*y, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: y**4, lambda x,y,z: 1/y, lambda x,y,z: np.exp(y), lambda x,y,z: np.sin(y), lambda x,y,z: np.cos(y)])
    
basis_functions_mix2_3d_ = np.array([
        lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**2*y, lambda x,y,z: x**2*z, \
        lambda x,y,z: y**2*x, lambda x,y,z: y**2*z, \
        lambda x,y,z: z**2*x, lambda x,y,z: z**2*y, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: z**4, lambda x,y,z: 1/z, lambda x,y,z: np.exp(z), lambda x,y,z: np.sin(z), lambda x,y,z: np.cos(z)])

    
basis_functions_name_mix0_3d = [
        lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'x^2z', \
        lambda x,y,z: 'y^2x', lambda x,y,z: 'y^2z', \
        lambda x,y,z: 'z^2x', lambda x,y,z: 'z^2y', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'x^4', lambda x,y,z: '1/x', lambda x,y,z: 'e^x', lambda x,y,z: 'sin(x)', lambda x,y,z: 'cos(x)']

basis_functions_name_mix1_3d = [
        lambda y,x,z: '1', lambda y,x,z: 'x', lambda y,x,z: 'y', lambda y,x,z: 'z',\
        lambda y,x,z: 'x^2', lambda y,x,z: 'y^2', lambda x,y,z: 'z^2', lambda y,x,z: 'xy', lambda y,x,z: 'xz', lambda y,x,z: 'yz', \
        lambda y,x,z: 'x^2y', lambda y,x,z: 'x^2z', \
        lambda y,x,z: 'y^2x', lambda y,x,z: 'y^2z', \
        lambda y,x,z: 'z^2x', lambda y,x,z: 'z^2y', \
        lambda y,x,z: 'x^3', lambda y,x,z: 'y^3', lambda y,x,z: 'z^3',  \
        lambda y,x,z: 'y^4', lambda y,x,z: '1/y', lambda y,x,z: 'e^y', lambda y,x,z: 'sin(y)', lambda y,x,z: 'cos(y)']

basis_functions_name_mix2_3d = [
        lambda z,x,y: '1', lambda z,x,y: 'x', lambda z,x,y: 'y', lambda z,x,y: 'z',\
        lambda z,x,y: 'x^2', lambda z,x,y: 'y^2', lambda z,x,y: 'z^2', lambda z,x,y: 'xy', lambda z,x,y: 'xz', lambda z,x,y: 'yz', \
        lambda z,x,y: 'x^2y', lambda z,x,y: 'x^2z', \
        lambda z,x,y: 'y^2x', lambda z,x,y: 'y^2z', \
        lambda z,x,y: 'z^2x', lambda z,x,y: 'z^2y', \
        lambda z,x,y: 'x^3', lambda z,x,y: 'y^3', lambda z,x,y: 'z^3',  \
        lambda z,x,y: 'z^4', lambda z,x,y: '1/z', lambda z,x,y: 'e^z', lambda z,x,y: 'sin(z)', lambda z,x,y: 'cos(z)']
        
basis_functions_name_mix1_3d_ = [
        lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'x^2z', \
        lambda x,y,z: 'y^2x', lambda x,y,z: 'y^2z', \
        lambda x,y,z: 'z^2x', lambda x,y,z: 'z^2y', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'y^4', lambda x,y,z: '1/y', lambda x,y,z: 'e^y', lambda x,y,z: 'sin(y)', lambda x,y,z: 'cos(y)']

basis_functions_name_mix2_3d_ = [
        lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'x^2z', \
        lambda x,y,z: 'y^2x', lambda x,y,z: 'y^2z', \
        lambda x,y,z: 'z^2x', lambda x,y,z: 'z^2y', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'z^4', lambda x,y,z: '1/z', lambda x,y,z: 'e^z', lambda x,y,z: 'sin(z)', lambda x,y,z: 'cos(z)']
    
    
############ mix same basis functions 3d ################
basis_functions_3d = np.array([
        lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**2*y, lambda x,y,z: x**2*z, \
        lambda x,y,z: y**2*x, lambda x,y,z: y**2*z, \
        lambda x,y,z: z**2*x, lambda x,y,z: z**2*y, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: x**4,  lambda x,y,z: y**4,  lambda x,y,z: z**4, \
        lambda x,y,z: 1/x, lambda x,y,z: 1/y, lambda x,y,z: 1/z, \
        lambda x,y,z: np.exp(x), lambda x,y,z: np.exp(y), lambda x,y,z: np.exp(z), \
        lambda x,y,z: np.sin(x), lambda x,y,z: np.sin(y), lambda x,y,z: np.sin(z), \
        lambda x,y,z: np.cos(x), lambda x,y,z: np.cos(y), lambda x,y,z: np.cos(z)])
    
basis_functions_name_3d = [
        lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'x^2z', \
        lambda x,y,z: 'y^2x', lambda x,y,z: 'y^2z', \
        lambda x,y,z: 'z^2x', lambda x,y,z: 'z^2y', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'x^4',  lambda x,y,z: 'y^4',  lambda x,y,z: 'z^4', \
        lambda x,y,z: '1/x', lambda x,y,z: '1/y', lambda x,y,z: '1/z', \
        lambda x,y,z: 'e^x', lambda x,y,z: 'e^y', lambda x,y,z: 'e^z', \
        lambda x,y,z: 'sin(x)', lambda x,y,z: 'sin(y)', lambda x,y,z: 'sin(z)', \
        lambda x,y,z: 'cos(x)', lambda x,y,z: 'cos(y)', lambda x,y,z: 'cos(z)']
    

############ poly same basis functions 3d ################
basis_functions_poly_3d = np.array([
        lambda x,y,z: 1, lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z,\
        lambda x,y,z: x**2, lambda x,y,z: y**2, lambda x,y,z: z**2, lambda x,y,z: x*y, lambda x,y,z: x*z, lambda x,y,z: y*z, \
        lambda x,y,z: x**2*y, lambda x,y,z: x**2*z, \
        lambda x,y,z: y**2*x, lambda x,y,z: y**2*z, \
        lambda x,y,z: z**2*x, lambda x,y,z: z**2*y, \
        lambda x,y,z: x**3, lambda x,y,z: y**3, lambda x,y,z: z**3,  \
        lambda x,y,z: x**3*y,  lambda x,y,z: x**3*z, 
        lambda x,y,z: y**3*x,  lambda x,y,z: y**3*z,
        lambda x,y,z: z**3*x,  lambda x,y,z: z**3*y,
        lambda x,y,z: x**2*y**2, lambda x,y,z: x**2*z**2, lambda x,y,z: y**2*z**2,
        lambda x,y,z: x**4,  lambda x,y,z: y**4,  lambda x,y,z: z**4])
    
basis_functions_poly_name_3d = [
        lambda x,y,z: '1', lambda x,y,z: 'x', lambda x,y,z: 'y', lambda x,y,z: 'z',\
        lambda x,y,z: 'x^2', lambda x,y,z: 'y^2', lambda x,y,z: 'z^2', lambda x,y,z: 'xy', lambda x,y,z: 'xz', lambda x,y,z: 'yz', \
        lambda x,y,z: 'x^2y', lambda x,y,z: 'x^2z', \
        lambda x,y,z: 'y^2x', lambda x,y,z: 'y^2z', \
        lambda x,y,z: 'z^2x', lambda x,y,z: 'z^2y', \
        lambda x,y,z: 'x^3', lambda x,y,z: 'y^3', lambda x,y,z: 'z^3',  \
        lambda x,y,z: 'x^3y',  lambda x,y,z: 'x^3z', 
        lambda x,y,z: 'y^3x',  lambda x,y,z: 'y^3z',
        lambda x,y,z: 'z^3x',  lambda x,y,z: 'z^3y',
        lambda x,y,z: 'x^2y^2', lambda x,y,z: 'x^2z^2', lambda x,y,z: 'y^2z^2',
        lambda x,y,z: 'x^4',  lambda x,y,z: 'y^4',  lambda x,y,z: 'z^4']
    


    
    
########################################
##### data generation ##################
########################################
def data_generator(func, x0, t, a, real_list, num=None, num_split=None, noise_var=0):
    sol_org_list = get_multi_sol(func, x0, t, a, noise_var)
    
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
    if len(real_list)>2:
        ax.text(1, .65, f'${real_list[2]}$', fontsize=12)
    
    
from scipy.integrate import odeint
def ode_solver(func, x0, t, a, noise_var=0):
    sol = odeint(func, x0, t, args=a)
    
    if noise_var==0:
        return sol, t

    else:
        ### add noise
        sol_noised = sol + np.random.randn(*sol.shape)*noise_var
        ### smooth
        sol, t = smooth(sol_noised, t, window_size=None, poly_order=2, verbose=False)
        
        return sol, t

def get_multi_sol(func, x0, t, a, noise_var):
    num_traj = len(a)
    sol_org_list_ = []
    for i in range(num_traj):
        sol_, _ = ode_solver(func, x0[i], t, a[i], noise_var)
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
    
    
    

def smooth(y, t, window_size, poly_order=2, verbose=False):    
    from scipy.signal import savgol_filter
    from statsmodels.tsa.statespace.tools import diff


    # Automatic tunning of the window size 
    if window_size == None: 
        
        y_norm0 = (y[:,0]-min(y[:,0]))/(max(y[:,0])-min(y[:,0]))
        smoothed_vec_0 = [y_norm0] 
        std_prev = np.std(diff(y_norm0,1))
        window_size_used = 1 
        std1 = [] 
        while True:
            std1.append(std_prev)
            window_size_used += 10
            y_norm0 = savgol_filter(y_norm0, window_size_used, poly_order)
            std_new = np.std(diff(y_norm0,1))
            if verbose: 
                print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
            if abs((std_new-std_prev)/std_prev) < 0.1: 
                window_size_used -= 10
                break
            else:
                std_prev = std_new
                smoothed_vec_0.append(y_norm0)
                y_norm0 = (y[:,0]-min(y[:,0]))/(max(y[:,0])-min(y[:,0]))  
            
        if window_size_used > 1: 
            print('Smoothing window size (dimension 1): '+str(window_size_used),'\n')
            
            y[:,0] = savgol_filter(y[:,0], window_size_used, poly_order)
        else: 
            print('No smoothing applied')
            print('\n')
        
        
        y_norm1 = (y[:,1]-min(y[:,1]))/(max(y[:,1])-min(y[:,1])) 
        smoothed_vec_1 = [y_norm1] 
        std_prev = np.std(diff(y_norm1,1))
        window_size_used = 1 
        std2 = [] 
        while True:
            std2.append(std_prev)
            window_size_used += 10 
            y_norm1 = savgol_filter(y_norm1, window_size_used, poly_order)
            std_new = np.std(diff(y_norm1,1))
            if verbose: 
                print('Prev STD: %.5f - New STD: %.5f - Percent change: %.5f' % (std_prev, std_new, 100*(std_new-std_prev)/std_prev))
            if abs((std_new-std_prev)/std_prev)  < 0.1: 
                window_size_used -= 10
                break   
            else:
                std_prev = std_new
                smoothed_vec_1.append(y_norm1)
                y_norm1 = (y[:,1]-min(y[:,1]))/(max(y[:,1])-min(y[:,1])) 
                     
             

        if window_size_used > 1: 
            print('Smoothing window size (dimension 2): '+str(window_size_used),'\n')
            y[:,1] = savgol_filter(y[:,1], window_size_used, poly_order)
        else: 
            print('No smoothing applied')
            print('\n')
            
        
    # Pre-specified window size
    else: 
        y[:,0] = savgol_filter(y[:,0], window_size, poly_order)
        y[:,1] = savgol_filter(y[:,1], window_size, poly_order)
        
        t = t[:len(y)]
    
    return y, t


