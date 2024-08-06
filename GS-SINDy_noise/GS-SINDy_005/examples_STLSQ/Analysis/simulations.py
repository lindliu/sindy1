#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:15:38 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '../Exp1_Lotka_Volterra')
sys.path.insert(1, '../Exp2_Modified_Lotka_Volterra')
sys.path.insert(1, '../Exp3_Brusselator')
sys.path.insert(1, '../Exp4_Van_der_Pol')
sys.path.insert(1, '../Exp5_Lorenz')
sys.path.insert(1, '../Exp6_Pendulum')

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from GSINDy import SLS
from utils import ode_solver, get_deriv, get_theta
from utils import func_Lotka_Voltera, func_M_Lotka_Voltera, func_Brusselator, func_Van_der_Pol, func_Lorenz, func_Pendulum
from utils import basis_functions_mix0, basis_functions_mix1, basis_functions_name_mix0, basis_functions_name_mix1, \
    basis_functions_poly_5, basis_functions_name_poly_5

from M_Lotka_constants import get_basis_functions
import M_Lotka_constants as constants
np.set_printoptions(formatter={'float': lambda x: "{0:.4f}".format(x)})




########## hyper parameters ###########
ensemble = constants.ensemble
precision = constants.precision
deriv_spline = constants.deriv_spline
alpha = constants.alpha
threshold_sindy_list = constants.threshold_sindy_list

########## function variable ###########
t = constants.t
x0_list = constants.x0_list
a_list = constants.a_list

func = constants.func
real_list = constants.real_list

########## basis functions and optimizer ###########
basis_type = constants.basis_type
basis, opt = get_basis_functions(basis_type=basis_type, GSINDY=False)

basis_functions_list = basis['functions']

threshold_sindy = .01


######################################################
################## get data ##########################
######################################################
i = 1

x0 = x0_list[i]
a = a_list[i]

sol_, t_ = ode_solver(func, x0, t, a)
_, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)


### load Xi ###
Xi_gsindy_all = np.load(glob(os.path.join(f'../Exp2_Modified_Lotka_Volterra/results/coeff/gsindy_all_{basis_type}*6.npy'))[0])[i]
Xi_sindy = np.load(glob(os.path.join(f'../Exp2_Modified_Lotka_Volterra/results/coeff/sindy_{basis_type}*{i}.npy'))[0])



############################################
############# simulations ##################
############################################
def func_simulation(x, t, param, basis_functions_list):
    mask0 = param[0]!=0
    mask1 = param[1]!=0
    
    x1, x2 = x
    dx1dt = 0
    for par,f in zip(param[0][mask0], basis_functions_list[0][mask0]):
        dx1dt = dx1dt+par*f(x1,x2)
    
    dx2dt = 0
    for par,f in zip(param[1][mask1], basis_functions_list[1][mask1]):
        dx2dt = dx2dt+par*f(x1,x2)
        
    dxdt = [dx1dt, dx2dt]
    return dxdt

from ModelSelection import ModelSelection
from scipy.integrate import odeint

basis_functions_list = basis['functions']
basis_functions_name_list = basis['names']
basis_functions_name_list_ = [np.array([f(1,1) for f in basis['names'][0]]), \
                             np.array([f(1,1) for f in basis['names'][1]])]
t_s = np.linspace(t[0],t[-1],1000)


##### gsindy all simulations #####
args = (Xi_gsindy_all, basis_functions_list)
simulation_gsindy_all = odeint(func_simulation, x0, t_s, args=args)

##### sindy simulations #####
args = (Xi_sindy, basis_functions_list)
simulation_sindy = odeint(func_simulation, x0, t_s, args=args)



############### plot ##################
fig, ax = plt.subplots(1,1)
ax.plot(sol_[:,0],sol_[:,1],'o')
ax.plot(simulation_gsindy_all[:,0],simulation_gsindy_all[:,1])
ax.plot(simulation_sindy[:,0],simulation_sindy[:,1])


