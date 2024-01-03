#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:59:39 2024

@author: do0236li
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
    
    
from Lotka_constants import get_basis_functions
import Lotka_constants as constants
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
i = 3

x0 = x0_list[i]
a = a_list[i]

sol_, t_ = ode_solver(func, x0, t, a)
_, sol_deriv_, _ = get_deriv(sol_, t_, deriv_spline)


# ====================================
#        plot 
# ====================================

plt.figure(1)
plt.plot(sol_[:,0], sol_[:,1])
plt.xlabel('$\hat{x}(t)$')
plt.ylabel('$\hat{y}(t)$')




ax = plt.figure(2).add_subplot(projection='3d')
ax.plot(t[:56], 1*np.ones_like(t)[:56], sol_[:56,1], c='c', alpha=.7)
ax.plot(t[12:68], 2*np.ones_like(t)[12:68], sol_[12:68,1], c='y', alpha=.7)
ax.plot(t[24:80], 3*np.ones_like(t)[24:80], sol_[24:80,1], c='orange', alpha=.7)
ax.plot(t, 4*np.ones_like(t), sol_[:,1], c='k')


ax.plot(t[0]*np.ones(40), np.linspace(1, 4, 40), sol_[0,1]*np.ones(40), '--', c='c', alpha=.5)
ax.plot(t[55]*np.ones(40), np.linspace(1, 4, 40), sol_[55,1]*np.ones(40), '--', c='c', alpha=.5)

ax.plot(t[12]*np.ones(40), np.linspace(2, 4, 40), sol_[12,1]*np.ones(40), '--', c='y', alpha=.5)
ax.plot(t[67]*np.ones(40), np.linspace(2, 4, 40), sol_[67,1]*np.ones(40), '--', c='y', alpha=.5)

ax.plot(t[24]*np.ones(40), np.linspace(3, 4, 40), sol_[24,1]*np.ones(40), '--', c='orange', alpha=.5)
ax.plot(t[79]*np.ones(40), np.linspace(3, 4, 40), sol_[79,1]*np.ones(40), '--', c='orange', alpha=.5)


x = t[:56]
y = np.linspace(1, 2)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[:56,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='c', alpha=.3)

x = t[12:68]
y = np.linspace(2, 3)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[12:68,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='y', alpha=.3)

x = t[24:80]
y = np.linspace(3, 4)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[24:80,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='orange', alpha=.3)



ax.set_xlabel('$Time$')
ax.set_ylabel('$sub-series$')
ax.set_zlabel('$\hat{x}$')

plt.savefig('demo.png', transparent=True)




