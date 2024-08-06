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
import matplotlib

font = {'size': 15}
matplotlib.rc('font', **font)

### plot data
plt.figure(1, figsize=[5,4])
plt.plot(sol_[:,0], sol_[:,1])
plt.xlabel('$\hat{x}(t)$',size=18)
plt.ylabel('$\hat{y}(t)$',size=18)

plt.tight_layout()
plt.savefig('Figures/data.png', dpi=300)





### plot distribution
# import seaborn as sns

colors = ['forestgreen', 'blue', 'firebrick']
fig, ax = plt.subplots(2,3,figsize=[15,7])

sample = np.random.randn(3,1000)*.01
# sns.histplot(ax=ax[0,0], x=[-.1,-.05,0,.05,.1], weights=[5,30,60,4,1])
ax[0,0].hist([-.1,-.05,0,.05,.1], weights=[5,30,60,4,1], alpha = 0.5, color=colors[0], rwidth=0.4)
ax[0,0].hist([-.2,-.15,-.1,0,.08,.1,.15,.2], weights=[5,10,20,15,7,10,22,11], alpha = 0.5, color=colors[1], rwidth=0.2)
ax[0,0].hist([-.03,-.01,0,.02], weights=[10,40,45,5], alpha = 0.5, color=colors[2], rwidth=1)
ax[0,0].set_xlim(-.3,.3)
ax[0,0].set_xticks(ticks=[-.3,.3])
ax[0,0].set_yticks(ticks=[20,40,60])
ax[0,0].set_xlabel(r'$1$', size=18)

ax[0,1].hist([.6,.63,.66,.68,.7,.73,.75,1], weights=[5,10,4,20,31,20,7,3], alpha = 0.5, color=colors[0], rwidth=0.2)
ax[0,1].hist([.9,.95,.99,1.05,1.1,1.3], weights=[3,37,48,2,3,7], alpha = 0.5, color=colors[1], rwidth=0.3)
ax[0,1].hist([.48,.5,.52,.53], weights=[30,33,24,13], alpha = 0.5, color=colors[2], rwidth=1)
ax[0,1].set_xlim(.3,1.4)
ax[0,1].set_xticks(ticks=[.3,1.4])
ax[0,1].set_yticks(ticks=[20,40,60])
ax[0,1].set_xlabel(r'$x$', size=18)

ax[0,2].hist([-.05,-.01,0,.03,.05], weights=[3,25,63,4,5], alpha = 0.5, color=colors[0], rwidth=0.5)
ax[0,2].hist([-.03,-.02,0,.01,.02], weights=[8,27,55,7,3], alpha = 0.5, color=colors[1], rwidth=0.5)
ax[0,2].hist([-.04,.02,.04], weights=[22,58,20], alpha = 0.5, color=colors[2], rwidth=1)
ax[0,2].set_xlim(-.3,.3)
ax[0,2].set_xticks(ticks=[-.3,.3])
ax[0,2].set_yticks(ticks=[20,40,60])
ax[0,2].set_xlabel(r'$y$', size=18)

ax[1,0].hist([-.05,-.01,0,.03,.05], weights=[3,63,4,5,25], alpha = 0.5, color=colors[0], rwidth=0.5)
ax[1,0].hist([-.03,-.02,0,.01,.02], weights=[55,8,27,7,3], alpha = 0.5, color=colors[1], rwidth=0.5)
ax[1,0].hist([-.04,0,.02,.04], weights=[3,40,50,7], alpha = 0.5, color=colors[2], rwidth=1)
ax[1,0].set_xlim(-.3,.3)
ax[1,0].set_xticks(ticks=[-.3,.3])
ax[1,0].set_yticks(ticks=[20,40,60])
ax[1,0].set_xlabel(r'$x^2$', size=18)

ax[1,1].hist([-.85,-.82,-.8,-.77], weights=[8,63,24,5], alpha = 0.5, color=colors[0], rwidth=1)
ax[1,1].hist([-1.2,-1.05,-1.,-.96], weights=[5,55,30,10], alpha = 0.5, color=colors[1], rwidth=.5)
ax[1,1].hist([-.63,-.6,-.57], weights=[30,64,6], alpha = 0.5, color=colors[2], rwidth=1)
ax[1,1].set_xlim(-1.3,-.4)
ax[1,1].set_xticks(ticks=[-1.3,-.4])
ax[1,1].set_yticks(ticks=[20,40,60])
ax[1,1].set_xlabel(r'$xy$', size=18)

ax[1,2].hist([-.05,-.01,0,.03,.05], weights=[3,4,35,25,33], alpha = 0.5, color=colors[0], rwidth=0.5)
ax[1,2].hist([-.03,-.02,0,.01,.02], weights=[27,12,45,13,3], alpha = 0.5, color=colors[1], rwidth=0.5)
ax[1,2].hist([-.04,0,.02,.04], weights=[21,36,36,7], alpha = 0.5, color=colors[2], rwidth=1)
ax[1,2].set_xlim(-.3,.3)
ax[1,2].set_xticks(ticks=[-.3,.3])
ax[1,2].set_yticks(ticks=[20,40,60])
ax[1,2].set_xlabel(r'$y^2$', size=18)

fig.tight_layout()
fig.savefig('Figures/distribution.png', dpi=300)




### plot sub-series
from mpl_toolkits.mplot3d import Axes3D
font = {'size'   : 10}
matplotlib.rc('font', **font)


start1, end1 = 0, 35
start2, end2 = 20, 60
start3, end3 = 45, 80
ratio1 = .30
ratio2 = .60
 
ax = plt.figure(3, constrained_layout=True).add_subplot(projection='3d')
ax.plot(t[start1:end1], 0*np.zeros_like(t)[start1:end1], sol_[start1:end1,1], c='c', linewidth=.8, alpha=.5)
ax.plot(t[start2:end2], 3*ratio1*np.ones_like(t)[start2:end2], sol_[start2:end2,1], c='orange', linewidth=.8, alpha=.5)
ax.plot(t[start3:end3], 3*ratio2*np.ones_like(t)[start3:end3], sol_[start3:end3,1], c='r', linewidth=.8, alpha=.5)
ax.plot(t, 3*np.ones_like(t), sol_[:,1], linewidth=.8, c='k')


ax.plot(t[start1]*np.ones(40), np.linspace(0, 3, 40), sol_[start1,1]*np.ones(40), '--', c='c', alpha=.7)
ax.plot(t[end1-1]*np.ones(40), np.linspace(0, 3, 40), sol_[end1-1,1]*np.ones(40), '--', c='c', alpha=.7)

ax.plot(t[start2]*np.ones(40), np.linspace(3*ratio1, 3, 40), sol_[start2,1]*np.ones(40), '--', c='orange', alpha=.7)
ax.plot(t[end2-1]*np.ones(40), np.linspace(3*ratio1, 3, 40), sol_[end2-1,1]*np.ones(40), '--', c='orange', alpha=.7)

ax.plot(t[start3]*np.ones(40), np.linspace(3*ratio2, 3, 40), sol_[start3,1]*np.ones(40), '--', c='r', alpha=.7)
ax.plot(t[end3-1]*np.ones(40), np.linspace(3*ratio2, 3, 40), sol_[end3-1,1]*np.ones(40), '--', c='r', alpha=.7)


x = t[start1:end1]
y = np.linspace(0, 3*ratio1)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[start1:end1,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='c', alpha=.3)

x = t[start2:end2]
y = np.linspace(3*ratio1, 3*ratio2)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[start2:end2,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='orange', alpha=.3)

x = t[start3:end3]
y = np.linspace(3*ratio2, 3)
X, Y = np.meshgrid(x, y)
zs = np.repeat(sol_[start3:end3,1].reshape([-1,1]), y.shape[0], axis=1).T.flatten()
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, color='r', alpha=.3)


ax.text(1, 1.5, 0, r"$i$", color='c', size=8)
ax.text(4.7, 1.5, 0, r"$i+1$", color='orange', size=8)
ax.text(7.2, 1.5, 0, r"$i+2$", color='r', size=8)

ax.set_yticks(ticks=[0, 3*ratio1, 3*ratio2])
ax.set_yticklabels([])
ax.set_zticks(ticks=[1,2,3])


ax.set_facecolor('white')  # Set background color to white
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

ax.set_xlabel('$Time$')
ax.set_ylabel('$i$', labelpad=-10)
ax.set_zlabel('$\hat{x}$')

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.5, 1, 1]))
ax.view_init(elev=41, azim=-89)  # Adjust the angles as needed

plt.tight_layout()
plt.savefig('Figures/sub_series.png', bbox_inches='tight', dpi=300)

