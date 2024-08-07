#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:04:31 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../GSINDy')
sys.path.insert(1, '..')
sys.path.insert(1, './tools')
sys.path.insert(1, 'Exp1_Lotka_Volterra')

from utils import ode_solver
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from utils import func_Lotka_Voltera
import Lotka_constants


fig, axes = plt.subplots(6,6,figsize=[12,8], sharex='col')


x0_list = Lotka_constants.x0_list
a_list = Lotka_constants.a_list
t = Lotka_constants.t
func = func_Lotka_Voltera

ax = axes[:,0]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)



sys.path.insert(1, 'Exp3_Brusselator')
from utils import func_Brusselator
import Brusselator_constants

x0_list = Brusselator_constants.x0_list
a_list = Brusselator_constants.a_list
t = Brusselator_constants.t
func = func_Brusselator

ax = axes[:,1]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)
    
    

sys.path.insert(1, 'Exp4_Van_der_Pol')
from utils import func_Van_der_Pol
import Van_constants

x0_list = Van_constants.x0_list
a_list = Van_constants.a_list
t = Van_constants.t
func = func_Van_der_Pol

ax = axes[:,2]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)

    

sys.path.insert(1, 'Exp5_Lorenz')
from utils import func_Lorenz
import Lorenz_constants

x0_list = Lorenz_constants.x0_list
a_list = Lorenz_constants.a_list
t = Lorenz_constants.t
func = func_Lorenz

ax = axes[:,3]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)
    
    
    
sys.path.insert(1, 'Exp2_Modified_Lotka_Volterra')
from utils import func_M_Lotka_Voltera
import M_Lotka_constants

x0_list = M_Lotka_constants.x0_list
a_list = M_Lotka_constants.a_list
t = M_Lotka_constants.t
func = func_M_Lotka_Voltera

ax = axes[:,4]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)
    
    
    
sys.path.insert(1, 'Exp7_FitzHugh')
from utils import func_FitzHugh
import FitzHugh_constants

x0_list = FitzHugh_constants.x0_list
a_list = FitzHugh_constants.a_list
t = FitzHugh_constants.t
func = func_FitzHugh

ax = axes[:,5]
# fig, ax = plt.subplots(1,6,figsize=[15,2])
for i in range(6):
    xx, tt = ode_solver(func, x0_list[i], t, a_list[i], noise_var=0)
    ax[i].plot(tt, xx, 'o', markersize=1)
    ax[i].set_xticks([0,15,30])


pad = 5
cols = ['Lotka-Volterra', 'Brusselator','Van der Pol','Lorenz', 'Hopf', 'FitzHugh']
for ax_, col in zip(axes[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='center', va='baseline')

rows = ['1','2','3','4','5','6']
for ax_, row in zip(axes[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=15, ha='center', va='center', rotation=90)



fig.tight_layout(pad=0, w_pad=0, h_pad=0)
fig.savefig(f'./trajectory.png', bbox_inches='tight', dpi=300)


