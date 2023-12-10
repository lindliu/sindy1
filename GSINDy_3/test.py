#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:11:11 2023

@author: dliu
"""


import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name
from GSINDy import *


deriv_spline = True#False#
threshold_tol = 1e-3

################### 3 variable ####################
alpha = .05
dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,10,dt)
x0 = [.5, 1]
a = [(.2, -.6, -.5), (.4, -.8, -.7), (.6, -1, -1)]
func = func3__
monomial = monomial_poly
monomial_name = monomial_poly_name
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
threshold_sindy=1e-2
threshold_similarity = 1e-2


from utils import ode_solver, get_deriv
i = 0
sol_, t_ = ode_solver(func, x0, t, a[i])
_, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)


import pysindy_ as ps
from pysindy_.feature_library import Shell_custom_theta
from pysindy_.utils import AxesArray
from pysindy_.optimizers import STLSQ

def Axes_transfer(theta_):
    xp = AxesArray(
        np.empty(
            theta_.shape,
            dtype=np.float64,
            order='C',
        ),
        {'ax_time': 0, 'ax_coord': 1, 'ax_sample': None, 'ax_spatial': []},
    )
    
    num_basis = theta_.shape[1]
    for i in range(num_basis):
        xp[..., i] = theta_[:, i]
    
    theta = [] + [xp]
    
    return theta


theta_ = monomial(sol_)
theta = Axes_transfer(theta_)

lib_generalized = Shell_custom_theta(theta=theta)
# lib_generalized.fit(sol_)
# lib_generalized.transform(sol_)

# import pysindy as ps
optimizer = STLSQ(threshold=1e-2, alpha=.05)
model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
model.fit(np.ones([1]), t=1, x_dot=sol_deriv_[:,:])#, ensemble=True, quiet=True)
print(model.coefficients())