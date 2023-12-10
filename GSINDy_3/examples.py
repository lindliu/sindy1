#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:49:29 2023

@author: do0236li
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name
from GSINDy import *

opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
deriv_spline = True#False#
threshold_tol = 1e-3

# ################### 3 variable ####################
# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,10,dt)
# x0 = [.5, 1]
# a = [(.2, -.6, -.5), (.4, -.8, -.7), (.6, -1, -1)]
# func = func3__
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y + c*y^3"
# threshold_sindy=1e-2
# threshold_similarity = 1e-2


################### 2 variable ####################
# alpha = .05
# dt = .1   ## 0,3
# t = np.arange(0,1.5,dt)
# x0 = [.5, 1]
# a = [(.16, .25), (.3, .4), (.3, .5)]
# func = func12_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + b*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_similarity = 1e-3
# threshold_tol = 1e-2

# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,12,dt)
# x0 = [.5, 1]
# a = [(.2, -.6), (.4, -.8), (.6, -1)]
# func = func3_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
# threshold_sindy=1e-2
# threshold_similarity = 1e-2

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,6,dt)
# x0 = [4, 1]
# a = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5)]
# func = func4_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x + b*xy"
# real1 = "y'=b*y + a*xy"
# threshold_sindy=1e-2
# threshold_similarity = 1e-3


################### 1 variable ####################
# alpha = .05
# dt = .05   ## 0,3
# t = np.arange(0,2.5,dt)
# x0 = [.2, 1]
# a = [(.12,), (.16,), (.2,)]
# func = func1
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + x^2"
# real1 = "y'=-y"
# threshold_sindy=5e-2
# threshold_similarity = 1e-3
# threshold_tol = 1e-2

# alpha = .05
# dt = .1
# t = np.arange(0,2,dt)
# x0 = [.5, 1]
# a = [(.25,), (.3,), (.5,)]
# # a = [(.1,), (.25,), (.3,), (.5,)]
# func = func2
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=.2 + a*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_similarity = 1e-3
# threshold_tol = 1e-2

# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,12,dt)
# x0 = [.5, 1]
# a = [(.2,), (.4,), (.6,)]
# func = func3
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"    
# threshold_sindy=1e-2
# threshold_similarity = 1e-2

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [4, 1]
# a = [(.7,), (1,)]#,(.5,),(.6,)]
# func = func4
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"    
# threshold_sindy=1e-2
# threshold_similarity = 1e-3

# alpha = .05
# dt = .1
# t = np.arange(0,3.3,dt)
# x0 = [np.pi-.1, 0]
# a = [(-.25,), (-.35,)]
# func = func6
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=a*y-5sin(x)"
# threshold_sindy=1e-2
# threshold_similarity = 1e-3

alpha = .05
dt = .1
t = np.arange(0,6,dt)
x0 = [np.pi-.1, 0]
# a = [-5, -6]
a = [(-.15,), (-1,), (-2,), (-5,)]
func = func7
monomial = monomial_trig
monomial_name = monomial_trig_name
real0 = "x'=y"
real1 = "y'=-0.25*y+a*sin(x)"
threshold_sindy=1e-2
threshold_similarity = 1e-3




num_traj = len(a)
num_feature = len(x0)

from utils import get_multi_sol
sol_org_list = get_multi_sol(func, x0, t, a)


fig, ax = plt.subplots(1,1,figsize=[6,3])
for i in range(num_traj):
    ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
ax.legend()
ax.text(1, .95, f'${real0}$', fontsize=12)
ax.text(1, .8, f'${real1}$', fontsize=12)


############# group sindy ##############
# opts_params =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1]
opts_params =  [threshold_sindy]
for param in opts_params:
    print(f'################### [GSINDy] threshold: {param} ################')
    
    gsindy = GSINDy(monomial=monomial,\
                    monomial_name=monomial_name, \
                    num_traj = num_traj, \
                    num_feature = num_feature, \
                    threshold_sindy = param,  #threshold_sindy, \
                    threshold_tol=threshold_tol, \
                    threshold_similarity=threshold_similarity, \
                    alpha=alpha,\
                    deriv_spline=deriv_spline,\
                    optimizer=opt, ##['Manually', 'SQTL', 'LASSO', 'SR3']
                    ensemble=ensemble)    
        
    gsindy.get_multi_sub_series(sol_org_list, t, num_series=60, window_per=.7) ### to get theta_list, sol_deriv_list
    gsindy.basis_identification(remove_per=.2, plot_dist=False) ##True
    
    Xi_final = gsindy.prediction(sol_org_list, t)
    
    
    
    all_basis = gsindy.all_basis
    diff_basis = gsindy.diff_basis
    np.set_printoptions(formatter={'float': lambda x: "{0:.3f}".format(x)})
    print('*'*50)
    print(f'real0: {real0}')
    print(f'feature 0 with different basis {monomial_name[diff_basis[0]]}: \n {Xi_final[:,0,all_basis[0]]} \n {monomial_name[all_basis[0]]}')
    print(f'real1: {real1}')
    print(f'feature 1 with different basis {monomial_name[diff_basis[1]]}: \n {Xi_final[:,1,all_basis[1]]} \n {monomial_name[all_basis[1]]}')



    MSE = lambda x, y: ((x-y)**2).mean()
    loss = []
    for j in range(num_traj):
        sol0_deriv_prediction = gsindy.theta_org_list[0][j,:,:]@Xi_final[j,0,:]
        sol1_deriv_prediction = gsindy.theta_org_list[1][j,:,:]@Xi_final[j,1,:]
        
        sol0_deriv = gsindy.sol_deriv_org_list[0][j].squeeze()
        sol1_deriv = gsindy.sol_deriv_org_list[1][j].squeeze()
        
        loss_ = MSE(sol0_deriv_prediction, sol0_deriv) + MSE(sol1_deriv_prediction, sol1_deriv)
        loss.append(loss_)
        print(f'trajectory {j} MSE loss: {loss_}')
    print(f'Mean of each trajecctory loss: {np.mean(loss)}')


#%% compare to pysindy
import pysindy_ as ps

if func.__name__ not in ['func6', 'func7']:
    from pysindy.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
    from pysindy.optimizers import STLSQ
    
    # lib_generalized = PolynomialLibrary(degree=5)
    functions = [lambda x,y: 1, \
            lambda x,y: x, lambda x,y: y, \
            lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
            lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
            lambda x,y: x**4, lambda x,y: x**3*y, lambda x,y: x**2*y**2, lambda x,y: x*y**3, lambda x,y: y**4, \
            lambda x,y: x**5, lambda x,y: x**4*y, lambda x,y: x**3*y**2, lambda x,y: x**2*y**3, lambda x,y: x*y**4, lambda x,y: y**5]
    names = [lambda x,y: '1', \
            lambda x,y: 'x', lambda x,y: 'y', \
            lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
            lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
            lambda x,y: 'x^4', lambda x,y: 'x^3y', lambda x,y: 'x^2y^2', lambda x,y: 'xy^3', lambda x,y: 'y^4', \
            lambda x,y: 'x^5', lambda x,y: 'x^4y', lambda x,y: 'x^3y^2', lambda x,y: 'x^2y^3', lambda x,y: 'xy^4', lambda x,y: 'y^5']
    lib_custom = CustomLibrary(library_functions=functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])

else:
    from pysindy.feature_library import FourierLibrary, CustomLibrary
    from pysindy.feature_library import GeneralizedLibrary
    from pysindy.optimizers import STLSQ
    
    functions = [lambda x,y: 1, lambda x,y: x, lambda x,y: y, lambda x,y: np.sin(x), lambda x,y: np.sin(y), \
                  lambda x,y: np.cos(x), lambda x,y: np.cos(y)]
    names = [lambda x,y: '1', lambda x,y: 'x', lambda x,y: 'y', lambda x,y: 'sin(x)', lambda x,y: 'sin(y)', \
              lambda x,y: 'cos(x)', lambda x,y: 'cos(y)']
    lib_custom = CustomLibrary(library_functions=functions, function_names=names)
    # lib_generalized = GeneralizedLibrary([lib_custom, lib_fourier])
    lib_generalized = GeneralizedLibrary([lib_custom])
    # x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    # lib_generalized.fit(x)
    # lib_generalized.transform(x)


############# sindy ##############
from utils import ode_solver, get_deriv
for param in opts_params:
    optimizer = STLSQ(threshold=param, alpha=alpha)
    model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)

    print(f'################### [SINDy] threshold: {param} ################')
    for i in range(len(a)):
        # sol_, sol_deriv_, t_ = get_sol_deriv(func, x0, t, a[i], deriv_spline)
        sol_, t_ = ode_solver(func, x0, t, a[i])
        _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
        
        if ensemble:
            model.fit(sol_, t=t_, x_dot=sol_deriv_, ensemble=True, quiet=True)
        else:
            model.fit(sol_, t=t_, x_dot=sol_deriv_)
        model.print()
        # model.coefficients()
        
        # theta_ = monomial(sol_)
        # print(SLS(theta_, sol_deriv_, threshold_sindy, threshold_tol))
        
