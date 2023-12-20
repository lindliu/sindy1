#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:43:45 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')

import numpy as np
import matplotlib.pyplot as plt
from utils import func1, func2, func3, func4, func5, func6, func7, \
                func12_, func3_, func4_, func3__, \
                monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name
from GSINDy import *

MSE = lambda x, y: ((x-y)**2).mean()
SSE = lambda x, y: ((x-y)**2).sum()

opt = 'SQTL' ##['Manually', 'SQTL', 'LASSO', 'SR3']
ensemble = False
precision = 1e-3
deriv_spline = True#False#
alpha = .05

# ####################################################
# #################### Lotka-Volterra ################
# ####################################################
# #################### 2 variable ####################
# dt = .1 
# t = np.arange(0,8,dt)
# num = 1

# x0_list = [[3, 1], [3, 1], [3, 1], [3, 1]]
# a_list = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 2
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func4_
# real0 = "x'=a*x + b*xy"
# real1 = "y'=b*y + a*xy" 
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################

# #################### 1 variable ####################
# dt = .1 
# t = np.arange(0,8,dt)
# num = 1

# x0_list = [[3, 1], [3, 1]]
# a_list = [(.7,), (1,)]#,(.5,),(.6,)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 2
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func4
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################


# #############################################################
# #################### Modified Lotka-Volterra ################
# #############################################################
# #################### 3 variable ####################
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,10,dt)
# num = 2

# x0_list = [[.4, 1], [.4, 1], [.4, 1]]
# a_list = [(.2, -.6, -.5), (.4, -.8, -.7), (.6, -1, -1)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 3
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func3__
# real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y + c*y^3"
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################

# #################### 2 variable ####################
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,10,dt)
# num = 2

# x0_list = [[.4, 1], [.4, 1], [.4, 1]]
# a_list = [(.2, -.6), (.4, -.8), (.6, -1)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 3
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func3_
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################

# #################### 1 variable ####################
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,10,dt)
# num = 2

# x0_list = [[.4, 1], [.4, 1], [.4, 1]]
# a_list = [(.2,), (.4,), (.6,)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 2
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func3
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"   
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################



# ####################################################
# #################### Brusselator ###################
# ####################################################
# ##################### 2 variable ####################
# dt = .1    
# t = np.arange(0,20,dt)
# num = 1

# x0_list = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
# a_list = [(1, 3), (.8, 2.5), (.9, 2.8), (.5, 2.8), (.6, 2.6), (.7, 2.8)]

# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 3 
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func8
# real0 = "x'=a-4x+x^2y"
# real1 = "y'=bx-x^2y"
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ######################################################




####################################################
#################### Van der Pol ###################
####################################################

# # ################## 1 variable ####################
# dt = .05 
# t = np.arange(0,20,dt)
# num = 6

# x0_list = [[1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5], [1.5, -.5]]
# a_list = [(.5,), (.3,), (.4,),(.2,), (.35,), (.6,)]
# if num==1:
#     x0 = x0_list[:num]
#     a = a_list[:num]
#     num_split = 2
    
# elif num>1:
#     x0 = x0_list[:num]
#     a = a_list[:num]

# func = func5
# real0 = "x'=5*(x - y- a*x^3)"
# real1 = "y'=.2*x"    
# # monomial = monomial_all
# # monomial_name = monomial_all_name
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# ####################################################



###################################################
################### quadratic #####################
###################################################
################## 2 variable ####################
alpha = .05
dt = .1   ## 0,3
t = np.arange(0,2.,dt)
num = 1

x0_list = [[.5, 1], [.5, 1], [.5, 1]]
a_list = [(.16, .25), (.3, .4), (.3, .5)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]

func = func12_
real0 = "x'=a + b*x^2"
real1 = "y'=-y"
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name
##################################################

################### 1 variable ####################
alpha = .05
dt = .05   ## 0,3
t = np.arange(0,2.3,dt)
num = 3

x0_list = [[.2, 1], [.2, 1], [.1, 1]]
a_list = [(.12,), (.16,), (.2,)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]
    
func = func1
real0 = "x'=a + x^2"
real1 = "y'=-y"
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name
####################################################

################### 1 variable ####################
alpha = .05
dt = .05   ## 0,3
t = np.arange(0,2.3,dt)
num = 3

x0_list = [[.5, 1], [.5, 1], [.5, 1]]
a_list = [(.25,), (.3,), (.5,)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]
    
func = func2
real0 = "x'=.2 + a*x^2"
real1 = "y'=-y"
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_poly
monomial_name = monomial_poly_name
##################################################


################################################################
#################### Pendulum motion model #####################
################################################################
################### 1 variable ####################
dt = .2
t = np.arange(0,3.3,dt)
num = 2

x0_list = [[np.pi-.1, 0], [np.pi-.1, 0]]
a_list = [(-.25,), (-.35,)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]

func = func6
real0 = "x'=y"
real1 = "y'=a*y-5sin(x)"
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_trig
monomial_name = monomial_trig_name
###################################################

################### 1 variable ####################
dt = .1
t = np.arange(0,6,dt)
num = 4

x0_list = [[np.pi-.1, 0], [np.pi-.1, 0], [np.pi-.1, 0], [np.pi-.1, 0]]
a_list = [(-.15,), (-1,), (-2,), (-5,)]
if num==1:
    x0 = x0_list[:num]
    a = a_list[:num]
    num_split = 2
    
elif num>1:
    x0 = x0_list[:num]
    a = a_list[:num]
    
func = func7
real0 = "x'=y"
real1 = "y'=-0.25*y+a*sin(x)"
# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = monomial_trig
monomial_name = monomial_trig_name
###################################################


if __name__ == "__main__":
    #%% generate data
    from utils import get_multi_sol
    sol_org_list = get_multi_sol(func, x0, t, a)

    ### generate data ###
    num_traj = len(a)
    num_feature = len(x0[0])

    ### plot data ###
    fig, ax = plt.subplots(1,1,figsize=[6,3])
    for i in range(num_traj):
        ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
    ax.legend()
    ax.text(1, .95, f'${real0}$', fontsize=12)
    ax.text(1, .8, f'${real1}$', fontsize=12)

    #%% pysindy settings
    import pysindy_ as ps
    from pysindy_.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
    from sklearn.linear_model import Lasso

    if monomial.__name__ == 'monomial_poly':
        # lib_generalized = PolynomialLibrary(degree=5)
        basis_functions = [lambda x,y: 1, \
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

    if monomial.__name__ == 'monomial_all':
        basis_functions = [lambda x,y: 1, \
                lambda x,y: x, lambda x,y: y, \
                lambda x,y: x**2, lambda x,y: x*y, lambda x,y: y**2, \
                lambda x,y: x**3, lambda x,y: x**2*y, lambda x,y: x*y**2, lambda x,y: y**3, \
                lambda x,y: x**4, lambda x,y: y**4, \
                lambda x,y: 1/x, lambda x,y: 1/y, 
                lambda x,y: np.sin(x), lambda x,y: np.sin(y),lambda x,y: np.cos(x), lambda x,y: np.cos(y),\
                lambda x,y: np.exp(x), lambda x,y: np.exp(y)]
        names = [lambda x,y: '1', \
                lambda x,y: 'x', lambda x,y: 'y', \
                lambda x,y: 'x^2', lambda x,y: 'xy', lambda x,y: 'y^2', \
                lambda x,y: 'x^3', lambda x,y: 'x^2y', lambda x,y: 'xy^2', lambda x,y: 'y^3', \
                lambda x,y: 'x^4', lambda x,y: 'y^4', \
                lambda x,y: '1/x', lambda x,y: '1/y', \
                lambda x,y: 'sin(x)', lambda x,y: 'sin(y)',lambda x,y: 'cos(x)', lambda x,y: 'cos(y)',\
                lambda x,y: 'exp(x)', lambda x,y: 'exp(y)']
    if monomial.__name__ == 'monomial_trig':
        basis_functions = [lambda x,y: 1, lambda x,y: x, lambda x,y: y, lambda x,y: np.sin(x), lambda x,y: np.sin(y), \
                           lambda x,y: np.cos(x), lambda x,y: np.cos(y)]
        names = [lambda x,y: '1', lambda x,y: 'x', lambda x,y: 'y', lambda x,y: 'sin(x)', lambda x,y: 'sin(y)', \
                 lambda x,y: 'cos(x)', lambda x,y: 'cos(y)']
    
    lib_custom = CustomLibrary(library_functions=basis_functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])
    
    
    #%% 
    threshold_sindy_list =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    # threshold_sindy_list =  [1e-2]
    from utils import ode_solver, get_deriv
    from sklearn.linear_model import Lasso

    for traj_i in range(len(a)):
        
        ##################################
        ############# sindy ##############
        ##################################
        model_set = []
        parameter_list = []
        for threshold_sindy in threshold_sindy_list:
            # optimizer = STLSQ(threshold=threshold_sindy, alpha=alpha)
            if opt=='SQTL':
                optimizer = ps.STLSQ(threshold=threshold_sindy, alpha=alpha)
            elif opt=='LASSO':
                optimizer = Lasso(alpha=alpha, max_iter=5000, fit_intercept=False)
            elif opt=='SR3':
                optimizer = ps.SR3(threshold=threshold_sindy, nu=.1)
            
            model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
        
            # print(f'################### [SINDy] threshold: {threshold_sindy} ################')
            sol_, t_ = ode_solver(func, x0[traj_i], t, a[traj_i])
            _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
            
            if ensemble:
                model.fit(sol_, t=t_, x_dot=sol_deriv_, ensemble=True, quiet=True)
            else:
                model.fit(sol_, t=t_, x_dot=sol_deriv_)
            # model.print()
            # model.coefficients()
            
            model_set.append(model)
            # theta_ = monomial(sol_)
            # print(SLS(theta_, sol_deriv_, threshold_sindy, precision))
            
            parameter_list.append(threshold_sindy)


        ###########################
        ##### model selection #####
        ###########################
        
        from ModelSelection import ModelSelection
        
        t_steps = t.shape[0]
        ms = ModelSelection(model_set, t_steps)
        ms.compute_k_sindy()
        
        for model_id, model in enumerate(model_set):    
            sse_sum = 0
            
            if np.abs(model.coefficients()).sum()>1e3:
                ms.set_model_SSE(model_id, 1e5)
                continue
                
            ##### simulations #####
            simulation = model.simulate(x0[traj_i], t=t, integrator = "odeint")
            
            sse_sum += ms.compute_SSE(sol_org_list[traj_i], simulation)
            
            # ms.set_model_SSE(model_id, sse_sum/num_traj)
            ms.set_model_SSE(model_id, sse_sum)
        
        best_AIC_model = ms.compute_AIC()
        best_AICc_model = ms.compute_AICc()
        best_BIC_model = ms.compute_BIC()
        
        ### Get best model
        print("Melhor modelo AIC = " + str(best_AIC_model) + "\n")
        print("Melhor modelo AICc = " + str(best_AICc_model) + "\n")
        print("Melhor modelo BIC = " + str(best_BIC_model) + "\n")
        
        def count_decimal_places(number):
            return len(str(1e-3).split('.')[1])
        
        ### best model 
        print('*'*65)
        print('*'*16+f' The best model of trajectory: {traj_i} '+'*'*16)
        print('*'*65)
        model_best = model_set[best_BIC_model]
        model_best.print(precision=count_decimal_places(precision))
        print('*'*65)
        print('*'*65)
        
        print(f'threshold: {parameter_list[best_BIC_model]}')