#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:55:57 2023

@author: dliu
"""


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


################### 3 variable ####################
# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,10,dt)
# x0 = [[.5, 1], [.5, 1], [.5, 1]]
# a = [(.2, -.6, -.5), (.4, -.8, -.7), (.6, -1, -1)]
# func = func3__
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y + c*y^3"
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-2
# precision = 1e-3
# deriv_spline = True#False#

################### 2 variable ####################
# alpha = .05
# dt = .1   ## 0,3
# t = np.arange(0,1.5,dt)
# x0 = [[.5, 1], [.5, 1], [.5, 1]]
# a = [(.16, .25), (.3, .4), (.3, .5)]
# func = func12_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + b*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,12,dt)
# x0 = [[.5, 1], [.5, 1], [.5, 1]]
# a = [(.2, -.6), (.4, -.8), (.6, -1)]
# func = func3_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=b*y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y + b*x^2y - y^3"
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-2
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,6,dt)
# x0 = [[4, 1], [4, 1], [4, 1], [4, 1]]
# a = [(.7,-.8), (1,-1), (.5,-.6), (1.5,-1.5)]
# func = func4_
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x + b*xy"
# real1 = "y'=b*y + a*xy"
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#


################### 1 variable ####################
# alpha = .05
# dt = .05   ## 0,3
# t = np.arange(0,2.3,dt)
# # x0 = [[.2, 1], [.2, 1], [.1, 1]]
# # a = [(.12,), (.16,), (.2,)]
# x0 = [[.2, 1], [.05, 1], [.1, 1]]
# a = [(.2,), (.2,), (.2,)]
# func = func1
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a + x^2"
# real1 = "y'=-y"
# threshold_sindy=5e-2
# threshold_similarity = 1e-3
# threshold_group = 1e-3
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1
# t = np.arange(0,2,dt)
# x0 = [[.5, 1], [.5, 1], [.5, 1]]
# a = [(.25,), (.3,), (.5,)]
# # a = [(.1,), (.25,), (.3,), (.5,)]
# func = func2
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=.2 + a*x^2"
# real1 = "y'=-y"
# threshold_sindy=7e-2
# threshold_group = 1e-2
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1      ## 2,3,6,8;     1,2,7,9
# t = np.arange(0,12,dt)
# x0 = [[.5, 1], [.5, 1], [.5, 1]]
# a = [(.2,), (.4,), (.6,)]
# func = func3
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=-y + a*x^2 - x^3 - xy^2"
# real1 = "y'=x + a*y - x^2y - y^3"    
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-2
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1    ## 1,4    2,4
# t = np.arange(0,5,dt)
# x0 = [[4, 1], [4, 1]]
# a = [(.7,), (1,)]#,(.5,),(.6,)]
# func = func4
# monomial = monomial_poly
# monomial_name = monomial_poly_name
# real0 = "x'=a*x - xy"
# real1 = "y'=-y + a*xy"    
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .2
# t = np.arange(0,3.3,dt)
# x0 = [[np.pi-.1, 0], [np.pi-.1, 0]]
# a = [(-.25,), (-.35,)]
# func = func6
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=a*y-5sin(x)"
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#

# alpha = .05
# dt = .1
# t = np.arange(0,6,dt)
# x0 = [[np.pi-.1, 0], [np.pi-.1, 0], [np.pi-.1, 0], [np.pi-.1, 0]]
# # a = [-5, -6]
# a = [(-.15,), (-1,), (-2,), (-5,)]
# func = func7
# monomial = monomial_trig
# monomial_name = monomial_trig_name
# real0 = "x'=y"
# real1 = "y'=-0.25*y+a*sin(x)"
# threshold_sindy=1e-2
# threshold_group = 1e-3
# threshold_similarity = 1e-3
# precision = 1e-3
# deriv_spline = True#False#
    
alpha = .05
dt = .05    
t = np.arange(0,20,dt)
x0 = [[1, -2], [1, -2], [1, -2], [1, -2]]
a = [(.3,), (.4,),(.5,),(.2,)]
func = func5
monomial = monomial_poly
monomial_name = monomial_poly_name
real0 = "x'=5*(x - y- a*x^3)"
real1 = "y'=.2*x"    
threshold_sindy=1e-2
threshold_group = 1e-3
threshold_similarity = 1e-3
precision = 1e-3
deriv_spline = True#False#   ### model selection process need to be improved. make it more focus on smaller model


### generate data ###
num_traj = len(a)
num_feature = len(x0[0])

from utils import get_multi_sol
sol_org_list = get_multi_sol(func, x0, t, a)

### plot data ###
fig, ax = plt.subplots(1,1,figsize=[6,3])
for i in range(num_traj):
    ax.plot(t, sol_org_list[i], 'o', markersize=1, label=f'{a[i]}')
ax.legend()
ax.text(1, .95, f'${real0}$', fontsize=12)
ax.text(1, .8, f'${real1}$', fontsize=12)



#%% compare to pysindy
import pysindy_ as ps
from pysindy_.feature_library import GeneralizedLibrary, PolynomialLibrary, CustomLibrary
from pysindy_.optimizers import STLSQ

if func.__name__ not in ['func6', 'func7']:
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
    functions = [lambda x,y: 1, lambda x,y: x, lambda x,y: y, lambda x,y: np.sin(x), lambda x,y: np.sin(y), \
                  lambda x,y: np.cos(x), lambda x,y: np.cos(y)]
    names = [lambda x,y: '1', lambda x,y: 'x', lambda x,y: 'y', lambda x,y: 'sin(x)', lambda x,y: 'sin(y)', \
              lambda x,y: 'cos(x)', lambda x,y: 'cos(y)']
    lib_custom = CustomLibrary(library_functions=functions, function_names=names)
    lib_generalized = GeneralizedLibrary([lib_custom])

if __name__ == "__main__":
    #%% 
    threshold_sindy_list =  [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    # threshold_sindy_list =  [1e-2]
    from utils import ode_solver, get_deriv
    
    ##################################
    ############# sindy ##############
    ##################################
    model_set = []
    parameter_list = []
    for threshold_sindy in threshold_sindy_list:
        optimizer = STLSQ(threshold=threshold_sindy, alpha=alpha)
        model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=optimizer)
        
        print(f'################### [SINDy] threshold: {threshold_sindy} ################')

        sol_list = []
        sol_deriv_list = []
        for traj_i in range(num_traj):
            sol_, t_ = ode_solver(func, x0[traj_i], t, a[traj_i])
            _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
            
            sol_list.append(sol_)
            sol_deriv_list.append(sol_deriv_)
            
        if ensemble:
            model.fit(sol_list, t=t_, x_dot=sol_deriv_list, ensemble=True, quiet=True, multiple_trajectories=True)
        else:
            model.fit(sol_list, t=t_, x_dot=sol_deriv_list, multiple_trajectories=True)
        model.print()
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
        if np.abs(model.coefficients()).sum()>1e3:
            ms.set_model_SSE(model_id, 1e5)
            continue
        
        sse_sum = 0
        ##### simulations #####
        for traj_i in range(num_traj):
            simulation = model.simulate(x0[traj_i], t=t, integrator="odeint")
        
            sse_sum += ms.compute_SSE(sol_list[traj_i], simulation)
        
        ms.set_model_SSE(model_id, sse_sum/num_traj)
        # ms.set_model_SSE(model_id, sse_sum)
    
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
    print('*'*16+f' The best model '+'*'*16)
    print('*'*65)
    model_best = model_set[best_BIC_model]
    model_best.print(precision=count_decimal_places(precision))
    print('*'*65)
    print('*'*65)
    
    print(f'threshold: {parameter_list[best_AIC_model]}')