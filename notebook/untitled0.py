#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:26:20 2023

@author: do0236li
"""

# Adding DySMHO repositories to the environment
import sys
# Insert path to directory here
import os
path_to_dysmho = os.path.split(os.getcwd())[0]
sys.path.insert(0, os.path.join(path_to_dysmho, 'DySMHO/DySMHO/model'))
sys.path.insert(0, os.path.join(path_to_dysmho, 'DySMHO/DySMHO/data'))

# Loading functions and packages
import model_2D
import utils
import B_data_generation
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt

# Define initial conditions for the 2 states 
y_init = [1, 1]

# Horizon length for optimization problem (arbitrary time units) 
horizon_length = 10  

# Number of sampling time periods taken per MHE step
time_steps = 20  

# Basis functions for dynamics of state 1 
basis_functions_y0 = [lambda y0,y1: 1, 
                  lambda y0, y1: y0,
                  lambda y0, y1: y1, 
                  lambda y0, y1: y0*y1,
                  lambda y0, y1: y0**2,
                  lambda y0, y1: y1**2,
                  lambda y0, y1: (y0**2)*y1,
                  lambda y0, y1: y0*(y1**2),
                  lambda y0, y1: y0**3,
                  lambda y0, y1: y0**4,
                  lambda y0, y1: np.exp(y0), 
                  lambda y0, y1: 1/y0, 
                  lambda y0, y1: np.sin(y0),
                  lambda y0, y1: np.cos(y0)]

# Basis functions for dynamics of state 2
basis_functions_y1 = [lambda y0,y1: 1, 
                  lambda y0, y1: y0,
                  lambda y0, y1: y1, 
                  lambda y0, y1: y0*y1,
                  lambda y0, y1: y0**2,
                  lambda y0, y1: y1**2,
                  lambda y0, y1: (y0**2)*y1,
                  lambda y0, y1: y0*(y1**2),
                  lambda y0, y1: y1**3,
                  lambda y0, y1: y1**4,
                  lambda y0, y1: np.exp(y1), 
                  lambda y0, y1: 1/y1, 
                  lambda y0, y1: np.sin(y1),
                  lambda y0, y1: np.cos(y1)]
# Basis function names
basis_functions_names_y0 = ['1','y0', 'y1', 'y0*y1', 'y0^2', 'y1^2', '(y0^2)*y1', 'y0*(y1^2)', ' y0^3',  'y0^4', 'exp(y0)', '1/y0', 'sin(y0)', 'cos(y0)']
basis_functions_names_y1 = ['1','y0', 'y1', 'y0*y1', 'y0^2', 'y1^2', '(y0^2)*y1', 'y0*(y1^2)', ' y1^3',  'y1^4', 'exp(y1)', '1/y1', 'sin(y1)', 'cos(y1)']
basis_y0 = {'functions': basis_functions_y0, 'names': basis_functions_names_y0} 
basis_y1 = {'functions': basis_functions_y1, 'names': basis_functions_names_y1}

# Data generation (time grid)
xs = np.linspace(0, horizon_length + time_steps, 1000 * (horizon_length + time_steps) + 1)
# Data generation (simulating true dynamics on the time grid with addition of white noise )
t, y = B_data_generation.data_gen(xs, [y_init[0], y_init[1]], [0, 0.05, 0, 0.05], False)
# Data generation (simulating true dynamics on the time grid without addition of white noise)
t_nf, y_nf = B_data_generation.data_gen(xs, [y_init[0], y_init[1]], [0, 0, 0, 0], False)

# Visualizing simulated data
plt.plot(t,y)
plt.show()


# Creating MHL class
B_example = model_2D.DySMHO(y,t, [basis_y0, basis_y1])
# Applying SV smoothing 
B_example.smooth()
# Pre-processing 1: generates features and tests for Granger Causality 
B_example.pre_processing_1()
# Pre-processing 2: uses OLS for initialization and for bounding parameters
B_example.pre_processing_2(significance = 0.8)
# # Calling for main discovery task
# B_example.discover(horizon_length, 
#                     time_steps, 
#                     data_step = 100, 
#                     optim_options = {'nfe': 60, 'ncp':15}, 
#                     thresholding_frequency = 10, 
#                     thresholding_tolerance = 1, 
#                     sign = True)
# # Validation of discovered equations
# B_example.validate(xs, y_nf, plot = True)
