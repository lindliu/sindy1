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


import model_2D
import utils
import LV_data_generation, VDP_data_generation
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import matplotlib 
from mpl_toolkits.mplot3d import Axes3D
import pickle 

font = {'size'   : 18}
matplotlib.rc('font', **font)