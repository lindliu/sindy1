#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 00:07:25 2023

@author: dliu
"""

import sys
sys.path.insert(1, '../../GSINDy')
sys.path.insert(1, '../..')
sys.path.insert(1, '..')
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import func3__, monomial_poly, monomial_poly_name, monomial_all, monomial_all_name
from utils import monomial_all_0, monomial_all_1, monomial_all_0_name, monomial_all_1_name
from train_gsindy_2d import data_generator, model_selection_gsindy_2d

# monomial = monomial_all
# monomial_name = monomial_all_name
monomial = [monomial_poly, monomial_poly]
monomial_name = [monomial_poly_name, monomial_poly_name]
# monomial = [monomial_all_0, monomial_all_1]
# monomial_name = [monomial_all_0_name, monomial_all_1_name]




#### results from gsindy all together
path_gsindy_all = glob(os.path.join(f'results/coeff/gsindy_all_{monomial[0].__name__}*.npy'))
coeff_gsindy_all = np.load(path_gsindy_all[0])
num_traj, num_feature, num_basis = coeff_gsindy_all.shape

#### results from gsindy one by one
max_split = max([int(os.path.split(path_)[1].split('_')[-2]) for path_ in glob(os.path.join(f'results/coeff/gsindy_one_{monomial[0].__name__}*.npy'))])
n_split = max_split-2+1   ##4
coeff_gsindy_one = np.zeros([n_split, num_traj, num_feature, num_basis])
for j in range(num_traj):
    for k in range(n_split):
        path_ = glob(os.path.join(f'results/coeff/gsindy_one_{monomial[0].__name__}_{k+2}_{j}.npy'))[0]
        coeff_gsindy_one[k,j,:,:] = np.load(path_)





###########################################################################
############# Select model from different number of split #################
###########################################################################
dt = .1      ## 2,3,6,8;     1,2,7,9
t = np.arange(0,10,dt)

x0_list = [[.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1], [.4, 1]]
a_list = [(.2, -.6, -.5), (.4, -.8, -.7), (.2, -.6, -1), (.4, -.8, -1), (.4, -1, -1), (.6, -1, -1)]


func = func3__
real0 = "x'=b*y + a*x^2 + c*x^3 - xy^2"
real1 = "y'=x + a*y + b*x^2y + c*y^3"
real_list = [real0, real1]


save_path = f'results/gsindy_one_by_one_{monomial[0].__name__}_final.txt'
open(save_path, 'w').close()
for i in range(num_traj):
    model_set = coeff_gsindy_one[:,[i],:,:]
    x0 = [x0_list[i]]
    a = [a_list[i]]
    t_, x0, a, sol_org_list, num_traj, num_feature = data_generator(func, x0, t, a, real_list)
    
    ms, best_BIC_model = model_selection_gsindy_2d(x0, t_, a, real_list, monomial, monomial_name, \
                                                   model_set, sol_org_list)
        
    print(f'best trajectory {i} split {best_BIC_model+2}')
    # print(f'{coeff_gsindy_one[best_BIC_model,i,:,:]}')
    
    coef = coeff_gsindy_one[best_BIC_model,i,:,:]
    np.save(f'results/coeff/gsindy_{monomial[0].__name__}_{i}.npy', coef)


    mask0 = coef[0,:]!=0
    mask1= coef[1,:]!=0
    with open(save_path, "a") as file2:
        file2.writelines(['*'*15, f'result of trajectory {i} ', '*'*15, '\n'])
        file2.write(f'coef of feature 0: {coef[0,:][mask0]} \n')
        file2.write(f'basis of feature 0: {monomial_name[0][mask0]} \n')
        file2.write(f'coef of feature 1: {coef[1,:][mask1]} \n')
        file2.write(f'basis of feature 1: {monomial_name[1][mask1]} \n\n')