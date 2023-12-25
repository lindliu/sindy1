#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:04:31 2023

@author: dliu
"""

import os
import numpy as np
from glob import glob




mean_poly_all = np.zeros([9,5])
mean_mix_all = np.zeros([9,5])
for i in range(1,6):
    path_poly = glob(os.path.join('Results/average', f'mean_poly_Exp{i}*.npy'))[0]
    path_mix = glob(os.path.join('Results/average', f'mean_mix_Exp{i}*.npy'))[0]
    mean_poly_all[:,i-1] = np.load(path_poly).flatten()
    mean_mix_all[:,i-1] = np.load(path_mix).flatten()


mean_all = np.zeros([9,11],dtype=object)
mean_all[:,0] = ['rmes', 'Mp', 'Mr','rmes', 'Mp', 'Mr','rmes', 'Mp', 'Mr']
mean_all[:,1:6] = mean_poly_all
mean_all[:,6:] = mean_mix_all




def get_latex_line(input_list, print_type='.2f'):
    
    line = []
    line.append('$'+input_list[0]+'$')
    line.append('&')
    for ele in input_list[1:]:
        if print_type[-1]=='f':
            line.append(f'{ele:{print_type}}'.rstrip('0'))
        elif print_type[-1]=='e':
            ss = f'{ele:{print_type}}'
            if ss[-2]=='0':
                line.append(ss[:-2]+ss[-1])
            else:
                line.append(ss)
        line.append('&')
    line = line[:-1]
    line.append('\\\ \n')
    return line

save_path = os.path.join('Results/average', 'mean metrics table.txt')
open(save_path, 'w').close()

with open(save_path, "a") as file:
    file.writelines(['columns: system poly 1,2,3,4,5, system mix 1,2,3,4,5', '\n'])
    
    
    file.writelines(['\n', '*'*15, ' sindy ', '*'*15, '\n'])
    line_ = get_latex_line(mean_all[0], '.2e')
    file.writelines(line_)
    for i in range(1,3):
        line_ = get_latex_line(mean_all[i])
        file.writelines(line_)
        
        
    file.writelines(['\n', '*'*15, ' gsindy one ', '*'*15, '\n'])
    line_ = get_latex_line(mean_all[3], '.2e')
    file.writelines(line_)
    for i in range(4,6):
        line_ = get_latex_line(mean_all[i])
        file.writelines(line_)
        
        
    file.writelines(['\n', '*'*15, ' gsindy all ', '*'*15, '\n'])
    line_ = get_latex_line(mean_all[6], '.2e')
    file.writelines(line_)
    for i in range(7,9):
        line_ = get_latex_line(mean_all[i])
        file.writelines(line_)
    
            