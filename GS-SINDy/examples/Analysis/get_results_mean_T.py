#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:04:31 2023

@author: dliu
"""

import os
import numpy as np
from glob import glob



path_base = os.path.join('Results_T')

mean_poly_all = np.zeros([9,5])
mean_mix_all = np.zeros([9,5])
for i in range(1,6):
    path_poly = glob(os.path.join(path_base, 'average', f'mean_poly_Exp{i}*.npy'))[0]
    path_mix = glob(os.path.join(path_base, 'average', f'mean_mix_Exp{i}*.npy'))[0]
    mean_poly_all[:,i-1] = np.load(path_poly).flatten()
    mean_mix_all[:,i-1] = np.load(path_mix).flatten()


mean_all = np.zeros([9,11],dtype=object)
mean_all[:,0] = ['$RMES$', '$Mp$', '$Mr$', '$RMES$', '$Mp$', '$Mr$', '$RMES$', '$Mp$', '$Mr$']
mean_all[:,1:6] = mean_poly_all
mean_all[:,6:] = mean_mix_all

mean_all = mean_all.T

table_mean_all = np.zeros([11,10], dtype=object)
table_mean_all[:,0] = ['Metric', 'System 1', 'System 2', 'System 3','System 4','System 5', \
                      'System 1', 'System 2', 'System 3','System 4','System 5']
table_mean_all[:,1:] = mean_all

def get_latex_line(input_list):
    print_type = ['.2e', '.2f', '.2f', '.2e', '.2f', '.2f', '.2e', '.2f', '.2f']
    
    line = ['&']
    line.append(input_list[0])
    line.append('&')
    for i, ele in enumerate(input_list[1:]):
        if print_type[i][-1]=='f':
            line.append(f'{ele:{print_type[i]}}'.rstrip('0'))
        elif print_type[i][-1]=='e':
            ss = f'{ele:{print_type[i]}}'
            if ss[-2]=='0':
                line.append(ss[:-2]+ss[-1])
            else:
                line.append(ss)
        line.append('&')
    line = line[:-1]
    line.append("\\\ \n")
    return line

save_path = os.path.join(path_base, 'average', 'mean metrics table.txt')
open(save_path, 'w').close()

with open(save_path, "a") as file:
    file.write(f'Mean of 6 trajectories for each system \n')
    file.writelines(['columns: M1, M2, M3', '\n', '\n'])

    
    # line_first = [table_metrics[0,0]]
    line_first = ['&'+ele for ele in table_mean_all[0,:]]
    line_first.append("\\\ \n")
    file.writelines(line_first)
    file.writelines(['\n', '\midrule', '\n','\n'])
    file.write('\multirow{6}{*}{\\rotatebox[origin=c]{90}{Polynomial}} \n')
    for i in range(1,6):
        line_ = get_latex_line(table_mean_all[i,:])
        file.writelines(line_)
        
    file.writelines(['\n', '\midrule', '\n','\n'])
    file.write('\multirow{6}{*}{\\rotatebox[origin=c]{90}{Mixed}} \n')
    for i in range(6,11):
        line_ = get_latex_line(table_mean_all[i,:])
        file.writelines(line_)
            
            
            
            
        