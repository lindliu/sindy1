#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:04:31 2023

@author: dliu
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


file_noise = ['GS-SINDy', 'GS-SINDy_001', 'GS-SINDy_005', 'GS-SINDy_01']
mean_mix_all = np.zeros([len(file_noise), 12, 6])
mean_poly_all = np.zeros([len(file_noise), 12, 6])

for j, file in enumerate(file_noise):
    path_base = os.path.join(f'./{file}/examples_STLSQ/Analysis/Results')
    
    for i, idx in enumerate([1,2,3,4,5,7]):
        path_mix = glob(os.path.join(path_base, 'average', f'mean_mix_Exp{idx}*.npy'))[0]
        mean_mix_all[j,:,i] = np.load(path_mix).flatten()
    

    for i, idx in enumerate([1,2,3,4,5,7]):
        path_poly = glob(os.path.join(path_base, 'average', f'mean_poly_Exp{idx}*.npy'))[0]
        mean_poly_all[j,:,i] = np.load(path_poly).flatten()
    
    
    

fig, ax = plt.subplots(2,6,figsize=[15,6], sharex='col')


idx = 0  ### rmse

exp = 0  ### example number
ax[idx,0].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5, label='M1')
ax[idx,0].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5, label='M2')
ax[idx,0].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5, label='M3')
ax[idx,0].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5, label='M4')
ax[idx,0].set_xticks([0,1,2,3])
ax[idx,0].legend()

exp = 2  ### example number
ax[idx,1].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,1].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,1].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,1].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,1].set_xticks([0,1,2,3])
ax[idx,1].set_yticklabels([-2,0,2,4,6,8,10,12])

exp = 3  ### example number
ax[idx,2].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,2].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,2].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,2].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,2].set_xticks([0,1,2,3])

exp = 4  ### example number
ax[idx,3].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,3].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,3].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,3].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,3].set_xticks([0,1,2,3])

exp = 1  ### example number
ax[idx,4].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,4].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,4].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,4].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,4].set_xticks([0,1,2,3])

exp = 5  ### example number
ax[idx,5].plot(np.log(mean_mix_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,5].plot(np.log(mean_mix_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,5].plot(np.log(mean_mix_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,5].plot(np.log(mean_mix_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,5].set_xticks([0,1,2,3])






idx = 1  ### Mp

exp = 0  ### example number
ax[idx,0].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5, label='M1')
ax[idx,0].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5, label='M2')
ax[idx,0].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5, label='M3')
ax[idx,0].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5, label='M4')
ax[idx,0].set_xticks([0,1,2,3])
ax[idx,0].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)
ax[idx,0].legend()

exp = 2  ### example number
ax[idx,1].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,1].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,1].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,1].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,1].set_xticks([0,1,2,3])
ax[idx,1].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 3  ### example number
ax[idx,2].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,2].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,2].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,2].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,2].set_xticks([0,1,2,3])
ax[idx,2].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 4  ### example number
ax[idx,3].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,3].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,3].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,3].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,3].set_xticks([0,1,2,3])
ax[idx,3].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 1  ### example number
ax[idx,4].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,4].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,4].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,4].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,4].set_xticks([0,1,2,3])
ax[idx,4].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 5  ### example number
ax[idx,5].plot(mean_mix_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,5].plot(mean_mix_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,5].plot(mean_mix_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,5].plot(mean_mix_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,5].set_xticks([0,1,2,3])
ax[idx,5].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

for a in ax[1,1:]:
    a.sharey(ax[1,0])
    

plt.subplots_adjust(wspace=.25, hspace=.1)

pad = 5
# rows = ['Mexico\n(estimated cases)', 'South Africa\n(estimated cases)', 'South Korea\n(estimated cases)']
rows = ['RMSE (log)', 'Mp']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=15, ha='center', va='center', rotation=90)

cols = ['Lotka-Volterra', 'Brusselator','Van der Pol','Lorenz', 'Hopf', 'FitzHugh']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='center', va='baseline')


# fig.tight_layout()
fig.savefig(f'./noise_level_mix.png', bbox_inches='tight', dpi=300)









fig, ax = plt.subplots(2,6,figsize=[15,6], sharex='col')


idx = 0  ### rmse

exp = 0  ### example number
ax[idx,0].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5, label='M1')
ax[idx,0].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5, label='M2')
ax[idx,0].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5, label='M3')
ax[idx,0].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5, label='M4')
ax[idx,0].set_xticks([0,1,2,3])
ax[idx,0].legend()

exp = 2  ### example number
ax[idx,1].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,1].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,1].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,1].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,1].set_xticks([0,1,2,3])
ax[idx,1].set_yticklabels([-2,0,2,4,6,8,10,12])

exp = 3  ### example number
ax[idx,2].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,2].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,2].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,2].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,2].set_xticks([0,1,2,3])

exp = 4  ### example number
ax[idx,3].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,3].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,3].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,3].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,3].set_xticks([0,1,2,3])

exp = 1  ### example number
ax[idx,4].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,4].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,4].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,4].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,4].set_xticks([0,1,2,3])

exp = 5  ### example number
ax[idx,5].plot(np.log(mean_poly_all[:,idx,   exp]), color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,5].plot(np.log(mean_poly_all[:,idx+3, exp]), color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,5].plot(np.log(mean_poly_all[:,idx+6, exp]), color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,5].plot(np.log(mean_poly_all[:,idx+9, exp]), color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,5].set_xticks([0,1,2,3])






idx = 1  ### Mp

exp = 0  ### example number
ax[idx,0].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5, label='M1')
ax[idx,0].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5, label='M2')
ax[idx,0].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5, label='M3')
ax[idx,0].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5, label='M4')
ax[idx,0].set_xticks([0,1,2,3])
ax[idx,0].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)
ax[idx,0].legend()

exp = 2  ### example number
ax[idx,1].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,1].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,1].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,1].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,1].set_xticks([0,1,2,3])
ax[idx,1].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 3  ### example number
ax[idx,2].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,2].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,2].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,2].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,2].set_xticks([0,1,2,3])
ax[idx,2].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 4  ### example number
ax[idx,3].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,3].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,3].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,3].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,3].set_xticks([0,1,2,3])
ax[idx,3].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 1  ### example number
ax[idx,4].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,4].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,4].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,4].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,4].set_xticks([0,1,2,3])
ax[idx,4].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

exp = 5  ### example number
ax[idx,5].plot(mean_poly_all[:,idx,   exp], color='r', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'r', markersize = 5)
ax[idx,5].plot(mean_poly_all[:,idx+3, exp], color='tab:blue', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'tab:blue', markersize = 5)
ax[idx,5].plot(mean_poly_all[:,idx+6, exp], color='green', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'green', markersize = 5)
ax[idx,5].plot(mean_poly_all[:,idx+9, exp], color='pink', linestyle='dashed',
                 marker = 'o', markerfacecolor = 'pink', markersize = 5)
ax[idx,5].set_xticks([0,1,2,3])
ax[idx,5].set_xticklabels(['0','$10^{-2}$',r'$5\times 10^{-2}$','$10^{-1}$'], fontsize=9)

for a in ax[1,1:]:
    a.sharey(ax[1,0])
    

plt.subplots_adjust(wspace=.25, hspace=.1)

pad = 5
# rows = ['Mexico\n(estimated cases)', 'South Africa\n(estimated cases)', 'South Korea\n(estimated cases)']
rows = ['RMSE (log)', 'Mp']
for ax_, row in zip(ax[:,0], rows):
    ax_.annotate(row, xy=(0, 0.5), xytext=(-ax_.yaxis.labelpad - pad, 0),
                xycoords=ax_.yaxis.label, textcoords='offset points',
                fontsize=15, ha='center', va='center', rotation=90)

cols = ['Lotka-Volterra', 'Brusselator','Van der Pol','Lorenz', 'Hopf', 'FitzHugh']
for ax_, col in zip(ax[0], cols):
    ax_.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                fontsize=15, ha='center', va='baseline')


# fig.tight_layout()
fig.savefig(f'./noise_level_poly.png', bbox_inches='tight', dpi=300)


