#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:48:50 2023

@author: do0236li
"""


# %%
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import ridge_regression, LinearRegression#, SGDRegressor

from utils import get_deriv
# MSE = lambda x: (np.array(x)**2).mean()
# MSE = lambda x: np.linalg.norm(x)

    
def SLS(Theta, DXdt, threshold, threshold_tol=1e-3, alpha=.05):
    n_feature = DXdt.shape[1]
    Xi = ridge_regression(Theta,DXdt, alpha=alpha).T
    # Xi = np.linalg.lstsq(Theta,DXdt, rcond=None)[0]
    # Xi = solve_minnonzero(Theta,DXdt)
    Xi[np.abs(Xi)<threshold] = 0
    # print(Xi)
    for _ in range(20):
        smallinds = np.abs(Xi)<threshold
        Xi[smallinds] = 0
        for ind in range(n_feature):
            
            if Xi[:,ind].sum()==0:
                break
            
            biginds = ~smallinds[:,ind]
            Xi[biginds,ind] = ridge_regression(Theta[:,biginds], DXdt[:,ind], alpha=alpha).T
            # Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],DXdt[:,ind], rcond=None)[0]
            # Xi[biginds,ind] = solve_minnonzero(Theta[:,biginds],DXdt[:,ind])
        
    # reg = LinearRegression(fit_intercept=False)
    # ind_ = np.abs(Xi.T) > 1e-14
    # ind_[:,:num_traj] = True
    # num_basis = ind_.sum()
    # while True:
    #     coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    #     for i in range(ind_.shape[0]):
    #         if np.any(ind_[i]):
    #             coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
                
    #             ind_[i, np.abs(coef[i,:])<threshold_tol] = False
    #             ind_[i,:num_traj] = True
        
    #     if num_basis==ind_.sum():
    #         break
    #     num_basis = ind_.sum()
        
    # Xi = coef.T
    # Xi[np.abs(Xi)<threshold_tol] = 0


    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-14
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T
    Xi[np.abs(Xi)<threshold_tol] = 0

    return Xi

import pynumdiff
def data_interp(x_new, x, y, deriv_spline=True):
    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    
    if deriv_spline:
        fd1 = f._spline.derivative(nu=1)
        return f(x_new), fd1(x_new)
    else:
        y_new = f(x_new)
        dx_new = x_new[1]-x_new[0]
        y_hat, dydx_hat = pynumdiff.finite_difference.second_order(y_new, dx_new)
        return y_new, dydx_hat.reshape([-1,1])
# data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())


#%%
import scipy
def block_diag(A, B):
    return scipy.linalg.block_diag(A,B)

def block_diag_multi_traj(A):
    if len(A)<=1:
        return A
    
    block = scipy.linalg.block_diag(A[0], A[1])
    for i in range(2, len(A)):
        block = block_diag(block, A[i])
    return block

def plot_distribution(Xi0_group, nth_feature, epoch, monomial_name, idx):
    fig, ax = plt.subplots(5,5,figsize=(12,12), constrained_layout=True)
    ax = ax.flatten()
    for i in idx:
        # for j in range(num_traj):
        ax[i].hist(list(Xi0_group[:,:,i]), alpha = 0.5, label=monomial_name[i])
        ax[i].set_title(monomial_name[i])
        # ax[i].legend()
    fig.suptitle(f'{nth_feature}th feature with iteration:{epoch}', fontsize=20)



class GSINDy():
    def __init__(self, monomial, monomial_name, \
                     num_traj, num_feature, \
                     threshold_sindy=1e-2, threshold_tol=1e-3, threshold_similarity=1e-2, \
                     alpha=0.05, deriv_spline=True, max_iter=20):
        
        self.monomial = monomial
        self.monomial_name = monomial_name
        self.num_traj = num_traj
        self.num_feature = num_feature
        self.threshold_sindy = threshold_sindy
        self.threshold_tol = threshold_tol
        self.threshold_similarity = threshold_similarity
        self.alpha = alpha
        self.deriv_spline = deriv_spline
        self.max_iter = 20

    def find_interval(self, series, tail):
        ### find i such that interval between index [i, length-(tail-i)] has minimum value
        ### for example: for list a=[-9, 0, 1, 1, 2, 3, 3, 4, 5, 10], if tail=2 then i=1
        ### because -9 and 10 is outliers, index between 1:9 cover most values with minimum interval [0, 5]
        length = series.shape[0]
        interval_len = []
        for i in range(tail):
            interval = series[i:length-(tail-i)]
            interval_len_ = interval[-1]-interval[0]
            interval_len.append(interval_len_)
        i_min = np.argmin(interval_len)
        return i_min
    
    ### get data
    def get_multi_sub_series(self, sol_org_list, t, num_series=60, window_per=.7):
        num_traj = self.num_traj

        length = t[-1]-t[0]
        length_sub = length*window_per
        
        dt = t[1]-t[0]
        # num_series = int(100*(1-window_per))*2
        step = (1-window_per)/num_series 
        
        theta0, theta1 = [], []   ### num_series, length
        sol0_deriv, sol1_deriv = [], [] ### num_series, length
        Xi0_list_, Xi1_list_ = [], []
        for k in range(num_traj):
            for i in range(num_series):
                # t_new = np.sort(t[0] + np.random.rand(100)*length)
                t_new = np.linspace(length*(i*step),length*(i*step)+length_sub, num=int(length_sub//dt))
                sol0_, sol0_deriv_ = data_interp(t_new, t, sol_org_list[k][:,0].squeeze(), self.deriv_spline)
                sol1_, sol1_deriv_ = data_interp(t_new, t, sol_org_list[k][:,1].squeeze(), self.deriv_spline)
        
                theta_ = self.monomial(np.c_[sol0_,sol1_])
                theta0.append(theta_)
                theta1.append(theta_)
                sol0_deriv.append(sol0_deriv_)
                sol1_deriv.append(sol1_deriv_)
        
        theta0 = np.c_[theta0]
        theta1 = np.c_[theta1]
        sol0_deriv = np.c_[sol0_deriv]
        sol1_deriv = np.c_[sol1_deriv]
        
        theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:]) ##num_traj, num_series, length_series, num_basis
        theta1 = theta1.reshape(num_traj, -1, *theta1.shape[1:])
        sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:]) ##num_traj, num_series, length_series, 1
        sol1_deriv = sol1_deriv.reshape(num_traj, -1, *sol1_deriv.shape[1:])
        
        sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)
        sol1_deriv = np.vstack(sol1_deriv.transpose(0,2,1,3)).transpose(1,0,2) ##num_series, num_traj*length_series, 1
        
        theta_list = [theta0, theta1]
        sol_deriv_list = [sol0_deriv, sol1_deriv]
        
        _, self.num_series, self.length_series, self.num_basis = theta0.shape
        self.idx_basis = np.arange(self.num_basis)
        
        self.theta_list = theta_list
        self.sol_deriv_list = sol_deriv_list
        

    def basis_identification_(self, theta_, sol_deriv_, nth_feature):
        num_traj, num_series, length_series, num_basis = theta_.shape
        
        idx_activ = np.ones([num_basis],dtype=bool)
        
        idx_same_activ = np.zeros_like(idx_activ,dtype=bool)
        idx_diff_activ = np.ones_like(idx_activ,dtype=bool)
        idx_same_activ_pre = copy.deepcopy(idx_same_activ)
        idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)
    
        for epoch in range(self.max_iter):
            ### do SINDy over each sub-series
            Xi0_group = np.zeros([self.num_traj, num_series, num_basis])
            for j in range(num_series):
                block_diff_list = [block_[j][:,idx_diff_activ] for block_ in theta_]
                block_diff = block_diag_multi_traj(block_diff_list)
                
                block_same_list = [block_[j][:,idx_same_activ] for block_ in theta_]
                block_same = np.vstack(block_same_list)
            
                Theta = np.c_[block_diff, block_same]
                dXdt = sol_deriv_[j]
                Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.threshold_tol, self.alpha)[...,0]
                
                num_diff_ = idx_diff_activ.sum()*self.num_traj
                Xi0_group[:,j,idx_diff_activ] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
                Xi0_group[:,j,idx_same_activ] = Xi0_[num_diff_:]
            
            plot_distribution(Xi0_group, nth_feature, epoch, self.monomial_name, idx=self.idx_basis[idx_activ])
            
            ### remove part outliers of estimated parameters ###
            tail = int(num_series*self.remove_per) #### 
            Xi0_group = np.sort(Xi0_group, axis=1)
            for p in range(self.num_traj):
                for q in range(num_basis):
                    i_min = self.find_interval(Xi0_group[p,:,q], tail)
                    Xi0_group[p,tail:,q] = Xi0_group[p, i_min:num_series-(tail-i_min), q]
                    Xi0_group[p,:tail,q] = 0  ##only use statistical meaning later, so doesn't mater where are zeros

            ### group threshold ###
            # idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_sindy)
            idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>self.threshold_tol)
            Xi0_group[:,:,~idx_activ] = 0
            
            ##### Xi0_group normalization for calculate distance of distributions ##### !!!!!! for func3
            norm_each_coef = np.linalg.norm(np.vstack(Xi0_group),axis=0)  ## num_basis
            # norm_each_coef = np.mean(np.abs(np.vstack(Xi0_group)),axis=0)  ## num_basis
            Xi0_group[:,:,idx_activ] = Xi0_group[:,:,idx_activ]/norm_each_coef[idx_activ]
            
            # plot_distribution(Xi0_group, nth_feature, epoch, self.monomial_name, idx=self.idx_basis[idx_activ])

            
            ##### To find the identical basis
            idx_same_activ = np.logical_and(idx_activ, idx_same_activ)
            idx_diff_activ = copy.deepcopy(idx_activ)
            idx_diff_activ[idx_same_activ] = False
            
            from itertools import combinations
            from scipy.stats import wasserstein_distance
            radius = []
            for k in self.idx_basis[idx_diff_activ]:
                radius_ = []
                for i,j in combinations(np.arange(self.num_traj), 2):
                    radius_.append(wasserstein_distance(Xi0_group[i,:,k],Xi0_group[j,:,k]))
                dist_ = scipy.cluster.hierarchy.linkage(radius_, method='single')[:,2]
                # radius.append(np.max(dist_))
                # radius.append(np.mean(dist_))
                radius.append(np.median(dist_))
            radius = np.array(radius)
            
            idx_similar = np.where(radius<self.threshold_similarity) 
            idx_same = self.idx_basis[idx_diff_activ][idx_similar]
            idx_same_activ[idx_same] = True
            idx_diff_activ[idx_same] = False
            
            assert (np.logical_or(idx_diff_activ, idx_same_activ) == idx_activ).any(0)
            
            if (idx_same_activ==idx_same_activ_pre).all() and (idx_diff_activ==idx_diff_activ_pre).all():
                break
            else:
                idx_same_activ_pre = copy.deepcopy(idx_same_activ)
                idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)
        
        return idx_activ, idx_same_activ, idx_diff_activ

    def basis_identification(self, remove_per=.2):
        self.remove_per = remove_per
        
        all_basis = []
        same_basis = []
        diff_basis = []
        for nth_feature, (theta_, sol_deriv_) in enumerate(zip(self.theta_list, self.sol_deriv_list)):
            idx_activ, idx_same_activ, idx_diff_activ = self.basis_identification_(theta_, sol_deriv_, nth_feature)
    
            all_basis.append(idx_activ)
            same_basis.append(idx_same_activ)
            diff_basis.append(idx_diff_activ)
        
        self.all_basis, self.same_basis, self.diff_basis = all_basis, same_basis, diff_basis  ## num_feature, num_basis
        # return all_basis, same_basis, diff_basis

# # for k in range(21):
# #     std = Xi0_group[0,:,k].std()
# #     mean = Xi0_group[0,:,k].mean()
# #     if mean!=0:
# #         print(f'{k} std: {std:.2f}, mean: {mean:.2f}, ratio: {std/mean:.2f}')

    #%% get final predicted Xi
    def prediction(self, sol_org_list, t):
        theta_org_list, sol_deriv_org_list = self.get_multi_theta(sol_org_list, t)
        
        Xi_final = np.zeros([self.num_traj, self.num_feature, self.num_basis])
        for k, (theta_org_, sol_deriv_org_) in enumerate(zip(theta_org_list, sol_deriv_org_list)):
            block_diff_list = [block_[:,self.diff_basis[k]] for block_ in theta_org_]
            block_diff = block_diag_multi_traj(block_diff_list)
            
            block_same_list = [block_[:,self.same_basis[k]] for block_ in theta_org_]
            block_same = np.vstack(block_same_list)
        
            Theta = np.c_[block_diff, block_same]
            dXdt = sol_deriv_org_
            Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.threshold_tol, self.alpha)[...,0]
            
            num_diff_ = self.diff_basis[k].sum()*self.num_traj
            Xi_final[:,k,self.diff_basis[k]] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
            Xi_final[:,k,self.same_basis[k]] = Xi0_[num_diff_:]
            
        mask_tol = np.abs(Xi_final.mean(0))>self.threshold_tol
        self.all_basis[0] = np.logical_and(mask_tol[0],self.all_basis[0])
        self.all_basis[1] = np.logical_and(mask_tol[1],self.all_basis[1])
    
        return Xi_final
    
    def get_multi_theta(self, sol_org_list, t):
        theta_org_list = [[] for _ in range(self.num_feature)]
        sol_deriv_org_list = [[] for _ in range(self.num_feature)]
        for i in range(self.num_traj):
            sol_ = sol_org_list[i]
            _, sol_deriv_, _ = get_deriv(sol_, t, self.deriv_spline)
            
            theta_ = self.monomial(sol_)
            for j in range(self.num_feature):
                theta_org_list[j].append(theta_)
                sol_deriv_org_list[j].append(sol_deriv_[:,[j]])
                
        theta_org_list = [np.c_[theta_] for theta_ in theta_org_list]   ### num_feature, num_traj, length_series, num_basis
        sol_deriv_org_list = [np.vstack(sol_deriv_) for sol_deriv_ in sol_deriv_org_list] ### num_feature, num_traj*length_series, 1
        return theta_org_list, sol_deriv_org_list
    