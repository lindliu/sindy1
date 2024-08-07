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

from utils import get_deriv, get_theta
# MSE = lambda x: (np.array(x)**2).mean()
# MSE = lambda x: np.linalg.norm(x)


# import scipy
# def SLS(Theta, DXdt, threshold, alpha=.05):
#     n_feature = DXdt.shape[1]
#     Xi = ridge_regression(scipy.sparse.coo_matrix(Theta),DXdt, alpha=alpha).T
#     # Xi = np.linalg.lstsq(Theta,DXdt, rcond=None)[0]
#     # Xi = solve_minnonzero(Theta,DXdt)
#     Xi[np.abs(Xi)<threshold] = 0
#     # print(Xi)
#     for _ in range(20):
#         smallinds = np.abs(Xi)<threshold
#         Xi[smallinds] = 0
#         for ind in range(n_feature):
            
#             if Xi[:,ind].sum()==0:
#                 break
            
#             biginds = ~smallinds[:,ind]
            
#             Xi[biginds,ind] = ridge_regression(scipy.sparse.coo_matrix(Theta[:,biginds]), DXdt[:,ind], alpha=alpha).T
#             # Xi[biginds,ind] = np.linalg.lstsq(Theta[:,biginds],DXdt[:,ind], rcond=None)[0]
#             # Xi[biginds,ind] = solve_minnonzero(Theta[:,biginds],DXdt[:,ind])
        
#     reg = LinearRegression(fit_intercept=False)
#     ind_ = np.abs(Xi.T) > 1e-14
#     coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
#     for i in range(ind_.shape[0]):
#         if np.any(ind_[i]):
#             coef[i, ind_[i]] = reg.fit(scipy.sparse.coo_matrix(Theta[:, ind_[i]]), DXdt[:, i]).coef_
#     Xi = coef.T

#     return Xi

def SLS(Theta, DXdt, threshold, alpha=.05):
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
        
    reg = LinearRegression(fit_intercept=False)
    ind_ = np.abs(Xi.T) > 1e-14
    coef = np.zeros((DXdt.shape[1], Theta.shape[1]))
    for i in range(ind_.shape[0]):
        if np.any(ind_[i]):
            coef[i, ind_[i]] = reg.fit(Theta[:, ind_[i]], DXdt[:, i]).coef_
    Xi = coef.T

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

def plot_distribution(Xi0_group, nth_feature, epoch, basis_functions_name, idx):
    fig, ax = plt.subplots(5,5,figsize=(12,12), constrained_layout=True)
    ax = ax.flatten()
    for i in idx:
        ax[i].hist(list(Xi0_group[:,:,i]), alpha = 0.5, label=basis_functions_name[i])
        ax[i].set_title(basis_functions_name[i])
    fig.suptitle(f'{nth_feature}th feature with iteration:{epoch}', fontsize=20)



from sklearn.linear_model import Lasso
import pysindy_ as ps
from pysindy_.utils import AxesArray
from pysindy_.feature_library import Shell_custom_theta
def Axes_transfer(theta_):
    xp = AxesArray(
        np.empty(
            theta_.shape,
            dtype=np.float64,
            order='C',
        ),
        {'ax_time': 0, 'ax_coord': 1, 'ax_sample': None, 'ax_spatial': []},
    )
    
    num_basis = theta_.shape[1]
    for i in range(num_basis):
        xp[..., i] = theta_[:, i]
    
    theta = [] + [xp]
    
    return theta


class GSINDy():
    def __init__(self, basis, num_traj, num_feature, \
                     threshold_sindy=1e-2, threshold_group=1e-3, threshold_similarity=1e-2, \
                     alpha=0.05, deriv_spline=True, max_iter=20, optimizer='Manually', ensemble=False, normalize_columns=False):
        basis_functions_list = basis['functions']
        basis_functions_name_list = basis['names']
        
        if len(basis_functions_list)==1:
            self.basis_functions_0 = basis_functions_list[0]
            self.basis_functions_0_name = basis_functions_name_list[0]
            self.basis_functions_1 = basis_functions_list[0]
            self.basis_functions_1_name = basis_functions_name_list[0]
            self.basis_functions_2 = basis_functions_list[0]
            self.basis_functions_2_name = basis_functions_name_list[0]
            
        if len(basis_functions_list)==2:
            self.basis_functions_0 = basis_functions_list[0]
            self.basis_functions_0_name = basis_functions_name_list[0]
            self.basis_functions_1 = basis_functions_list[1]
            self.basis_functions_1_name = basis_functions_name_list[1]

        if len(basis_functions_list)==3:
            self.basis_functions_0 = basis_functions_list[0]
            self.basis_functions_0_name = basis_functions_name_list[0]
            self.basis_functions_1 = basis_functions_list[1]
            self.basis_functions_1_name = basis_functions_name_list[1]
            self.basis_functions_2 = basis_functions_list[2]
            self.basis_functions_2_name = basis_functions_name_list[2]

        self.num_traj = num_traj
        self.num_feature = num_feature
        self.threshold_sindy = threshold_sindy
        self.threshold_group = threshold_group
        self.threshold_similarity = threshold_similarity
        self.alpha = alpha
        self.deriv_spline = deriv_spline
        self.max_iter = 20
        
        assert optimizer in ['Manually', 'SQTL', 'LASSO', 'SR3'], 'No such optimizer'
        
        if optimizer=='Manually':
            self.optimizer = 'Manually'
        elif optimizer=='SQTL':
            self.optimizer = ps.STLSQ(threshold=self.threshold_sindy, alpha=self.alpha, normalize_columns=normalize_columns)
        elif optimizer=='LASSO':
            self.optimizer = Lasso(alpha=self.alpha, max_iter=5000, fit_intercept=False)
        elif optimizer=='SR3':
            self.optimizer = ps.SR3(threshold=self.threshold_sindy, nu=.1)
        
        self.ensemble = ensemble

                
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
    def get_multi_sub_series_2D(self, sol_org_list, t, num_series=60, window_per=.7):
        assert window_per<1, 'window_per has to be less than 1'
        
        num_traj = self.num_traj

        length = t[-1]-t[0]
        length_sub = length*window_per
        
        dt = t[1]-t[0]
        step = (1-window_per)/num_series 
        
        theta0, theta1 = [], []   ### num_series, length
        sol0_deriv, sol1_deriv = [], [] ### num_series, length
        for k in range(num_traj):
            for i in range(num_series):
                # t_new = np.sort(t[0] + np.random.rand(100)*length)
                t_new = np.linspace(length*(i*step), length*(i*step)+length_sub, num=int(length_sub//dt))
                # start = (length-length_sub)*np.random.rand()
                # t_new = np.linspace(start, start+length_sub, num=int(length_sub//dt))
                sol0_, sol0_deriv_ = data_interp(t_new, t, sol_org_list[k][:,0].squeeze(), self.deriv_spline)
                sol1_, sol1_deriv_ = data_interp(t_new, t, sol_org_list[k][:,1].squeeze(), self.deriv_spline)
        
                theta0_ = get_theta(np.c_[sol0_,sol1_], self.basis_functions_0)
                theta1_ = get_theta(np.c_[sol0_,sol1_], self.basis_functions_1)
                theta0.append(theta0_)
                theta1.append(theta1_)
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
        

    ### get data
    def get_multi_sub_series_3D(self, sol_org_list, t, num_series=60, window_per=.7):
        assert window_per<1, 'window_per has to be less than 1'
        
        num_traj = self.num_traj
    
        length = t[-1]-t[0]
        length_sub = length*window_per
        
        dt = t[1]-t[0]
        step = (1-window_per)/num_series 
        
        theta0, theta1, theta2 = [], [], []   ### num_series, length
        sol0_deriv, sol1_deriv, sol2_deriv = [], [], [] ### num_series, length
        for k in range(num_traj):
            for i in range(num_series):
                # t_new = np.sort(t[0] + np.random.rand(100)*length)
                t_new = np.linspace(length*(i*step), length*(i*step)+length_sub, num=int(length_sub//dt))
                # start = (length-length_sub)*np.random.rand()
                # t_new = np.linspace(start, start+length_sub, num=int(length_sub//dt))
                sol0_, sol0_deriv_ = data_interp(t_new, t, sol_org_list[k][:,0].squeeze(), self.deriv_spline)
                sol1_, sol1_deriv_ = data_interp(t_new, t, sol_org_list[k][:,1].squeeze(), self.deriv_spline)
                sol2_, sol2_deriv_ = data_interp(t_new, t, sol_org_list[k][:,2].squeeze(), self.deriv_spline)

                theta0_ = get_theta(np.c_[sol0_,sol1_,sol2_], self.basis_functions_0)
                theta1_ = get_theta(np.c_[sol0_,sol1_,sol2_], self.basis_functions_1)
                theta2_ = get_theta(np.c_[sol0_,sol1_,sol2_], self.basis_functions_2)
                theta0.append(theta0_)
                theta1.append(theta1_)
                theta2.append(theta2_)
                sol0_deriv.append(sol0_deriv_)
                sol1_deriv.append(sol1_deriv_)
                sol2_deriv.append(sol2_deriv_)

        theta0 = np.c_[theta0]
        theta1 = np.c_[theta1]
        theta2 = np.c_[theta2]
        sol0_deriv = np.c_[sol0_deriv]
        sol1_deriv = np.c_[sol1_deriv]
        sol2_deriv = np.c_[sol2_deriv]

        theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:]) ##num_traj, num_series, length_series, num_basis
        theta1 = theta1.reshape(num_traj, -1, *theta1.shape[1:])
        theta2 = theta2.reshape(num_traj, -1, *theta2.shape[1:])

        sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:]) ##num_traj, num_series, length_series, 1
        sol1_deriv = sol1_deriv.reshape(num_traj, -1, *sol1_deriv.shape[1:])
        sol2_deriv = sol2_deriv.reshape(num_traj, -1, *sol2_deriv.shape[1:])

        sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)
        sol1_deriv = np.vstack(sol1_deriv.transpose(0,2,1,3)).transpose(1,0,2) ##num_series, num_traj*length_series, 1
        sol2_deriv = np.vstack(sol2_deriv.transpose(0,2,1,3)).transpose(1,0,2) ##num_series, num_traj*length_series, 1

        theta_list = [theta0, theta1, theta2]
        sol_deriv_list = [sol0_deriv, sol1_deriv, sol2_deriv]
        
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
                
                
                if self.optimizer=='Manually':
                    ### using own solver, should be the same in sindy without ensemble case
                    Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.alpha)[...,0]
                else:
                    ### using pysindy solver
                    Theta_ = Axes_transfer(Theta)
                    lib_generalized = Shell_custom_theta(theta=Theta_)###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
                    model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=self.optimizer)
                    if self.ensemble:
                        model.fit(np.ones([1]), t=1, x_dot=dXdt, ensemble=True, quiet=True) ###first 2 inputs has no meanings, since theta and dXdt has been fixed
                    else:
                        model.fit(np.ones([1]), t=1, x_dot=dXdt) ###first 2 inputs has no meanings, since theta and dXdt has been fixed
                    Xi0_ = model.coefficients()[0,...]
        
        
                num_diff_ = idx_diff_activ.sum()*self.num_traj
                Xi0_group[:,j,idx_diff_activ] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
                Xi0_group[:,j,idx_same_activ] = Xi0_[num_diff_:]
            
            ### plot current frequency distribution for each basis functions
            if self.plot_dist and nth_feature==0:
                plot_distribution(Xi0_group, nth_feature, epoch, self.basis_functions_0_name, idx=self.idx_basis[idx_activ])
            if self.plot_dist and nth_feature==1:
                plot_distribution(Xi0_group, nth_feature, epoch, self.basis_functions_1_name, idx=self.idx_basis[idx_activ])
            if self.plot_dist and nth_feature==2:
                plot_distribution(Xi0_group, nth_feature, epoch, self.basis_functions_2_name, idx=self.idx_basis[idx_activ])
            
            ### group sparsity 
            idx_activ, Xi0_group_r = self.group_threshold(Xi0_group, num_series, num_basis)
            
            ### To find the identical basis
            idx_activ, idx_diff_activ, idx_same_activ = self.similarity_threshold(idx_activ, idx_same_activ, Xi0_group_r)
            
            ### check if convergent
            if (idx_same_activ==idx_same_activ_pre).all() and (idx_diff_activ==idx_diff_activ_pre).all():
                break
            else:
                idx_same_activ_pre = copy.deepcopy(idx_same_activ)
                idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)
            
            ### if no basis selected
            if not idx_activ.any():
                break
            
        return idx_activ, idx_same_activ, idx_diff_activ
    
    def group_threshold(self, Xi0_group, num_series, num_basis):
        ### remove part outliers of estimated parameters ###
        tail = int(num_series*self.remove_per) #### 
        Xi0_group = np.sort(Xi0_group, axis=1)

        Xi0_group_r = np.zeros([self.num_traj, num_series-tail, num_basis])
        for p in range(self.num_traj):
            for q in range(num_basis):
                i_min = self.find_interval(Xi0_group[p,:,q], tail)
                Xi0_group_r[p,:,q] = Xi0_group[p, i_min:num_series-(tail-i_min), q]
                
        ### group threshold ###
        idx_activ = np.abs(Xi0_group_r.mean(1)).mean(0)>self.threshold_group
        Xi0_group_r[:,:,~idx_activ] = 0
        
        ##### Xi0_group_r normalization for calculate distance of distributions ##### 
        norm_each_coef = np.linalg.norm(np.vstack(Xi0_group_r),axis=0)  ## num_basis
        # norm_each_coef = np.mean(np.abs(np.vstack(Xi0_group_r)),axis=0)  ## num_basis
        Xi0_group_r[:,:,idx_activ] = Xi0_group_r[:,:,idx_activ]/norm_each_coef[idx_activ]
        
        return idx_activ, Xi0_group_r
            
    def similarity_threshold(self, idx_activ, idx_same_activ, Xi0_group_r):
        idx_same_activ = np.logical_and(idx_activ, idx_same_activ)
        idx_diff_activ = copy.deepcopy(idx_activ)
        idx_diff_activ[idx_same_activ] = False
        
        from itertools import combinations
        from scipy.stats import wasserstein_distance
        radius = []
        for k in self.idx_basis[idx_diff_activ]:
            radius_ = []
            for i,j in combinations(np.arange(self.num_traj), 2):
                radius_.append(wasserstein_distance(Xi0_group_r[i,:,k],Xi0_group_r[j,:,k]))
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
            
        return idx_activ, idx_diff_activ, idx_same_activ
            
    def basis_identification(self, remove_per=.2, plot_dist=True):
        self.remove_per = remove_per
        self.plot_dist = plot_dist
        
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

    #%% get final predicted Xi
    def prediction(self, sol_org_list, t, split_basis=True):
        if split_basis:
            all_basis = copy.deepcopy(self.all_basis)
            diff_basis = copy.deepcopy(self.diff_basis)
            same_basis = copy.deepcopy(self.same_basis)
        else:
            all_basis = copy.deepcopy(self.all_basis)
            diff_basis = copy.deepcopy(self.all_basis)
            same_basis = [[] for _ in range(self.num_feature)]
            
        theta_org_list, sol_deriv_org_list = self.get_multi_theta(sol_org_list, t)        
        
        Xi_final = np.zeros([self.num_traj, self.num_feature, self.num_basis])
        for k, (theta_org_, sol_deriv_org_) in enumerate(zip(theta_org_list, sol_deriv_org_list)):
            
            ## if no basis selected
            if not all_basis[k].any():
                continue
            
            block_diff_list = [block_[:,diff_basis[k]] for block_ in theta_org_]
            block_diff = block_diag_multi_traj(block_diff_list)
            
            block_same_list = [block_[:,same_basis[k]] for block_ in theta_org_]
            block_same = np.vstack(block_same_list)
        
            Theta = np.c_[block_diff, block_same]
            dXdt = sol_deriv_org_
            
            if self.optimizer=='Manually':
                ### using own solver, should be the same in sindy without ensemble case
                Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.alpha)[...,0]
            else:
                ### using pysindy solver
                Theta_ = Axes_transfer(Theta)
                lib_generalized = Shell_custom_theta(theta=Theta_)  ###此处Shell_custom_theta只是壳，方便带入Theta_，无实际意义
                model = ps.SINDy(feature_names=["x", "y"], feature_library=lib_generalized, optimizer=self.optimizer)

                model.fit(np.ones([1]), t=1, x_dot=dXdt, ensemble=self.ensemble, quiet=True) ###first 2 inputs has no meanings

                Xi0_ = model.coefficients()[0,...]
                                
            num_diff_ = diff_basis[k].sum()*self.num_traj
            Xi_final[:,k,diff_basis[k]] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
            Xi_final[:,k,same_basis[k]] = Xi0_[num_diff_:]
        
        #############?????????????????################
    	# mask_tol = np.abs(Xi_final.mean(0))>self.precision  
        mask_tol = (Xi_final!=0).any(0)
        all_basis[0] = np.logical_and(mask_tol[0],all_basis[0])
        all_basis[1] = np.logical_and(mask_tol[1],all_basis[1])
        ##############################################
        
        if split_basis:
            self.all_basis = copy.deepcopy(all_basis)
        else:
            self.all_basis = copy.deepcopy(all_basis)
        return Xi_final
    
    def get_multi_theta(self, sol_org_list, t):
        theta_org_list = [[] for _ in range(self.num_feature)]
        sol_deriv_org_list = [[] for _ in range(self.num_feature)]
        for i in range(self.num_traj):
            sol_ = sol_org_list[i]
            _, sol_deriv_, _ = get_deriv(sol_, t, self.deriv_spline)
            
            for j in range(self.num_feature):
                if j==0:
                    theta_ = get_theta(sol_, self.basis_functions_0)
                if j==1:
                    theta_ = get_theta(sol_, self.basis_functions_1)
                if j==2:
                    theta_ = get_theta(sol_, self.basis_functions_2)
                    
                theta_org_list[j].append(theta_)
                sol_deriv_org_list[j].append(sol_deriv_[:,[j]])
                
        theta_org_list_ = [np.c_[theta_] for theta_ in theta_org_list]   ### num_feature, num_traj, length_series, num_basis
        sol_deriv_org_list_ = [np.vstack(sol_deriv_) for sol_deriv_ in sol_deriv_org_list] ### num_feature, num_traj*length_series, 1
        
        self.theta_org_list = theta_org_list_
        self.sol_deriv_org_list = sol_deriv_org_list
        return theta_org_list_, sol_deriv_org_list_
    