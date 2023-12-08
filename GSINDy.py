#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:48:50 2023

@author: do0236li
"""


# %%
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import ridge_regression, LinearRegression, SGDRegressor
from func import func1, func2, func3, func4, func5, func6, func7, \
    monomial_poly, monomial_trig, monomial_poly_name, monomial_trig_name


# MSE = lambda x: (np.array(x)**2).mean()
# MSE = lambda x: np.linalg.norm(x)

from scipy.integrate import odeint
def get_sol(func, x0, t, a, deriv_spline=True):
    sol = odeint(func, x0, t, args=a)
    return sol, t

def get_deriv(sol, t, deriv_spline=True):
    if deriv_spline:
        from scipy import interpolate
        f = interpolate.interp1d(t, sol.T, kind='cubic')
        fd1 = f._spline.derivative(nu=1)
        return f(t).T, fd1(t), t
    else:
        ###https://github.com/florisvb/PyNumDiff/blob/master/examples/1_basic_tutorial.ipynb
        import pynumdiff
        sol_deriv = np.zeros_like(sol)
        dt = t[1]-t[0]
        for i in range(sol.shape[1]):
            # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol1[:,i], dt)
            x_hat, dxdt_hat = pynumdiff.finite_difference.second_order(sol[:,i], dt)
            # x_hat, dxdt_hat = pynumdiff.finite_difference.first_order(sol[:,i], dt, params=[50], options={'iterate': True})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.gaussiandiff(sol[:,i], dt, params=[20], options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.friedrichsdiff(sol[:,i], dt, params=10, options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(sol[:,i], dt, params=[3, 0.09], options={'iterate': False})
            # x_hat, dxdt_hat = pynumdiff.linear_model.spectraldiff(sol[:,i], dt, params=[0.05])
            # x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(sol[:,i], dt, params=[2, 10, 10])
    
            sol_deriv[:,i] = dxdt_hat
        
        return sol, sol_deriv, t
    
def SLS(Theta, DXdt, threshold, alpha=.05, threshold_tol=1e-3):
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

def data_interp(x_new, x, y, deriv_spline=True):
    from scipy import interpolate
    f = interpolate.interp1d(x, y, kind='cubic')
    
    if deriv_spline:
        fd1 = f._spline.derivative(nu=1)
        return f(x_new), fd1(x_new)
    else:
        import pynumdiff
        y_new = f(x_new)
        dx_new = x_new[1]-x_new[0]
        y_hat, dydx_hat = pynumdiff.finite_difference.second_order(y_new, dx_new)
        return y_new, dydx_hat.reshape([-1,1])
# data_interp(np.linspace(0,t[-1]), t, sol0[2].squeeze())


#%%
def get_series(a, x0, t, func, monomial, real0, real1, deriv_spline=True):
    num_traj = len(a)
    num_feature = len(x0)
    
    sol_org_list = [[] for _ in range(num_feature)]
    theta_org_list = [[] for _ in range(num_feature)]
    sol_deriv_org_list = [[] for _ in range(num_feature)]
    for i in range(num_traj):
        sol_, _ = get_sol(func, x0, t, a[i], deriv_spline)
        _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
        
        theta_ = monomial(sol_)
    
        for j in range(num_feature):
            sol_org_list[j].append(sol_[:,[j]])
            theta_org_list[j].append(theta_)
            sol_deriv_org_list[j].append(sol_deriv_[:,[j]])

        plt.plot(t, sol_, 'o', markersize=1, label=f'{a[i]}')
    plt.legend()
    plt.text(1, .95, f'${real0}$', fontsize=12)
    plt.text(1, .8, f'${real1}$', fontsize=12)
    
    theta_org_list = [np.c_[theta_] for theta_ in theta_org_list]
    sol_deriv_org_list = [np.vstack(sol_deriv_) for sol_deriv_ in sol_deriv_org_list]
    
    return sol_org_list, theta_org_list, sol_deriv_org_list



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

def plot(Xi0_group, nth_feature, epoch, monomial_name, idx):
    fig, ax = plt.subplots(5,5,figsize=(12,12), constrained_layout=True)
    ax = ax.flatten()
    for i in idx:
        # for j in range(num_traj):
        ax[i].hist(list(Xi0_group[:,:,i]), alpha = 0.5, label=monomial_name[i])
        ax[i].set_title(monomial_name[i])
        # ax[i].legend()
    fig.suptitle(f'{nth_feature}th feature with iteration:{epoch}', fontsize=20)



class GSINDy():
    def __init__(self, monomial, monomial_name, threshold_sindy=1e-2, threshold_tol=1e-3, threshold_similarity=1e-2, deriv_spline=True, max_iter=20):
        self.monomial = monomial
        self.monomial_name = monomial_name
        self.threshold_sindy = threshold_sindy
        self.threshold_tol = threshold_tol
        self.threshold_similarity = threshold_similarity
        self.deriv_spline = deriv_spline
        self.max_iter = 20

    def find_interval(self, series, tail):
        length = series.shape[0]
        interval_len = []
        for i in range(tail):
            interval = series[i:length-(tail-i)]
            interval_len_ = interval[-1]-interval[0]
            interval_len.append(interval_len_)
        i_min = np.argmin(interval_len)
        return i_min
    
    ### get data
    def get_multi_series(self, sol_org_list, t, per=.7):
        per = .7
        length = t[-1]-t[0]
        length_sub = length*per
        
        dt = t[1]-t[0]
        num_traj = len(sol_org_list[0])
        
        num_series = int(100*(1-per))*2
        theta0, theta1 = [], []   ### num_series, length
        sol0_deriv, sol1_deriv = [], [] ### num_series, length
        Xi0_list_, Xi1_list_ = [], []
        for k in range(num_traj):
            for i in range(num_series):
                # t_new = np.sort(t[0] + np.random.rand(100)*length)
                t_new = np.linspace(length*(i*.005),length*(per+i*.005), num=int(length_sub//dt))
                sol0_, sol0_deriv_ = data_interp(t_new, t, sol_org_list[0][k].squeeze(), self.deriv_spline)
                sol1_, sol1_deriv_ = data_interp(t_new, t, sol_org_list[1][k].squeeze(), self.deriv_spline)
        
                theta_ = self.monomial(np.c_[sol0_,sol1_])
                theta0.append(theta_)
                theta1.append(theta_)
                sol0_deriv.append(sol0_deriv_)
                sol1_deriv.append(sol1_deriv_)
        
        theta0 = np.c_[theta0]
        theta1 = np.c_[theta1]
        sol0_deriv = np.c_[sol0_deriv]
        sol1_deriv = np.c_[sol1_deriv]
        
        theta0 = theta0.reshape(num_traj, -1, *theta0.shape[1:])
        theta1 = theta1.reshape(num_traj, -1, *theta1.shape[1:])
        sol0_deriv = sol0_deriv.reshape(num_traj, -1, *sol0_deriv.shape[1:])
        sol1_deriv = sol1_deriv.reshape(num_traj, -1, *sol1_deriv.shape[1:])
        
        sol0_deriv = np.vstack(sol0_deriv.transpose(0,2,1,3)).transpose(1,0,2)
        sol1_deriv = np.vstack(sol1_deriv.transpose(0,2,1,3)).transpose(1,0,2)
        
        
        theta_list = [theta0, theta1]
        sol_deriv_list = [sol0_deriv, sol1_deriv]
        # num_feature = len(theta_list)
        # assert num_feature==len(x0)

        self.num_feature = len(theta_list)
        self.num_traj, self.num_series, self.length_series, self.num_basis = theta0.shape
        
        self.idx_basis = np.arange(self.num_basis)
        
        self.theta_list = theta_list
        self.sol_deriv_list = sol_deriv_list
        # return theta_list, sol_deriv_list

    def basis_identification_(self, theta_, sol_deriv_, nth_feature):
        num_traj, num_series, length_series, num_basis = theta_.shape
        
        idx_activ = np.ones([num_basis],dtype=bool)
        
        idx_same_activ = np.zeros_like(idx_activ,dtype=bool)
        idx_diff_activ = np.ones_like(idx_activ,dtype=bool)
        idx_same_activ_pre = copy.deepcopy(idx_same_activ)
        idx_diff_activ_pre = copy.deepcopy(idx_diff_activ)
    
        for epoch in range(self.max_iter):
            ### do SINDy over each sub-series
            Xi0_group = np.zeros([self.num_traj, self.num_series, self.num_basis])
            for j in range(num_series):
                block_diff_list = [block_[j][:,idx_diff_activ] for block_ in theta_]
                block_diff = block_diag_multi_traj(block_diff_list)
                
                block_same_list = [block_[j][:,idx_same_activ] for block_ in theta_]
                block_same = np.vstack(block_same_list)
            
                Theta = np.c_[block_diff, block_same]
                dXdt = sol_deriv_[j]
                Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.threshold_tol)[...,0]
                
                num_diff_ = idx_diff_activ.sum()*self.num_traj
                Xi0_group[:,j,idx_diff_activ] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
                Xi0_group[:,j,idx_same_activ] = Xi0_[num_diff_:]
            
            plot(Xi0_group, nth_feature, epoch, self.monomial_name, idx=self.idx_basis[idx_activ])
            
            ### remove part outliers of estimated parameters
            tail = int(num_series*.2) #### can be improved by more advanced method!!!!!!!!!!!!!!
            Xi0_group = np.sort(Xi0_group, axis=1)
            for p in range(self.num_traj):
                for q in range(self.num_basis):
                    i_min = self.find_interval(Xi0_group[p,:,q], tail)
                    Xi0_group[p,tail:,q] = Xi0_group[p, i_min:num_series-(tail-i_min), q]
                    Xi0_group[p,:tail,q] = 0
            # idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_sindy)
            idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>self.threshold_tol)
            
            # plot(Xi0_group, nth_feature, epoch, monomial_name)
    
            # ### remove part outliers of estimated parameters
            # tail = int(num_series*.1) #### can be improved by more advanced method!!!!!!!!!!!!!!
            # Xi0_group = np.sort(Xi0_group, axis=1)[:,tail:-tail,:]
            # # idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_sindy)
            # idx_activ = (np.abs(Xi0_group.mean(0).mean(0))>threshold_tol)
            
            ##### Xi0_group normalization for calculate distance of distributions#####
            Xi0_group[:,:,~idx_activ] = 0
            norm_each_coef = np.linalg.norm(np.vstack(Xi0_group),axis=0)
            Xi0_group[:,:,idx_activ] = Xi0_group[:,:,idx_activ]/norm_each_coef[idx_activ]
              
            
            ##### To find the identical basis
            idx_same_activ = np.logical_and(idx_activ, idx_same_activ)
            idx_diff_activ = copy.deepcopy(idx_activ)
            idx_diff_activ[idx_same_activ] = False
            
            from itertools import combinations
            from scipy.stats import wasserstein_distance
            radius = []
            for k in self.idx_basis[idx_diff_activ]:
                radius_ = []
                for i,j in combinations(np.arange(num_traj), 2):
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

    def basis_identification(self):
        all_basis = []
        same_basis = []
        diff_basis = []
        for nth_feature, (theta_, sol_deriv_) in enumerate(zip(self.theta_list, self.sol_deriv_list)):
            idx_activ, idx_same_activ, idx_diff_activ = self.basis_identification_(theta_, sol_deriv_, nth_feature)
    
            all_basis.append(idx_activ)
            same_basis.append(idx_same_activ)
            diff_basis.append(idx_diff_activ)
        
        self.all_basis, self.same_basis, self.diff_basis = all_basis, same_basis, diff_basis
        return all_basis, same_basis, diff_basis

# # for k in range(21):
# #     std = Xi0_group[0,:,k].std()
# #     mean = Xi0_group[0,:,k].mean()
# #     if mean!=0:
# #         print(f'{k} std: {std:.2f}, mean: {mean:.2f}, ratio: {std/mean:.2f}')

    #%% get final predicted Xi
    def prediction(self, theta_org_list, sol_deriv_org_list):
        
        # _, sol_deriv_, _ = get_deriv(sol_, t, deriv_spline)
        
        
        
        Xi_final = np.zeros([self.num_traj, self.num_feature, self.num_basis])
        for k, (theta_org_, sol_deriv_org_) in enumerate(zip(theta_org_list, sol_deriv_org_list)):
            block_diff_list = [block_[:,self.diff_basis[k]] for block_ in theta_org_]
            block_diff = block_diag_multi_traj(block_diff_list)
            
            block_same_list = [block_[:,self.same_basis[k]] for block_ in theta_org_]
            block_same = np.vstack(block_same_list)
        
            Theta = np.c_[block_diff, block_same]
            dXdt = sol_deriv_org_
            Xi0_ = SLS(Theta, dXdt, self.threshold_sindy, self.threshold_tol)[...,0]
            
            num_diff_ = self.diff_basis[k].sum()*self.num_traj
            Xi_final[:,k,self.diff_basis[k]] = Xi0_[:num_diff_].reshape([self.num_traj,-1])
            Xi_final[:,k,self.same_basis[k]] = Xi0_[num_diff_:]
            
        mask_tol = np.abs(Xi_final.mean(0))>self.threshold_tol
        self.all_basis[0] = np.logical_and(mask_tol[0],self.all_basis[0])
        self.all_basis[1] = np.logical_and(mask_tol[1],self.all_basis[1])
    
        return Xi_final
    
    
    