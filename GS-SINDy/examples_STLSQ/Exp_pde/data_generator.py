#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:29:41 2024

@author: do0236li
"""

# https://www.sciencedirect.com/science/article/pii/S0021999121004204
# https://scipy-cookbook.readthedocs.io/items/KdV.html
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
import matplotlib.pyplot as plt



############################ Inviscid_Burgers ########################
def Inviscid_Burgers(x, t, A=1000, alpha=0.5):
    """ shock-forming solution for Inviscid_Burgers du/dt=-u*ux """
    if t>=max(1/A*x+1/alpha, 2/A*x+1/alpha):
        u = A
        
    elif x>=A*(t-1/alpha) and x<=0:
        u = -alpha*x/(1-alpha*t)
        
    else:
        u = 0
    
    return u

dx = 31.25
dt = 0.0157        
x = np.arange(-4000+dx, -4000+256*dx, dx)
t = np.arange(dt, 256*dt, dt)

u_IB = np.zeros([x.shape[0], t.shape[0]])
for i in range(x.shape[0]):
    for j in range(t.shape[0]):
        u_IB[i,j] = Inviscid_Burgers(x[i], t[j])

plt.figure(figsize=(6,5))
plt.pcolormesh(t, x, u_IB)
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
# plt.axis('normal')
plt.title('Inviscid Burgers')
plt.show()

np.savez('data/IB_1.npz', t=t, x=x, usol=u_IB)
#######################################################################




############################ Korteweg-de Vries ########################
def kdv_init(x):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    # u = c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    
    A, B = 25, 16
    u = 3 * A**2 * 1/np.cosh(0.5*A*(x+2))**2 + 3 * B**2 * 1/np.cosh(0.5*B*(x+1))**2
    return u

def kdv(u, t, L, c=[-1,-1]):
    """Differential equations for the Korteweg-de Vries equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.    
    dudt = c[0]*u*ux + c[1]*uxxx

    return dudt

L = 2*np.pi
x = np.linspace(L-3*np.pi, L-np.pi, 401)[:-1]
t = np.linspace(0, 0.006, 2400+1)
u0 = kdv_init(x)
# plt.plot(u0)

sol_kvd = odeint(kdv, u0, t, args=(L,), mxstep=5000)
plt.imshow(sol_kvd.T[:,::4])

np.savez('data/KDV_1.npz', t=t[::4], x=x, usol=sol_kvd.T[:,::4])
#######################################################################



########################### Kuramoto-Sivashinsky #######################
def ks_init(x):
    u = np.cos(x/16)*(1+np.sin(x/16))
    return u

def ks(u, t, L, c=[-1,-1,-1]):
    """Differential equations for the Kuramoto-Sivashinsky equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxx = psdiff(u, period=L, order=2)
    uxxxx = psdiff(u, period=L, order=4)
    
    # Compute du/dt.    
    dudt = c[0]*u*ux + c[1]*uxx + c[2]*uxxxx
    
    return dudt

L = 32*np.pi
x = np.linspace(0, L, 256+1)[1:]
t = np.linspace(0, 150, 1500+1)
u0 = ks_init(x)
# plt.plot(u0)
    
sol_ks = odeint(ks, u0, t, args=(L,), mxstep=5000)
plt.imshow(sol_ks.T[:,::5])
    
np.savez('data/KS_1.npz', t=t[::5], x=x, usol=sol_ks.T[:,::5])
#######################################################################



