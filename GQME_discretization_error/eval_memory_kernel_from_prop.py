from GQME_discretization_error.heom import *
from GQME_discretization_error.ttm import * 

import numpy as np
import scipy as sp
import h5py

import sys


#compute the 1d derivative of the function we will use different points depending on where we are in the integrand
def compute_derivative(Us, dt):
    Usd = np.zeros_like(Us)

    forward = -25./12.,4.,-3.,4./3.,-1./4.
    central = 1./12.,-2./3.,2./3.,-1./12. 
    n = Us.shape[0]
    for i in range(n):
        if i in [0,1]:
            coeff = forward
            Ups = Us[i:i+5]
        elif i in [n-1,n-2]:
            coeff = forward[::-1]
            Ups = -Us[i-4:i+1]
        else:
            coeff = central
            Ups = Us[i-2],Us[i-1],Us[i+1],Us[i+2]
        for c,U in zip(coeff,Ups):
            Usd[i, :, :] += c*U
    return Usd/dt
def compute_second_derivative(Us, dt):
    Usdd = np.zeros_like(Us)

    n = Us.shape[0]
    central = -1./12.,4./3.,-5./2.,4./3.,-1./12.
    forward = 15./4,-77./6.,107./6.,-13.,61./12.,-5./6.
    for i in range(n):
        if i in [0,1]:
            coeff = forward
            Ups = Us[i:i+6]
        elif i in [n-1,n-2]:
            coeff = forward[::-1]
            Ups = Us[i-5:i+1]
        else:
            coeff = central
            Ups = Us[i-2:i+3]
        for c,U in zip(coeff,Ups):
            Usdd[i, :, :] += c*U 
    return Usdd/dt**2
def compute_3rd_derivative0(Us,dt):
    forward = -49./8,29.,-461./8.,62.,-307./8.,13.,-15./8.
    Ups = Us[0:7]
    _,n,_ = Us.shape
    Uddd = np.zeros((n,n),dtype=Us.dtype)
    for c,U in zip(forward,Ups):
        Uddd += c*U
    return Uddd/dt**3
def eval_integral(Usd, kappa, imax, dt):
    ret = 0
    for i in range(imax+1):
        w = .5*dt if i in [0,imax] else dt
        ret += w*np.dot(kappa[i],Usd[imax-i])
    return ret
def compute_F(kappa,Hs,dt,every=None):
    Ls = -1.0j*commutator(Hs)
    F = np.zeros_like(kappa)
    print('computing F...')
    for i in range(kappa.shape[0]):
        if every is not None and i%every==0:
            print('t=',i*dt)
        F[i] = np.dot(Ls,kappa[i])+np.dot(kappa[i],Ls)
        if i>0:
            F[i] += eval_integral(kappa,kappa,i,dt)
    return F
def compute_B2(kappa,Hs,dt):
    Ls = -1.0j*commutator(Hs)
    B = compute_derivative(kappa,dt) 
    for i in range(kappa.shape[0]):
        B[i] += np.dot(Ls,kappa[i])
    return B

def compute_kernel(Us, Hs, dt,every=None):
    Ls = -1.0j*commutator(Hs)
    print('computing derivative...')
    Usd = compute_derivative(Us, dt)
    print('computing second derivative...')
    Usdd = compute_second_derivative(Us, dt)

    print('computing kernel...')
    kappa = np.zeros_like(Us)
    for i in range(Us.shape[0]):
        if every is not None and i%every==0:
            print('t=',i*dt)
        kappa[i] = Usdd[i] - np.dot(Ls,Usd[i]) - eval_integral(Usd, kappa, i, dt)
    return -kappa

