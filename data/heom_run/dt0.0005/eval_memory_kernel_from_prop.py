from GQME_discretization_error.heom import *
from GQME_discretization_error.ttm import * 

import numpy as np
import scipy as sp
import h5py

import sys


#compute the 1d derivative of the function we will use different points depending on where we are in the integrand
def compute_derivative(Us, dt):
    Usd = np.zeros(Us.shape, dtype=np.complex128)

    forward = -25./12.,4,-3,4./3,-1./4
    central = 1./12,-2./3,2./3,-1./12 
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
            Usd[i, :, :] += c*U/dt 
    return Usd
def compute_second_derivative(Us, dt):
    Usdd = np.zeros(Us.shape, dtype=np.complex128)

    n = Us.shape[0]
    central = -1./12,4./3,-5./2,4./3,-1./12
    forward = 15./4,-77./6,107./6,-13,61./12,-5./6
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
            Usdd[i, :, :] += c*U/dt**2 
    return Usdd
def compute_3rd_derivative0(Us,dt):
    forward = -49./8,29,-461./8,62,-307./8,13,-15./8
    Ups = Us[0:7]
    _,n,_ = Us.shape
    Uddd = np.zeros((n,n),dtype=complex)
    for c,U in zip(forward,Ups):
        Uddd += c*U/dt**3
    return Uddd
def eval_integral(Usd, kappa, imax, dt):
    ret = 0.0
    for i in range(imax+1):
        w = .5*dt if i in [0,imax] else dt
        ret += w*np.dot(kappa[i],Usd[imax-i])
    return ret
def compute_F(kappa,Hs,dt,every=None):
    Ls = -1.0j*commutator(Hs)
    F = np.zeros(kappa.shape, dtype=np.complex128)
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
    kappa = np.zeros(Us.shape, dtype=np.complex128)
    for i in range(Us.shape[0]):
        if every is not None and i%every==0:
            print('t=',i*dt)
        kappa[i] = Usdd[i] - np.dot(Ls,Usd[i]) - eval_integral(Usd, kappa, i, dt)
    return -kappa

def main(dt):
    sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

    eps = 0.0
    delta = -1.0

    def H(t):
        return eps*sz + delta*sx

    dt_run = 0.0005
    itv = int(dt/dt_run+1e-6)

    M = int(3./dt+1e-6)
    rhos = [None] * 4
    for i in range(4):
        f = h5py.File(f'long_path{i}.hdf5', 'r')
        rhos[i] = process_trajectory(f['rho'][:][::itv][:M])
        f.close()
    Us = rho2U(rhos)
    
    kappa = compute_kernel(Us,  H(0), dt, every=1000)
    np.save(f'kappa_{dt}_4th.npy',kappa)
    #kappa = np.load(f'kappa_{dt}_4th.npy')
    F = compute_F(-kappa,H(0),dt,every=1000)
    np.save(f'F_{dt}_4th.npy',F)
    #B = compute_B2(-kappa,H(0),dt)
    #np.save(f'B2_{dt}_4th.npy',B)
    dddU0 = compute_3rd_derivative0(Us,dt)
    np.save(f'dddU0_{dt}_4th.npy',dddU0)

    titles = ['00', '01', '10', '11']
    ijs = [[1, 0], [1, 1], [1, 2]]
    colors = 'r','g','b'
    fig,ax = plt.subplots(nrows=1,ncols=1)
    for ij,color in zip(ijs,colors):
        i = ij[0]
        j = ij[1]
        ax.plot(np.arange(kappa.shape[0])*dt, np.real(kappa[:, i, j]), '-',color=color,label=f'{i},{j}')
        ax.plot(np.arange(kappa.shape[0])*dt, np.imag(kappa[:, i, j]), '--',color=color)
    ax.set_xlabel('Time')
    ax.legend()
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.99, top=0.98)
    fig.savefig(f"kappa_{dt}.png", dpi=250)
    plt.close(fig)

if __name__ == "__main__":
    main(0.005)

