from GQME_discretization_error.heom import *
from GQME_discretization_error.ttm import * 
from GQME_discretization_error.eval_memory_kernel_from_prop import * 

import numpy as np
import scipy as sp
import h5py

import sys

sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

eps = 0.0
delta = -1.0

def H(t):
    return eps*sz + delta*sx

fig1,ax1 = plt.subplots(nrows=1,ncols=1)
fig2,ax2 = plt.subplots(nrows=1,ncols=1)
fig3,ax3 = plt.subplots(nrows=1,ncols=1)
fig4,ax4 = plt.subplots(nrows=1,ncols=1)
colors = 'r','g','b','y','c','k'
for dt,c in zip([.01,.005,.001,.0005,.0001,.00005],colors):
    print('\ndt=',dt)
    f = h5py.File(f'propagator_{dt}.h5','r')
    Us = f['Us'][:]
    dUs = f['dUs'][:]
    d2Us = f['d2Us'][:]
    f.close()
    ls = []
    for Ui,fname in zip((Us,dUs,d2Us),('Us','dUs','d2Us')):
        data = Ui.copy()
        data[:,2,:] = Ui[:,1,:].copy()
        data[:,1,:] = Ui[:,2,:].copy()
        ls.append(data)
        np.save(fname+f'_{dt}.npy',data)
    continue
    Us,dUs,d2Us = ls
    kappa = compute_kernel(Us,  H(0), dt, Usd=dUs,Usdd=d2Us,every=1000)
    np.save(f'kappa_{dt}.npy',kappa)
    #kappa = np.load(f'kappa_{dt}.npy')
    F = compute_F(-kappa,H(0),dt,every=1000)
    np.save(f'F_{dt}.npy',F)

    dddU0 = compute_3rd_derivative0(Us,dt)
    np.save(f'dddU0_{dt}_4th.npy',dddU0)

    ax1.plot(np.arange(kappa.shape[0])*dt, -(kappa[:, 1, 0]).real, '-',color=c)
    ax2.plot(np.arange(kappa.shape[0])*dt, -(kappa[:, 1, 0]).imag, '-',color=c)
    ax3.plot(np.arange(kappa.shape[0])*dt, -(kappa[:, 1, 1]).real, '-',color=c)
    ax4.plot(np.arange(kappa.shape[0])*dt, -(kappa[:, 1, 2]).real, '-',color=c)
for ax in [ax1,ax2,ax3,ax4]:
    ax.set_xlabel('Time')
    ax.set_xlim((0,.5))
    #ax.legend()
for fig in [fig1,fig2,fig3,fig4]:
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.99, top=0.98)
fig1.savefig(f"kappa_1,0_real.png", dpi=250)
fig2.savefig(f"kappa_1,0_imag.png", dpi=250)
fig3.savefig(f"kappa_1,1_real.png", dpi=250)
fig4.savefig(f"kappa_1,2_real.png", dpi=250)

