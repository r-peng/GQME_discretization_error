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

dt = 0.001
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

