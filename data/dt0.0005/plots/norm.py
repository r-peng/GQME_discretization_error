
import numpy as np
import scipy.linalg
import h5py
from GQME_discretization_error.ttm import * 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000,suppress=True,linewidth=10000)
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})
M = int(3/.0005+1e-6) 
dir_ = '../heom_run/'
kappa = np.load(dir_+'kappa_0.0005_4th.npy')
F = np.load(dir_+'F_0.0005_4th.npy')
fig,ax = plt.subplots(nrows=1,ncols=1) 
for y,lab in zip((kappa,F),('K','F')):
    y = np.array([np.linalg.norm(yi) for yi in y])
    x = np.arange(len(y))*.0005
    ax.plot(x,np.log10(y),linestyle='-',label=lab)
ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel('log10(norm)')
#ax.set_xlim(0,0.5)
ax.legend()
fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.98)
fig.savefig(f'norm.png', dpi=250)
plt.close(fig)

fig,ax = plt.subplots(nrows=1,ncols=1) 
rhos_ex = [None] * 4
for i in range(4):
    f = h5py.File(dir_+f'long_path{i}.hdf5', 'r')
    rhos_ex[i] = process_trajectory(f['rho'][:][:M])
    f.close()
Uex = rho2U(rhos_ex)
colors = 'blue','orange','green','red','cyan'
eps = 0.0
delta = -1.0
H = eps*sz + delta*sx
Ls = H2L(H)
K0 = (-np.dot(Ls,Ls)-kappa[0])/2
dddU0 = np.load(dir_+'dddU0_0.0005_4th.npy')
for dt,color in zip((0.01,0.02,0.05,0.1),colors):
    itv = int(dt/0.0005+1e-6)
    print('dt=',dt,itv)
    U = Uex[::itv]
    K = -kappa[::itv]
    F_ = F[::itv]
    T1 = np.eye(4)-1j*dt*Ls+dt**2*K0

    U1 = np.zeros_like(U)
    U1[0] = np.eye(4)
    for N in range(U1.shape[0]-1):
        U1[N+1] = np.dot(T1,U[N]) 
        for m in range(N):
            U1[N+1] += dt**2*np.dot(K[N-m],U[m])
    y = np.array([np.linalg.norm(Di) for Di in U1-U])
    x = np.arange(len(y))*dt
    ax.plot(x[1:],np.log10(y[1:]),color=color,linestyle='-',label=dt)

    U2 = U1.copy()
    for N in range(U1.shape[0]-1):
        U2[N+1] += dt**2*np.dot(dt*dddU0/6,U[N]) 
        for m in range(N):
            U2[N+1] += dt**3*np.dot(F_[N-m],U[m])/2
    y = np.array([np.linalg.norm(Di) for Di in U2-U])
    ax.plot(x[1:],np.log10(y[1:]),color=color,linestyle='--')
ax.set_xlabel(r'$\Omega t$')
ax.set_ylabel('log10(dU)')
#ax.set_xlim(0,0.5)
ax.legend()
fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.98)
fig.savefig(f'scaling.png', dpi=250)
plt.close(fig)
