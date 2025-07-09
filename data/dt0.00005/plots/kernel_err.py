import numpy as np
import scipy.linalg
import h5py
from GQME_discretization_error.ttm import * 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000,suppress=True,linewidth=10000)
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})

fnames = (1,0,'real'),(1,0,'imag'),(1,1,'real'),(1,2,'real'),(None,None,'err')
ylabs = 'Re'+r'$\mathcal{K}_{01,00}$',\
        'Im'+r'$\mathcal{K}_{01,00}$',\
        'Re'+r'$\mathcal{K}_{01,01}$',\
        'Re'+r'$\mathcal{K}_{01,10}$',\
        r'$\log_{10}||\Delta\mathcal{K}||$',
#figs = [plt.subplots(nrows=1,ncols=1) for _ in range(5)]
fig,ax = plt.subplots(nrows=1,ncols=1) 
fig1,ax1 = plt.subplots(nrows=1,ncols=1)
fig2,ax2 = plt.subplots(nrows=1,ncols=1)
fig3,ax3 = plt.subplots(nrows=1,ncols=1)
fig4,ax4 = plt.subplots(nrows=1,ncols=1)
eps = 0.0
delta = -1.0
H = eps*sz + delta*sx
Ls = H2L(H)
def TTMK1(T,dt):
    K = T[1:].copy()
    K[0] -= np.eye(4)-1j*dt*Ls
    K /= dt**2
    K[0] *= 2
    K[0] += np.dot(Ls,Ls)
    return K
dir_ = '../../dt0.0005/heom_run/'
dddU0 = np.load(dir_+f'dddU0_0.0005_4th.npy')
M = int(1./0.0005+1e-6) 
rhos_ex = [None] * 4
for i in range(4):
    f = h5py.File(dir_+f'long_path{i}.hdf5', 'r')
    rhos_ex[i] = process_trajectory(f['rho'][:][:M])
    f.close()
Uex = rho2U(rhos_ex)
for dt_min,ls in zip([.005,.0005,.00005],(':','--','-')):
    ax.plot([],[],linestyle=ls,color='k',label=r'$\Delta t_{\rm ref}=$'+str(dt_min))
    dir_ = '../heom_run/' 
    kappa = np.load(dir_+f'kappa_{dt_min}.npy')
    F = np.load(dir_+f'F_{dt_min}.npy')

    ax1.plot(np.arange(kappa.shape[0])*dt_min, -(kappa[:, 1, 0]).real, '-',color='k')
    ax2.plot(np.arange(kappa.shape[0])*dt_min, -(kappa[:, 1, 0]).imag, '-',color='k')
    ax3.plot(np.arange(kappa.shape[0])*dt_min, -(kappa[:, 1, 1]).real, '-',color='k')
    ax4.plot(np.arange(kappa.shape[0])*dt_min, -(kappa[:, 1, 2]).real, '-',color='k')
    for dt,c in zip((0.01,0.05,0.1),('blue','orange','green')):
        print('dt,dt_min=',dt,dt_min)
        itv = int(dt/dt_min+1e-6)
        kappa_ = -kappa[::itv]
        F_ = F[::itv]

        itv = int(dt/.0005+1e-6)
        U = Uex[::itv]
        T,_ = U2T(U)
        K = TTMK1(T,dt)[:kappa_.shape[0]]
        K[0] -= dt*dddU0/3
        K[1:] -= dt*F_[1:K.shape[0]]/2

        itv_ = 3 if dt<0.02 else 1
        y = np.array([np.linalg.norm(dKi) for dKi in K-kappa_])
        y = np.log10(y)
        x = np.arange(len(y))*dt
        ax.plot(x,y,linestyle=ls,color=c)

        ax1.plot(x, (K[:, 1, 0]).real, linestyle=ls,color=c)
        ax2.plot(x, (K[:, 1, 0]).imag, linestyle=ls,color=c)
        ax3.plot(x, (K[:, 1, 1]).real, linestyle=ls,color=c)
        ax4.plot(x, (K[:, 1, 2]).real, linestyle=ls,color=c)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\log_{10}||\Delta\mathcal{K}||$')
ax.set_xlim(0,.5)
ax.legend()
fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.98)
fig.savefig(f'K_err_5e-5.png', dpi=250)
plt.close(fig)

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
