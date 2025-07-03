import numpy as np
import scipy.linalg
import h5py
from GQME_discretization_error.ttm import * 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000,suppress=True,linewidth=10000)
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})

dt_min = .0005
itv = int(dt_min/.0001+1e-6)
M = int(3/dt_min+1e-6) 
rhos_ex = [None] * 4
dir_ = '../heom_run/'
for i in range(4):
    f = h5py.File(dir_+f'long_path{i}.hdf5', 'r')
    rhos_ex[i] = process_trajectory(f['rho'][:][::itv][:M])
    f.close()
Uex = rho2U(rhos_ex)

colors = 'blue','orange','green','red','cyan'
fnames = (1,0,'real'),(1,0,'imag'),(1,1,'real'),(1,2,'real'),(None,None,'err')
ylabs = 'Re'+r'$\mathcal{K}_{01,00}$',\
        'Im'+r'$\mathcal{K}_{01,00}$',\
        'Re'+r'$\mathcal{K}_{01,01}$',\
        'Re'+r'$\mathcal{K}_{01,10}$',\
        r'$\log_{10}||\Delta\mathcal{K}||$',
figs = [plt.subplots(nrows=1,ncols=1) for _ in range(5)]

def plot(x,K,ls,c,l=None,mk=None,mksz=None,itv=1,kappa=None):
    for (ix1,ix2,typ),(fig,ax) in zip(fnames,figs):
        mksz = 10 
        if typ=='real':
            y = K[:,ix1,ix2].real 
        elif typ=='imag':
            y = K[:,ix1,ix2].imag
        else:
            if kappa is None:
                continue
            dK = K+kappa[:K.shape[0]]
            y = np.array([np.linalg.norm(dKi) for dKi in dK])
            y = np.log10(y)
            mksz = 5
        ax.plot(x[::itv],y[::itv],linestyle=ls,color=c,marker=mk,markersize=mksz,label=l)

# check kernel
kappa = np.load(dir_+f'kappa_{dt_min}_4th.npy')
dddU0 = np.load(dir_+f'dddU0_{dt_min}_4th.npy')
F = np.load(dir_+f'F_{dt_min}_4th.npy')
#F = np.load(dir_+'B2_0.0005_4th.npy')
x = np.arange(M)*dt_min
print(x[-1])
plot(x,-kappa[:M],'-','k',l='exact')

eps = 0.0
delta = -1.0
H = eps*sz + delta*sx
Ls = H2L(H)

rho0 = np.array([1,0,0,0])
def TTMK1(T,dt):
    K = T[1:].copy()
    K[0] -= np.eye(4)-1j*dt*Ls
    K /= dt**2
    K[0] *= 2
    K[0] += np.dot(Ls,Ls)
    return K
for dt,color in zip((0.01,0.05,0.1),colors):
    itv = int(dt/dt_min+1e-6)
    print('dt=',dt,itv)
    U = Uex[::itv]
    T,_ = U2T(U)

    # TTM w/ F correction
    K = TTMK1(T,dt)
    K[0] -= dt*dddU0/3
    K[1:] -= dt*F[::itv][1:K.shape[0]]/2
    itv_ = 3 if dt<0.02 else 1
    x = np.arange(K.shape[0])*dt
    l = r'$\Omega\Delta t=$'+str(dt)
    plot(x,K,'-',color,l,mk='o',itv=itv_,kappa=kappa[::itv])

    # TTM
    K = TTMK1(T,dt)
    itv_ = 3 if dt<0.02 else 1
    x = np.arange(K.shape[0])*dt
    plot(x,K,'--',color,mk='v',itv=itv_,kappa=kappa[::itv])

    # midpoint
    G1 = scipy.linalg.expm(-1j*dt*Ls)
    Gh_inv = scipy.linalg.expm(1j*dt*Ls/2)

    T = U.copy()
    stop = U.shape[0]
    T[0] = 0
    T[1] = 2*U[1]-G1
    for N in range(2,stop):
        for m in range(1,N):
            T[N] -= np.dot(T[m],U[N-m])
        T[N] *= 2

    K = -T.copy()
    K[1] += G1
    for i in range(1,stop):
        K[i] = np.dot(Gh_inv,K[i])/dt**2
    K = np.array([(K[i]+K[i+1])/2 for i in range(1,stop-1)])
    x = np.arange(1,K.shape[0]+1)*dt
    plot(x,-K,':',color,mk='+',itv=itv_,kappa=kappa[::itv])

for (ix1,ix2,typ),(fig,ax),ylab in zip(fnames,figs,ylabs):
    ax.set_xlabel(r'$\Omega t$')
    ax.set_ylabel(ylab)
    #ax.set_xlim(0,0.5)
    ax.legend()
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.98)
    if typ=='err':
        fig.savefig(f'K_{typ}_{dt_min}.png', dpi=250)
    else:
        fig.savefig(f'K{ix1}{ix2}_{typ}_{dt_min}.png', dpi=250)
    plt.close(fig)
