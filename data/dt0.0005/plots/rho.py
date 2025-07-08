import numpy as np
import scipy.linalg
import h5py
from GQME_discretization_error.ttm import * 
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000,suppress=True,linewidth=10000)
plt.rcParams.update({'font.size':16})
plt.rcParams.update({'figure.figsize':(6.4,4.8)})

dir_ = '../heom_run/'
f = h5py.File(dir_+f'long_path0.hdf5', 'r')
rho_ref = process_trajectory(f['rho'][:])
f.close()

kappa = np.load(dir_+'kappa_0.0005_4th.npy')
dddU0 = np.load(dir_+'dddU0_0.0005_4th.npy')
F = np.load(dir_+'F_0.0005_4th.npy')
M = len(kappa) 
rhos_ex = [None] * 4
for i in range(4):
    f = h5py.File(dir_+f'long_path{i}.hdf5', 'r')
    rhos_ex[i] = process_trajectory(f['rho'][:][:M])
    f.close()
Uex = rho2U(rhos_ex)

eps = 0.0
delta = -1.0
H = eps*sz + delta*sx
Ls = H2L(H)

rho0 = np.array([1,0,0,0])
lw = 3
def plot(U,dt,ax,c,l):
    rho = np.array([np.dot(Ui,rho0) for Ui in U])
    x = np.arange(rho.shape[0])*dt
    ax.plot(x,rho[:,0].real,linestyle='-',linewidth=lw,color=c,label=l)
    ax.plot(x,rho[:,1].real,linestyle=':',linewidth=lw,color=c)

def TTMT1(kappa,dt):
    T = -kappa[::itv][:M-1]
    T = np.concatenate([np.zeros((1,4,4)),T.copy()],axis=0)
    print('len(T)=',len(T))
    T[1] -= np.dot(Ls,Ls)
    T[1] /= 2
    T *= dt**2
    T[1] += np.eye(4)-1j*dt*Ls
    return T
U0 = np.eye(4,dtype=complex).reshape(1,4,4)
for dt in np.array([.01,.1,.2]):
    print('dt=',dt)
    fig,ax = plt.subplots(nrows=1,ncols=1)
    x = np.arange(len(rho_ref))*0.0005
    ax.plot(x,rho_ref[:,0,0].real,linestyle='-',linewidth=lw,color='k',label='HEOM')
    ax.plot(x,rho_ref[:,0,1].real,linestyle=':',linewidth=lw,color='k')

    # TTM
    #print('TTM')
    itv = int(dt/0.0005+1e-6)
    M = int(1.2/dt+1e-6)+1
    #U = Uex[::itv][:M]
    stop = int(10/dt+1e-6)+1
    #T,_ = U2T(U)
    #print('len(T)=',len(T))
    #U = T2U(U,T,stop)
    #plot(U,dt,ax,'blue','TTM')

    # FDIO 
    print('FDIO')
    T = -kappa[::itv][:M-1]*dt**2
    T = np.concatenate([np.zeros((1,4,4)),T.copy()],axis=0)
    print('len(T)=',len(T))
    T[1] += np.eye(4)-1j*dt*Ls
    #U = T2U(U0,T,stop,mitigate=True)
    U = T2U(U0,T,stop,mitigate=False,check=None)
    plot(U,dt,ax,'r','FDIO')

    # TTM from kernel
    print('TTM(1)')
    T = TTMT1(kappa,dt)
    #U = T2U(U0,T,stop,mitigate=True)
    U = T2U(U0,T,stop,mitigate=False,check=None)
    plot(U,dt,ax,'g','TTM(1)')

    # TTM from kernel
    print('TTM(2)')
    T[1] += dt**3/6*dddU0
    T[2:] += dt**3/2*F[::itv][1:M-1]
    #U = T2U(U0,T,stop,mitigate=True)
    U = T2U(U0,T,stop,mitigate=False,check=None)
    plot(U,dt,ax,'b','TTM(2)')


    # midpoint from kernel
    itv = int(dt/2/.0005+1e-6)
    K = kappa[::itv]
    G1 = scipy.linalg.expm(-1j*dt*Ls)
    Gh = scipy.linalg.expm(-1j*dt*Ls/2)
    T = np.zeros((M,4,4),dtype=complex)
    T[1] = G1-dt**2*np.dot(Gh,K[1])
    for N in range(2,M):
        T[N] = -dt**2*np.dot(Gh,K[N*2-1])
    #U = T2U_MPDI(U0,T,G1,stop,mitigate=True)
    U = T2U_MPDI(U0,T,G1,stop,mitigate=False,check=False)
    plot(U,dt,ax,'y','MPD/I')

    ax.set_xlabel(r'$\Omega t$')
    ax.set_ylabel(r'$\rho$')
    ax.set_ylim(-.05,1.05)
    ax.set_xlim(0,10)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    ax.legend()
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.98)
    fig.savefig(f'rho_{dt}.png', dpi=250)
    plt.close(fig)
