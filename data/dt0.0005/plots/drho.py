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
def plot(U,dt,ax,ls,c,l,start=1):
    rho = np.array([np.dot(Ui,rho0) for Ui in U])
    itv = int(dt/0.0005+1e-6)
    drho = np.array([np.linalg.norm(rhoi-rhoi_ref.flatten()) for rhoi,rhoi_ref in zip(rho,rho_ref[::itv])])
    x = np.arange(drho.shape[0])*dt
    ax.plot(x[start:],np.log10(drho[start:]),linestyle=ls,linewidth=lw,color=c,label=l)

fig1,ax1 = plt.subplots(nrows=1,ncols=1)
fig2,ax2 = plt.subplots(nrows=1,ncols=1)
fig3,ax3 = plt.subplots(nrows=1,ncols=1)
fig4,ax4 = plt.subplots(nrows=1,ncols=1)
colors = 'blue','orange','green','red','cyan','pink'
tMs = 1.2,2.4
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
for dt,color in zip([0.005,.01,0.02,.05,.1,.2],colors):
    print('dt=',dt)
    stop = int(10/dt+1e-6)+1
    lab = r'$\Delta t=$'+str(dt)
    for tM,ls,l in zip(tMs,('-',':'),(lab,None)):
        # TTM
        print('tM=',tM)
        itv = int(dt/0.0005+1e-6)
        M = int(tM/dt+1e-6)+1
        #U = Uex[::itv][:M]
        #T,_ = U2T(U)
        #print('len(T)=',len(T))
        #U = T2U(U,T,stop)
        #plot(U,dt,ax1,ls,color,l,start=M)

        # TTM(1) from kernel
        T = TTMT1(kappa,dt)
        #U = T2U(U[:1],T,stop,mitigate=True)
        U = T2U(U0,T,stop,mitigate=False,check=None)
        plot(U,dt,ax2,ls,color,l)

        # TTM(2) from kernel
        T[1] += dt**3/6*dddU0
        T[2:] += dt**3/2*F[::itv][1:M-1]
        #U = T2U(U[:1],T,stop,mitigate=True)
        U = T2U(U0,T,stop,mitigate=False,check=None)
        plot(U,dt,ax3,ls,color,l)

        # midpoint from kernel
        G1 = scipy.linalg.expm(-1j*dt*Ls)
        Gh = scipy.linalg.expm(-1j*dt*Ls/2)
        itv = int(dt/2/.0005+1e-6)
        K = kappa[::itv]
        T = np.zeros((M,4,4),dtype=complex)
        T[1] = G1-dt**2*np.dot(Gh,K[1])
        for N in range(2,M):
            T[N] = -dt**2*np.dot(Gh,K[N*2-1])
        #U = T2U_MPDI(U[:1],T,G1,stop,mitigate=True)
        U = T2U_MPDI(U0,T,G1,stop,mitigate=False,check=False)
        plot(U,dt,ax4,ls,color,l)

for ax,fig in zip((ax1,ax2,ax3,ax4),(fig1,fig2,fig3,fig4)):
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\log_{10}||\Delta\rho||$')
    ax.set_ylim(-7,-.5)
    ax.set_xlim(0,10)
    ax.legend()
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.98)
#fig1.savefig(f'rho_TTM.png', dpi=250)
fig2.savefig(f'rho_TTM_K.png', dpi=250)
fig3.savefig(f'rho_TTM_K2.png', dpi=250)
fig4.savefig(f'rho_MPDI.png', dpi=250)
