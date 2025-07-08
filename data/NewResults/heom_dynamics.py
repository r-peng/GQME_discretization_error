from GQME_discretization_error.heom import *
from GQME_discretization_error.heom.mps import overlap

import numpy as np
import scipy as sp
import h5py
import copy
import multiprocessing

import sys
dt = 0.0001


#current function used to define the support points for the aaa algorithm. The best choice of support points will depend on the nature of the spectral function you are considering (e.g. where are the interesting features of the spectral function, discontinuities, derivative discontinuities, sharp peaks)
#This choice works rather well for exponential cutoffs with challenging points at zero but won't work well for generic spectral densities.

def heom_dynamics(rho0, H, baths, tmax, chimax, tol = 1e-6, plot_aaa = False, fname=None, output_stride =1000,iprint=0):
    nhilb = rho0.shape[0]
    nliouv = nhilb*nhilb

    bs = [x.discretise(output_fitting = plot_aaa) for x in baths]

    #if plot_aaa:
    #    import matplotlib.pyplot as plt
    #    for b in bs:
    #        plt.plot(b["wf"], b["Sw"])
    #        plt.plot(b["wf"], b["Sw_fit"])
    #    plt.show()

    nt = int(tmax/dt)+1
    print(nt)
    #now build the propagator matrices for HEOM

    #if RANK==iprint:
    #    print("Build HEOM propagator matrices", file=sys.stderr)
    print("Build HEOM propagator matrices", file=sys.stderr)
    prop = HEOM_propagator(bs, dt)
     
    A = setup_HEOM_ados(rho0, prop)
    L = HEOM_bath_mpo(H, bs[0])
    print(L)

    ms = []
    dms = []
    d2ms = []
    for i in range(nhilb):
        for j in range(nhilb):
            mat = np.zeros((nhilb, nhilb))
            mat[i,j] = 1.0
            opmpo = setup_HEOM_ados(mat, prop)
            dopmpo = setup_HEOM_ados(mat, prop)
            d2opmpo = setup_HEOM_ados(mat, prop)

            dopmpo.apply_MPO(L,method="naive")

            d2opmpo.apply_MPO(L,method="naive")
            d2opmpo.apply_MPO(L,method="naive")

            ms.append(opmpo)
            dms.append(dopmpo)
            d2ms.append(d2opmpo)

    print(len(A))

    #if RANK==iprint:
    #    print("Build HEOM propagator MPO", file=sys.stderr)
    print("Build HEOM propagator MPO", file=sys.stderr)

    rhos = np.zeros((nt+1, nliouv), dtype = np.complex128)
    drhos = np.zeros((nt+1, nliouv), dtype = np.complex128)
    d2rhos = np.zeros((nt+1, nliouv), dtype = np.complex128)

    rhos[0, :] = rho0.flatten()
    for mi in range(nhilb):
        for ni in range(nhilb):
            rhos[0, ni*nhilb+mi] = overlap(A, ms[mi*nhilb+ni])
            drhos[0, ni*nhilb+mi] = -1.0j*overlap(A, dms[mi*nhilb+ni])
            d2rhos[0, ni*nhilb+mi] = -overlap(A, d2ms[mi*nhilb+ni])
    #if RANK==iprint:
    #    print(0, end=' ' )
    print(0, end=' ' )
    for j in range(rhos.shape[1]):
        #if RANK==iprint:
        #    print(np.real(rhos[0, j]), np.imag(rhos[0, j]), end=' ')
        print(np.real(rhos[0, j]), np.imag(rhos[0, j]), end=' ')
    #if RANK==iprint:
    #    print(A.maximum_bond_dimension())
    print(A.maximum_bond_dimension())

    n = 1
    for i in range(nt):
        #update the time dependent system propagator at the midpoint of the next step
        Hsys = H
        Lsys = commutator(Hsys)
        Us = sp.linalg.expm(-1.0j*dt*Lsys)    

        #apply the propagator through a time step
        apply_propagator(Us, prop, A, tol=tol, nbond=chimax)

        for mi in range(nhilb):
            for ni in range(nhilb):
                rhos[i+1, ni*nhilb+mi] = overlap(ms[mi*nhilb+ni], A)
                drhos[i+1, ni*nhilb+mi] = -1.0j*overlap(dms[mi*nhilb+ni],A)
                d2rhos[i+1, ni*nhilb+mi] = -overlap(d2ms[mi*nhilb+ni],A)

        if (i % output_stride == 0 and not fname is None):
            print((i+1)*dt, end=' ' )
            for j in range(nliouv):
                print(np.real(rhos[i+1, j]), end=' ')
            for j in range(nliouv):
                print(drhos[i+1, j], end=' ')
            for j in range(nliouv):
                print(d2rhos[i+1, j], end=' ')

            print(A.maximum_bond_dimension())
            sys.stdout.flush()

            h5out = h5py.File(fname, 'w')
            h5out.create_dataset('t', data=np.arange(nt+1)*dt)
            h5out.create_dataset('rho', data=rhos)
            h5out.create_dataset('drho', data=drhos)
            h5out.create_dataset('d2rho', data=d2rhos)
            h5out.close()

    if not fname is None:
        h5out = h5py.File(fname, 'w')
        h5out.create_dataset('t', data=np.arange(nt+1)*dt)
        h5out.create_dataset('rho', data=rhos)
        h5out.create_dataset('drho', data=drhos)
        h5out.create_dataset('d2rho', data=d2rhos)

        h5out.close()
    return rhos, drhos, d2rhos, np.arange(nt+1)*dt

#def main(iprint=0):
def main(RANK,iprint=0):
    sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)

    #and the system part of the system bath coupling operator
    S = sz

    #function defining the bath spectral function
    beta = 5 
    s = 1
    alpha = 0.3
    wc = 5
    def Jw(w):
        return np.sign(w)*np.pi/2.0*alpha*wc*np.power(np.abs(w)/wc, s)*np.exp(-np.abs(w)/wc)

    def Sw(w):
        if beta == None:
            return Jw(w)*np.where(w > 0, 1.0, 0.0)
        else:
            return Jw(w)*0.5*(1+1.0/np.tanh(beta*w/2.0))


    #the initial value of the system density operator
    rho0s = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])] 
    rho0 = rho0s[RANK]
    fname = f'long_path{RANK}.hdf5'

    #set up the evolution parameters
    tmax = 0.5+dt 

    #setup the ados parameters - accuracy of representation of auxiliary density operator
    nbose = 20
    Lmin = 8
    tol = 1e-8
    chimax = 120

    #setup the aaa parameters - controls accuracy of representation of bath correlation function 
    #convergence with respect to the aaa_tol parameter depends significantly on the choice of support points used, and also on the form of the bath correlation function.  
    #Here I am using a softmspace grid with a rather tight zero.  This approach seems to work quite well for baths with an exponential cutoff at zero temperature but doesn't necessarily work the best for e.g. Debye baths.
    #The choice of support points is somewhat an art and the accuracy can really only be determined by monitoring both the accuracy of the aaa fitting of the spectral function and also the correlation function.  
    #To change the 
    aaa_tol = 1e-3
    wmax = wc

    #construct the HEOM baths object
    baths = [heom_bath(S, Sw, nbose, Lmin=Lmin, wmax = wc, aaa_tol=aaa_tol)]

    eps = 0.0
    delta = -1.0

    H = eps*sz + delta*sx
    
    return heom_dynamics(rho0, H, baths, tmax, chimax, tol=tol, plot_aaa=True, fname=fname,iprint=iprint)


def eval_integral(Usd, kappa, imax):
    ret = 0.0
    for i in range(imax):
        ret += (Usd[imax-(i+1)]@kappa[i+1]+Usd[imax-i]@kappa[i])*dt/2.0
    return ret

def compute_kernel(Us, Usd, Usdd, Hs):
    Ls = -1.0j*commutator(Hs)

    kappa = np.zeros(Us.shape, dtype=np.complex128)
    for i in range(Us.shape[0]):
        kappa[i] = Usdd[i] - Usd[i]@Ls - eval_integral(Usd, kappa, i)
    return -kappa

if __name__ == "__main__":
    p = multiprocessing.Pool(1)
    r = p.map(main, range(4))
    #print(r[0][1])

    Us = np.zeros((len(r[0][3]), 4, 4), dtype=np.complex128)
    dUs = np.zeros((len(r[0][3]), 4, 4), dtype=np.complex128)
    d2Us = np.zeros((len(r[0][3]), 4, 4), dtype=np.complex128)

    for i in range(4):
        for j in range(4):
            Us[:, j, i] = r[i][0][:, j]
    for i in range(4):
        for j in range(4):
            dUs[:, j, i] = r[i][1][:, j]
    for i in range(4):
        for j in range(4):
            d2Us[:, j, i] = r[i][2][:, j]

    fname = 'propagator_'+str(dt)+'.h5'
    if not fname == None:
        h5out = h5py.File(fname, 'w')
        h5out.create_dataset('Us', data=Us)
        h5out.create_dataset('dUs', data=dUs)
        h5out.create_dataset('d2Us', data=d2Us)

        h5out.close()

    eps = 0.0
    delta = -1.0
    sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)
    Hs = eps*sz + delta*sx
    kappa = compute_kernel(Us, dUs, d2Us, Hs)

    titles = ['00', '01', '10', '11']
    for i in range(4):
        for j in range(4):
            plt.figure(i*4+j)
            plt.title(titles[i] + titles[j])
            plt.plot(np.arange(kappa.shape[0])*dt, np.real(kappa[:, i, j]), 'k-')
            plt.plot(np.arange(kappa.shape[0])*dt, np.imag(kappa[:, i, j]), 'r-')
            #plt.plot(Usddp[:, j], 'r--')
            #plt.plot(np.imag(Usdd[:, j]), 'o--')


    plt.show()