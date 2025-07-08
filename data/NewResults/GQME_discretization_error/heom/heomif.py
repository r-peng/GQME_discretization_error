from .aaa import AAA_algorithm
from .mps import mps
from .mpo import mpo

import numpy as np
import scipy as sp


def commutator(L):
    return np.kron(L, np.identity(L.shape[0])) - np.kron(np.identity(L.shape[0]), L.T)

def anti_commutator(L):
    return np.kron(L, np.identity(L.shape[0])) + np.kron(np.identity(L.shape[0]), L.T)

def Mkp(nbose):
    b1 = np.zeros((nbose, nbose), dtype=np.complex128)

    for i in range(nbose-1):
        b1[i, i+1] = np.sqrt((i+1.0))

    return b1

def Mkm(nbose):
    b1 = np.zeros((nbose, nbose), dtype=np.complex128)
    for i in range(nbose-1):
        b1[i+1, i] = np.sqrt((i+1.0))

    return b1

def Lkp(nbose, dk, S, s=0.5):
    b1 = Mkp(nbose)*np.sqrt(dk)

    Scomm = commutator(S)
    return np.kron(Scomm, b1)


def Lkm(nbose, dk, S, mind = True, s=0.5):    
    coeff = 1
    if not mind:
        coeff = -1.0

    b1 = coeff *np.sqrt(dk)*Mkm(nbose)

    Sop = None
    if mind:
        Sop = np.kron(S, np.identity(S.shape[0]))
    else:
        Sop = np.kron(np.identity(S.shape[0]), S.T)
    return np.kron(Sop, b1)

def nop(D2, nbose, zk):
    return -1.0j*np.kron(np.identity(D2), np.diag((np.arange(nbose))*zk))


def mode_operator(nbose, dk, zk,  S, mind):
    op = None
    s = 0.5
    if mind:
        op = Lkp(nbose, dk, S, s=s) + Lkm(nbose, dk, S, mind=mind, s= s) + nop(S.shape[0]**2, nbose, zk)
    else:
        op = Lkp(nbose, np.conj(dk), S, s=s) + Lkm(nbose, np.conj(dk), S, mind=mind, s=s) + nop(S.shape[0]**2, nbose, np.conj(zk))
    return op

    
def Mk(nbose, dk, zk, S, mind, dt, nf=None):
    op = None
    if(nf > nbose):
        op = mode_operator(nf, dk, zk, S, mind)
        s = S.shape[0]*S.shape[1]
        expm = sp.linalg.expm(-1.0j*dt*op).reshape(s, nf, s, nf)
        return expm[:, :nbose, :, :nbose].reshape(s*nbose, s*nbose)
    else:
        op = mode_operator(nbose, dk, zk, S, mind)
        return sp.linalg.expm(-1.0j*dt*op)



def compute_dimensions(S, dk, zk, L, Lmin = None):
    ds = np.ones(2*len(dk)+1, dtype = int)
    ds[0] = S.shape[0]*S.shape[1]

    minzk = np.amin(np.real(zk))
    if(Lmin is None):
        for i in range(len(dk)):
            nb = L
            ds[2*i+1] = nb
            ds[2*i+2] = nb
    else:
        for i in range(len(dk)):
            nb = max(int(L*minzk/np.real(zk[i])), Lmin)
            ds[2*i+1] = nb
            ds[2*i+2] = nb

    return ds

#build the short time HEOM propagator 
def build_propagator_matrices(S, dk, zk, dt, L, Lmin=None, sf = 1):
    ds = compute_dimensions(S, dk, zk, L, Lmin=Lmin)

    Uks = []
    for i in range(len(dk)):
        nb = ds[2*i+1]
        Uks.append(Mk(nb, dk[i], zk[i], S, True, dt/2.0, nf = sf*nb))
        Uks.append(Mk(nb, dk[i], zk[i], S, False, dt/2.0, nf = sf*nb))
    return Uks


def HEOM_bath_propagator(bath, dt):
    Lmin = None
    sf = 1
    if 'Lmin' in bath.keys():
        Lmin = bath['Lmin']
    if 'sf' in bath.keys():
        sf = bath['sf']
    return build_propagator_matrices(bath['S'], bath['d'], bath['z'], dt, bath['L'], Lmin=Lmin, sf=sf)

def HEOM_propagator(baths, dt):
    #if we have a list of baths
    if isinstance(baths, list):
        Uks = []
        for bath in baths:
            Uks = Uks + HEOM_bath_propagator(bath, dt)
        return Uks

    elif isinstance(baths, dict):
        return HEOM_bath_propagator(baths, dt)


import copy
def mode_operator_proj(nbose, dk, zk,  S, mind):
    op = mode_operator(nbose, dk, zk, S, mind)
    
    s = S.shape[0]*S.shape[1]
    op = op.reshape(s, nbose, s, nbose)
    opres = copy.deepcopy(op)

    opres[:, 0, :, :] -= op[:, 0, :, :]

    return opres.reshape((s*nbose, s*nbose))

def Mk_proj(nbose, dk, zk, S, mind, dt, nf=None):
    op = None
    if(nf > nbose):
        op = mode_operator_proj(nf, dk, zk, S, mind)
        s = S.shape[0]*S.shape[1]
        expm = sp.linalg.expm(-1.0j*dt*op).reshape(s, nf, s, nf)
        return expm[:, :nbose, :, :nbose].reshape(s*nbose, s*nbose)
    else:
        op = mode_operator_proj(nbose, dk, zk, S, mind)
        return sp.linalg.expm(-1.0j*dt*op)


def build_projected_propagator_matrices(S, dk, zk, dt, L, Lmin=None, sf = 1):
    ds = compute_dimensions(S, dk, zk, L, Lmin=Lmin)

    Uks = []
    for i in range(len(dk)):
        nb = ds[2*i+1]
        Uks.append(Mk_proj(nb, dk[i], zk[i], S, True, dt/2.0, nf = sf*nb))
        Uks.append(Mk_proj(nb, dk[i], zk[i], S, False, dt/2.0, nf = sf*nb))
    return Uks

def HEOM_bath_projected_propagator(bath, dt):
    Lmin = None
    sf = 1
    if 'Lmin' in bath.keys():
        Lmin = bath['Lmin']
    if 'sf' in bath.keys():
        sf = bath['sf']
    return build_projected_propagator_matrices(bath['S'], bath['d'], bath['z'], dt, bath['L'], Lmin=Lmin, sf=sf)

def HEOM_projected_propagator(baths, dt):
    #if we have a list of baths
    if isinstance(baths, list):
        Uks = []
        for bath in baths:
            Uks = Uks + HEOM_bath_projected_propagator(bath, dt)
        return Uks

    elif isinstance(baths, dict):
        return HEOM_bath_projected_propagator(baths, dt)

def apply_propagator(Us, Uks, A, method='naive', tol=None, nbond = None):
    #apply non local two site gates.  Here we perform swap operations as we go, unless we are applying the last gate
    for i, Uk in enumerate(Uks):
        if(i+1 == len(Uks)):
            A.apply_two_site(Uk, i, i+1, method=method, tol=tol, nbond=nbond)
        else:
            A.apply_bond_tensor_and_swap(Uk, i, dir = 'right', tol=tol, nbond=nbond)

    A.apply_one_site(Us, -2)

    for i, Uk in reversed(list(enumerate(Uks))):
        if(i+1 == len(Uks)):
            A.apply_two_site(Uk, i, i+1, method=method, tol=tol, nbond=nbond)
        else:
            A.apply_bond_tensor_and_swap(Uk, i, dir = 'left', tol=tol, nbond=nbond)

    return A



def setup_HEOM_ados(rho0, prop):
    nliouv = rho0.flatten().shape[0]
    Nmodes = len(prop)+1
    d = np.zeros(Nmodes, dtype = int)
    for i, Uk in enumerate(prop):
        d[i+1] = Uk.shape[0]//nliouv
    d[0] = nliouv

    #setup the ado mps
    A = mps(chi = np.ones(len(prop), dtype=int), d=d, init='zeros', dtype=np.complex128)

    A[0][0, :, 0] = rho0.flatten()
    for i in range(1, len(A)):
        A[i][0, 0, 0] = 1.0
    A.orthogonalise()
    return A


def extract_rho(A):
    T = None
    for i in reversed(range(1, len(A))):
        if not isinstance(T, np.ndarray):
            T = A[i][:, 0, 0]
        else:
            M = A[i][:, 0, :]
            T = M@T

    Mi = A[0][:, :, :]
    T = np.tensordot(Mi, T, axes=([2], [0]))
    return T[0, :]


def Cr(func, t, tol=1e-8, limit=1000, workers=1, dx = 1e-12):
    return sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), dx, np.inf, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), -np.inf, -dx, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.real(func(w)*np.exp(-1.0j*t*w)), -dx, dx, limit=1000, points=[0])[0]/np.pi 

def Ci(func, t, tol=1e-8, limit=1000, workers=1, dx = 1e-12):
    return sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), dx, np.inf, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), -np.inf, -dx, limit=1000)[0]/np.pi + sp.integrate.quad_vec(lambda w : np.imag(func(w)*np.exp(-1.0j*t*w)), -dx, dx, limit=1000, points=[0])[0]/np.pi 


def HEOM_bath_mpo(Hs, bath, transpose=True):
    Lmin = None
    if 'Lmin' in bath.keys():
        Lmin = bath['Lmin']
    return heom_mpo(Hs, bath['S'], bath['d'], bath['z'], bath['L'], Lmin=Lmin, transpose=transpose)

def heom_mpo(Hs, S, dk, zk, L, Lmin=1, transpose=True):
    ds = compute_dimensions(S, dk, zk, L, Lmin=Lmin)
    chi = np.ones(2*len(dk), dtype=int)*5

    H = mpo(chi=chi, d = ds, dtype=np.complex128, init='zeros')

    H[0][0, :, :, 0] = commutator(Hs)
    H[0][0, :, :, 1] = commutator(S)
    H[0][0, :, :, 2] = np.kron(S, np.identity(S.shape[0]))
    H[0][0, :, :, 3] = np.kron(np.identity(S.shape[0]), S.T)
    H[0][0, :, :, 4] = np.identity(S.shape[0]**2)

    for i in range(len(dk)):
        nb = ds[2*i+1]
        Id = np.identity(nb)

        if(i + 1 != len(dk)):
            #insert all the identity terms that carry operators through so they act on the next site
            for j in range(5):
                H[2*i+1][j, :, :, j] = Id

            for j in range(5):
                H[2*i+2][j, :, :, j] = Id
    
            #now insert the bosonic contribution to the Lk+ terms
            b1 = Mkp(nb)

            H[2*i+1][1, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][1, :, :, 0] =b1*np.sqrt(np.conj(dk[i]))

            ##now add the bosonic contribution to the Lk- terms
            b1 = Mkm(nb)
            H[2*i+1][2, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][3, :, :, 0] =-b1*np.sqrt(np.conj(dk[i]))

            #now add on the diagonal terms
            H[2*i+1][4, :, :, 0] = -1.0j*np.diag(np.arange(nb)*zk[i])
            H[2*i+2][4, :, :, 0] = -1.0j*np.diag(np.arange(nb)*np.conj(zk[i]))
        else:
            #insert all the identity terms that carry operators through so they act on the next site
            for j in range(5):
                H[2*i+1][j, :, :, j] = Id

            H[2*i+2][0, :, :, 0] = Id
    
            #now insert the bosonic contribution to the Lk+ terms
            b1 = Mkp(nb)
            H[2*i+1][1, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][1, :, :, 0] =b1*np.sqrt(np.conj(dk[i]))

            #now add the bosonic contribution to the Lk- terms
            b1 = Mkm(nb)
            H[2*i+1][2, :, :, 0] = b1*np.sqrt(dk[i])
            H[2*i+2][3, :, :, 0] =-b1*np.sqrt(np.conj(dk[i]))

            #now add on the diagonal terms
            H[2*i+1][4, :, :, 0] = -1.0j*np.diag(np.arange(nb)*zk[i])
            H[2*i+2][4, :, :, 0] = -1.0j*np.diag(np.arange(nb)*np.conj(zk[i]))

    if transpose:
        for i in range(len(H)):
            for x in range(H[i].shape[0]):
                for y in range(H[i].shape[3]):
                    H[i][x, :, :, y] = (H[i][x, :, :, y]).T
    return H


def HEOM_Lsb_mpo(Hs, bath, transpose=True):
    Lmin = None
    if 'Lmin' in bath.keys():
        Lmin = bath['Lmin']
    return heom_Lsb_mpo(Hs, bath['S'], bath['d'], bath['z'], bath['L'], Lmin=Lmin, transpose=transpose)

def heom_Lsb_mpo(Hs, S, dk, zk, L, Lmin=1, transpose=True):
    ds = compute_dimensions(S, dk, zk, L, Lmin=Lmin)
    chi = np.ones(2*len(dk), dtype=int)*4

    H = mpo(chi=chi, d = ds, dtype=np.complex128, init='zeros')

    H[0][0, :, :, 1] = commutator(S)
    H[0][0, :, :, 2] = np.kron(S, np.identity(S.shape[0]))
    H[0][0, :, :, 3] = np.kron(np.identity(S.shape[0]), S.T)

    for i in range(len(dk)):
        nb = ds[2*i+1]
        Id = np.identity(nb)

        if(i + 1 != len(dk)):
            #insert all the identity terms that carry operators through so they act on the next site
            for j in range(4):
                H[2*i+1][j, :, :, j] = Id

            for j in range(4):
                H[2*i+2][j, :, :, j] = Id
    
            #now insert the bosonic contribution to the Lk+ terms
            b1 = Mkp(nb)

            H[2*i+1][1, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][1, :, :, 0] =b1*np.sqrt(np.conj(dk[i]))

            ##now add the bosonic contribution to the Lk- terms
            b1 = Mkm(nb)
            H[2*i+1][2, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][3, :, :, 0] =-b1*np.sqrt(np.conj(dk[i]))

        else:
            #insert all the identity terms that carry operators through so they act on the next site
            for j in range(4):
                H[2*i+1][j, :, :, j] = Id

            H[2*i+2][0, :, :, 0] = Id
    
            #now insert the bosonic contribution to the Lk+ terms
            b1 = Mkp(nb)
            H[2*i+1][1, :, :, 0] =b1*np.sqrt(dk[i])
            H[2*i+2][1, :, :, 0] =b1*np.sqrt(np.conj(dk[i]))

            #now add the bosonic contribution to the Lk- terms
            b1 = Mkm(nb)
            H[2*i+1][2, :, :, 0] = b1*np.sqrt(dk[i])
            H[2*i+2][3, :, :, 0] =-b1*np.sqrt(np.conj(dk[i]))

    if transpose:
        for i in range(len(H)):
            for x in range(H[i].shape[0]):
                for y in range(H[i].shape[3]):
                    H[i][x, :, :, y] = (H[i][x, :, :, y]).T
    return H