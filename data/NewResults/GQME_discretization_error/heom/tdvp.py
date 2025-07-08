
import numpy as np
from .mpo import *
from .mps import *
from .local_env import mps_env
from .sweeping_utils import * 
from .lanczos import *
from .mps_utils import *
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator

class tdvp:
    def __init__(self, H, A, m=8):
        self.env = mps_env()
        self.m = m
        self.L = []
        self.R = []

    def update_env_one_site(self, A, H, Li, Ri, i, dir):
        if dir == 'right':
            self.L[i] = left_environment.compute_matrix_element(A[i], A[i], H[i], Li)
        else:
            self.R[i] = right_environment.compute_matrix_element(A[i], A[i], H[i], Ri)

    def update_two_one_site(self, A, H, Li, Ri, i, dir):
        if dir == 'right':
            self.L[i] = left_environment.compute_matrix_element(A[i], A[i], H[i], Li)
        else:
            self.R[i] = right_environment.compute_matrix_element(A[i], A[i], H[i], Ri)



    def one_site_sweep(self, H, A, dt, dir = 'left', tol=None, nbond = None, which='LM', eigopts={}):
        #get the sweeping direction 
        sweep = range(len(A))
        if(dir == 'left'):
            sweep = reversed(range(len(A)))
        sweep = list(sweep)
        us = []
        #now lets actually do the sweeping
        for i in sweep:
            #get the current local hamiltonian environment object and construct a linear operator object from it
            Hi = H[i]
            dA = A[i].shape
            
            dim = dA[0]*dA[1]*dA[2]


            #now perform the local evolution
            if(dim <= self.m):
                M = self.env.one_site_matrix(i, Hi)
                u, v = np.linalg.eigh(M)
                Af = A._tensors[i].flatten()
                A._tensors[i] = (v@(np.exp(-1.0j*dt/2.0*u)*(np.conj(v).T@Af))).reshape(dA)

            else:
                opfunc = self.env.one_site_linear_operator(i, Hi)
                op = LinearOperator(shape = (dim, dim), matvec=opfunc, matmat = opfunc)
                A._tensors[i] = arnoldi_expmv(op, A[i].flatten(), coeff = -1.0j*dt/2.0, m = self.m).reshape(A.shape(i))


            if i != sweep[-1]:
                R = A.schmidt_decomposition(dir=dir)

                self.env.update(i, A[i], A[i], H[i], dir)

                Mt = self.env.bond_matrix(i, dir)
                Rs = R.shape

                R = arnoldi_expmv(Mt, R.reshape((Rs[0]*Rs[1])), coeff = 1.0j*dt/2.0, m = self.m).reshape((Rs[0], Rs[1]))

                if dir == 'right':
                    #and now apply the action of the bond tensor to the right node of this tensor
                    A._tensors[i+1]  = np.tensordot(R, A._tensors[i+1], axes=[ [1], [0]])
                    A._orth_centre = i+1
                else:
                    #and now apply the action of the bond tensor to the right node of this tensor
                    A._tensors[i-1] = np.tensordot(A._tensors[i-1], R, axes=[ [2], [0]] )
                    A._orth_centre = i-1
        return us

    def two_site_sweep(self, H, A, dt, dir = 'left', tol=None, nbond = None):
        #get the sweeping direction 
        sweep = range(A.nbonds())
        if(dir == 'left'):
            sweep = reversed(range(A.nbonds()))
        sweep = list(sweep)

        us = []
        #now lets actually do the sweeping
        for i in sweep:
            #decompose to reform the mps representation
            A2 = A.contract_bond(i)

            #get the current local hamiltonian environment object and construct a linear operator object from it

            Hil = H[i]
            Hir = H[i+1]
            dA = A2.shape
            
            dim = dA[0]*dA[1]*dA[2]*dA[3]

            opfunc = self.env.two_site_linear_operator(i, Hil, Hir)
            op = spla.LinearOperator(shape = (dim, dim), matvec=opfunc, matmat = opfunc)
            A2 = arnoldi_expmv(op, A2.flatten(), coeff = -1.0j*dt/2.0, m = self.m).reshape(A2.shape)
            A.decompose_two_site(A2, i, dir=dir, tol = tol, nbond=nbond)

            #and update the environment object
            if(dir == 'right'):
                self.env.update(i, A[i], A[i], H[i], dir)
            else:
                if(i != 0):
                    #and update the environment object
                    self.env.update(i+1, A[i+1], A[i+1], H[i+1], dir)

            #now backwards time evolve the current orthogonality centre
            inext = i
            if dir == 'right':
                inext = i+1

            Hi = H[inext]
            dA = A[inext].shape
            
            dim = dA[0]*dA[1]*dA[2]

            if(i != sweep[-1]):
                if(dim <= self.m):
                    M = self.env.one_site_matrix(inext, Hi)
                    u, v = np.linalg.eigh(M)

                    Af = (A._tensors[inext]).flatten()
                    t1 = np.conj(v).T@Af
                    t2 = np.exp(1.0j*dt/2.0*u)*t1
                    t1 = v@t2
                    A._tensors[inext] = (t1).reshape(dA)
                else:
                    #TO DO: replace exmp_multiply with more efficient short time matrix exponential code.  
                    #now do the local time evolution
                    opfunc = self.env.one_site_linear_operator(inext, Hi)
                    op = LinearOperator(shape = (dim, dim), matvec=opfunc, matmat = opfunc)
                    A._tensors[inext] = arnoldi_expmv(op, A[inext].flatten(), coeff = 1.0j*dt/2.0, m = self.m).reshape(A.shape(inext))

        return us

    def one_site_tdvp(self, H, A, dt, tol=None, nbond = None):
        self.one_site_sweep(H, A, dt, dir= 'right', tol = tol, nbond=nbond)
        return self.one_site_sweep(H, A, dt, dir= 'left', tol = tol, nbond=nbond)


    def two_site_tdvp(self, H, A, dt, bra = None, tol=None, nbond = None, recompute_env=False):
        self.two_site_sweep(H, A, dt, dir= 'right', tol = tol, nbond=nbond)
        return self.two_site_sweep(H, A, dt, dir= 'left', tol = tol, nbond=nbond)


    def tdvp_sweep(self, H, A, dt, nsite = 1, tol=None, nbond = None, recompute_env=False):
        if not A._orth_centre == 0:
            A.orthogonalise()
            recompute_env = True
        if(len(self.env) != len(A)+2 or recompute_env):
            self.env.initialise(H, A)

        if(nsite == 1):
            self.one_site_tdvp(H, A, dt, tol=tol, nbond=nbond)
        elif (nsite == 2):
            self.two_site_tdvp(H, A, dt, tol=tol, nbond=nbond)
        else:
            raise ValueError("only 1 and 2 site tdvp have been implemented.")

def tfim_dyn(N, J, h, nbond = 32, tol = 1e-6, nsite = 2):
    from ..qc import gates
    d = np.ones(N, dtype = int)*2
    chi = np.ones(N-1, dtype=int)*2

    A = mps(chi=chi, d=d, init='random', dtype=np.complex128)

    chi = np.ones(N-1, dtype=int)*3
    H = mpo(chi =chi, d = d, init='zeros', dtype=np.complex128)

    sx = gates.X()
    sz = gates.Z()
    Id = np.identity(2)

    #setup the mpo
    for i in range(N):
        if i == 0:
            H[i][0, :, :, 0] = h*sx
            H[i][0, :, :, 1] = -J*sz
            H[i][0, :, :, 2] = Id

        elif i + 1 == N:
            H[i][0, :, :, 0] = Id
            H[i][1, :, :, 0] = sz
            H[i][2, :, :, 0] = sx

        else:
            H[i][0, :, :, 0] = Id
            H[i][1, :, :, 0] = sz
            H[i][2, :, :, 0] = sx
            H[i][2, :, :, 1] = -J*sz
            H[i][2, :, :, 2] = Id

    solver = tdvp(H, A)
    nsweeps = 10
    for i in range(nsweeps):
        us = solver.tdvp_sweep(H, A, 0.1, nsite = nsite, tol = tol, nbond=nbond)
        print(us[-1]/N)



if __name__=="__main__":
    N = 16
    J = 1
    h = 1
    tfim(N, J, h, nsite = 1)

