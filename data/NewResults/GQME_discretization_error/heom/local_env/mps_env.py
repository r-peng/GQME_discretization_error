import numpy as np
from ..mpo import *
from ..mps import *

#functions for contracting a single slice of the environment tensor
class environment_slice:
    def compute_matrix_element(Ai, Bi, Hi):
        #if we don't have an r-tensor then we are just contracting the mpo and mps tensors
        temp = np.transpose(np.tensordot(np.conj(Bi), Hi, axes=([1], [1])), axes=(0, 2, 3, 1, 4))
        return np.transpose(np.tensordot(temp, Ai, axes=([2], [1])), axes=(0, 1, 4, 2, 3, 5))

    def compute_overlap(Ai, Bi):
        return np.tensordot(np.conj(Bi), Ai, axes=([1], [1]))

    def compute(Ai, Bi, Hi = None):
        if not isinstance(Hi,np.ndarray):
            return compute_matrix_element(Ai, Bi, Hi)
        else:
            return compute_overlap(Ai, Bi)


#functions for computing the right environment tensor
class right_environment:
    def compute_matrix_element(Ai, Bi, Hi, Ri = None):
        if not isinstance(Ri, np.ndarray):
            Rt = environment_slice.compute_matrix_element(Ai, Bi, Hi)
            s = Rt.shape
            return Rt.reshape((s[0], s[1], s[2]))
        else:
            #otherwise we need to contract the mpo and mps tensors into the rtensor
            temp = np.tensordot(Ri, np.conj(Bi), axes=([0], [2]))
            temp2 = np.tensordot(temp, Hi, axes=([3, 0], [1, 3]))
            return np.tensordot(temp2, Ai, axes = ([3, 0], [1, 2]))

    def compute_overlap(Ai, Bi, Ri = None):
        if not isinstance(Ri, np.ndarray):
            Rt = environment_slice.compute_overlap(Ai, Bi)
            s = Rt.shape
            return Rt.reshape((s[0], s[2]))
        else:
            #otherwise we need to contract the mpo and mps tensors into the rtensor
            temp = np.tensordot(np.conj(Bi), Ri, axes=([2], [0]))
            return np.tensordot(temp, Ai, axes = ([1, 2], [1, 2]))

    def compute(Ai, Bi, Hi = None, Ri = None):
        if not isinstance(Hi, np.ndarray):
            return right_environment.compute_overlap(Ai, Bi, Ri = Ri)
        else:
            return right_environment.compute_matrix_element(Ai, Bi, Hi, Ri = Ri)


#functions for computing the left environment tensor
class left_environment:
    def compute_matrix_element(Ai, Bi, Hi, Li = None):
        if not isinstance(Li, np.ndarray):
            Lt = environment_slice.compute_matrix_element(Ai, Bi, Hi)
            s = Lt.shape
            return Lt.reshape((s[3], s[4], s[5]))
        else:
            #otherwise we need to contract the mpo and mps tensors into the rtensor
            temp = np.tensordot(Li, np.conj(Bi), axes=([0], [0]))
            temp2 = np.tensordot(temp, Hi, axes=([0, 2], [0,1]))
        return np.tensordot(temp2, Ai, axes = ([0, 2], [0, 1]))

    def compute_overlap(Ai, Bi, Li = None):
        if not isinstance(Li, np.ndarray):
            Lt = environment_slice.compute_overlap(Ai, Bi)
            s = Lt.shape
            return Lt.reshape((s[1], s[3]))
        else:
            #otherwise we need to contract the mpo and mps tensors into the rtensor
            temp = np.tensordot(Li, np.conj(Bi), axes=([0], [0]))
            return np.tensordot(temp, Ai, axes = ([0, 1], [0, 1]))


    def compute(Ai, Bi, Hi = None, Li = None):
        if not isinstance(Hi, np.ndarray):
            return left_environment.compute_overlap(Ai, Bi, Li = Li)
        else:
            return left_environment.compute_matrix_element(Ai, Bi, Hi, Li = Li)










class mps_env:
    def __init__(self):
        self.L = []
        self.R = []
        self.ncs = []

    def __len__(self):
        return len(self.L)
    
    def initialise(self, H, A, B=None):
        if B == None:
            B = A

        self.R = [None]*(len(A)+2)
        Rc = None
        for i in reversed(range(A.nsites())):
            Rc = right_environment.compute_matrix_element(A[i], B[i], H[i], Ri = Rc)
            self.R[i] = Rc

        #setup r tensors for the end points so that we don't need to do anything special with the edges
        self.R[len(A)] = np.ones((1,1,1), dtype = np.complex128)
        self.R[-1] = np.ones((1,1,1), dtype = np.complex128)

        self.L = [None]*(len(A)+2)
        Lc = None
        for i in range(A.nsites()):
            Lc = left_environment.compute_matrix_element(A[i], B[i], H[i], Li = Lc)
            self.L[i] = Lc

        self.nc = [0]*len(A)
        self.L[len(A)] = np.ones((1,1,1), dtype = np.complex128)
        self.L[-1] = np.ones((1,1,1), dtype = np.complex128)

    def update(self, i, Ai, Bi, Hi, dir):
        if(dir == 'right'):
            self.L[i] = left_environment.compute_matrix_element(Ai, Bi, Hi, self.L[i-1])
        else:
            self.R[i] = right_environment.compute_matrix_element(Ai, Bi, Hi, self.R[i+1])

    def bond_matrix(self, i,dir, coeff = 1.0):
        Lm = None
        Rm = None
        if dir == 'right':
            Lm = self.L[i]
            Rm = self.R[i+1]
        elif dir == 'left':
            Lm = self.L[i-1]
            Rm = self.R[i]
        Mt = np.transpose(np.tensordot(Lm, Rm, axes=[[1], [1]]), axes=[0, 2, 1, 3])
        Mts = Mt.shape
        return Mt.reshape((Mts[0]*Mts[1], Mts[2]*Mts[3]))

    def one_site_matrix(self, i, Hi, coeff = 1.0):
        Li = self.L[i-1]
        Ri = self.R[i+1]

        temp = np.tensordot(Li, Hi, axes=([1], [0]))
        Mt = np.tensordot(temp, Ri, axes=([4], [1]))
        Mt = np.transpose(Mt, axes=[0, 2, 4, 1, 3, 5])
        dm = Mt.shape
        return Mt.reshape((dm[0]*dm[1]*dm[2], dm[3]*dm[4]*dm[5]))

    #functions for constructing the one site linear operator object.
    def one_site_linear_operator(self, i, Hi, coeff = 1.0, transpose=False):    
        Li = self.L[i-1]
        Ri = self.R[i+1]
        def f(v):
            av = None
            if v.ndim == 1:
                av = coeff*v.reshape((1, Li.shape[2], Hi.shape[2], Ri.shape[2]))
            elif v.ndim == 2:
                vt = v.T
                av = coeff*vt.reshape((vt.shape[0], Li.shape[2], Hi.shape[2], Ri.shape[2]))

            #first contract the tensor representation of v with L
            temp = None
            if not transpose:
                temp = np.tensordot(Li, av, axes=([2], [1]))
                temp2 = np.tensordot(temp, Hi, axes=([1, 3], [0, 2]))
                temp = np.transpose(np.tensordot(temp2, Ri, axes=( [4, 2], [1, 2])), [1, 0, 2, 3])

            else:
                temp = np.tensordot(Li, av, axes=([0], [1]))
                temp2 = np.tensordot(temp, Hi, axes=([0, 3], [0, 1]))
                temp = np.transpose(np.tensordot(temp2, Ri, axes=( [2, 4], [0, 1])), [1, 0, 2, 3])

            if v.ndim == 1:
                return temp.reshape( (temp.shape[0]*temp.shape[1]*temp.shape[2]*temp.shape[3]))
            else:
                return temp.reshape( (temp.shape[0], temp.shape[1]*temp.shape[2]*temp.shape[3])).T
        return f

    def one_site_linear_operator_transpose(self, i, Hi, coeff = 1.0):
        Li = self.L[i-1]
        Ri = self.R[i+1]
        def f(v):
            av = None
            if v.ndim == 1:
                av = coeff*v.reshape((1, Li.shape[2], Hi.shape[2], Ri.shape[2]))
            elif v.ndim == 2:
                vt = v.T
                av = coeff*vt.reshape((vt.shape[0], Li.shape[2], Hi.shape[2], Ri.shape[2]))

            #first contract the tensor representation of v with L
            temp = np.tensordot(Li, av, axes=([0], [1]))
            temp2 = np.tensordot(temp, Hi, axes=([0, 3], [0, 1]))
            temp = np.transpose(np.tensordot(temp2, Ri, axes=( [2, 4], [0, 1])), [1, 0, 2, 3])

            if v.ndim == 1:
                return temp.reshape( (temp.shape[0]*temp.shape[1]*temp.shape[2]*temp.shape[3]))
            else:
                return temp.reshape( (temp.shape[0], temp.shape[1]*temp.shape[2]*temp.shape[3])).T
        return f

    def one_site_linear_operator_trace(self, i, Hi, coeff=1.0):
        Li = self.L[i-1]
        Ri = self.R[i+1]

        #get the required L1 vector for evaluating the trace.  This is obtained by taking required diagonal of the rank 3 tensor and summing over the diagonal axis
        L1 = np.sum(np.diagonal(Li, axis1=0, axis2=2), axis=1)

        #do the exact same thing for R1
        R1 = np.sum(np.diagonal(Ri, axis1=0, axis2=2), axis=1)

        #now form the required matrix of Hi
        H1 = np.sum(np.diagonal(Hi, axis1=1, axis2=2), axis=2)

        M =  coeff*(L1@H1@R1)
        return M

    #a function for evaluating the value of the single site environment tensor network given the left and right environment tensors
    def one_site_environment(self, i, Ti, coeff = 1.0, Hi = None):
        Li = self.L[i-1]
        Ri = self.R[i+1]
        if not isinstance(Hi, np.ndarray):
            #first contract T into L
            temp = np.tensordot(coeff*Li, Ti, axes=([1], [0]))
            return np.tensordot(temp, Ri, axes=([2], [1]))
        else:
            temp = np.tensordot(coeff*Li, Ti, axes=([2], [0]))
            temp2 = np.tensordot(temp, Hi, axes=([1,2], [0, 2]))
            return np.tensordot(temp2, Ri, axes=([3,1], [1, 2]))



    def two_site_linear_operator(self, i, Hi, Hir, coeff = 1.0, transpose=False):
        Li = self.L[i-1]
        Ri = self.R[i+2]
        def f(v):
            av = None
            if v.ndim == 1:
                av = coeff*v.reshape((1, Li.shape[2], Hi.shape[2], Hir.shape[2], Ri.shape[2]))
            elif v.ndim == 2:
                av = coeff*v.reshape((v.shape[0], Li.shape[2], Hi.shape[2], Hir.shape[2], Ri.shape[2]))

            #first contract the tensor representation of v with L
            temp2 = None
            if not transpose:
                temp = np.tensordot(Li, av, axes=([2], [1]))
                temp2 = np.tensordot(temp, Hi, axes=([1, 3], [0, 2]))
                temp = np.tensordot(temp2, Hir, axes=([5, 2], [0, 2]))
                temp2 = np.transpose(np.tensordot(temp, Ri, axes=( [5, 2], [1, 2])), [1, 0, 2, 3, 4])
            else:
                temp = np.tensordot(Li, av, axes=([0], [1]))
                temp2 = np.tensordot(temp, Hi, axes=([0, 3], [0, 1]))
                temp = np.tensordot(temp2, Hir, axes=([5, 2], [0, 1]))
                temp2 = np.transpose(np.tensordot(temp, Ri, axes=( [5, 2], [0, 1])), [1, 0, 2, 3, 4])
            self.nc[i] += 1
            if v.ndim == 1:
                return temp2.reshape( (temp2.shape[0]*temp2.shape[1]*temp2.shape[2]*temp2.shape[3]*temp2.shape[4]))
            else:
                return temp2.reshape( (temp2.shape[0], temp2.shape[1]*temp2.shape[2]*temp2.shape[3]*temp2.shape[4]))
        return f

    #a function for evaluating the two site environment tensor network given the left and right environment tensors
    def two_site_environment(i, Ti, Tir, coeff = 1.0, Hi = None, Hir = None):
        Li = self.L[i-1]
        Ri = self.R[i+2]
        if not isinstance(Hi, np.ndarray) or not isinstance(Hir, np.ndarray):
            #first contract T into L
            temp = np.tensordot(coeff*Li, Ti, axes=([1], [0]))
            temp2 = np.tensordot(temp, Tir, axes = ([2], [0]))
            return np.tensordot(temp2, Ri, axes=([3], [1]))
        else:
            temp = np.tensordot(coeff*Li, Ti, axes=([2], [0]))
            temp2 = np.tensordot(temp, Hi, axes=([1,2], [0, 2]))
            temp = np.tensordot(temp2, Tir, axes=([1], [0]))
            temp2 = np.tensordot(temp, Hir, axes=([2, 3], [0, 2]))
            return np.tensordot(temp2, Ri, axes=([4, 2], [1, 2]))
