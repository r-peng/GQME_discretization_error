import numpy as np
from .mps_utils import *
from .mpo import *


#a basic shared memory mps implementation.  This contains a number 
class mps:
    #construct from array of bond dimensions and local hilbert space dimensions
    def __init__(self, chi = None, d = None, N = None, chil : int = 1, chir : int = 1, dtype = np.double, init = 'zeros', do_orthog=True):
        self.dtype = dtype 


        if isinstance(chi, (np.ndarray, list)) and isinstance(d, (np.ndarray, list)):
            self.build_from_arrays(chi, d, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        elif isinstance(chi, int) and isinstance(d, (np.ndarray, list)):
            N = len(d)
            chi_arr = np.ones(N-1, dtype=int)*chi
            self.build_from_arrays(chi_arr, d, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        elif isinstance(chi, (np.ndarray, list)) and isinstance(d, int):
            N = len(chi)+1
            d_arr = np.ones(N, dtype=int)*d
            self.build_from_arrays(chi, d_arr, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        else:
            if N == None:
                self._tensors = None
                self._is_valid = False
                self._orth_centre = None
            else:
                chi_arr = np.ones(N-1, dtype=int)*chi
                d_arr = np.ones(N, dtype=int)*d
                self.build_from_arrays(chi_arr, d_arr, chil = chil, chir = chir, dtype = dtype, init = init, do_orthog=do_orthog)
        
        self._is_conjugated = False

    def build_from_arrays(self, chi, d, chil = 1, chir = 1, dtype = np.double, init = 'zeros', do_orthog=True):
        Nchi = None
        Nd = None
        if not isinstance(chi, (list, np.ndarray)):
            raise TypeError("Bond dimension variable is an invalid type.")
        if not isinstance(d, (list, np.ndarray)):
            raise TypeError("Local Hilbert Space Dimension variable is an invalid type.")

        Nchi = len(chi)
        Nd = len(d)

        if not (Nchi + 1 == Nd):
            raise RuntimeError("bond dimension and local hilbert space arrays are not compatible.")

        self._tensors = [None]*Nd
        if Nd > 1:
            self._tensors[0] = np.zeros((chil, d[0], chi[0]), dtype=dtype)
            for i in range(1, Nd-1):
                self._tensors[i] = np.zeros((chi[i-1], d[i], chi[i]), dtype=dtype)
            self._tensors[-1] = np.zeros((chi[-1], d[-1], chir), dtype=dtype)
        elif Nd == 1:
            self._tensors[0] = np.zeros((chil, d[0], chir), dtype=dtype)

        self._orth_centre = None
        self._is_valid = True
        self.init_values(dtype=dtype, init=init, do_orthog=do_orthog)

    def init_values(self, dtype = np.double, init='zeros', do_orthog=True, sf = 1):
        if isinstance(init, str):
            if init == 'zeros':
                for A in self._tensors:
                    A = np.zeros(A.shape, dtype=dtype)

            elif init == 'random':
                for A in reversed(self._tensors):
                    if(dtype == np.complex128):
                        A += np.random.normal(0, 1, size = (np.prod(A.shape), 2)).view(np.complex128).flatten().reshape(A.shape)
                    else:
                        A += np.random.normal(0, 1, size=A.shape)
                        
                if(do_orthog):
                    for i in reversed(range(self.nbonds())):
                        self.shift_bond(i, dir='left')
                        norm = np.dot(np.conj(self._tensors[i].flatten()), self._tensors[i].flatten())
                        self._tensors[i] /= (np.sqrt(norm)*sf)

                    self._orth_centre = 0
                    
        elif isinstance(init, (np.ndarray, list)):
            if(len(init) != self.nsites()):
                raise ValueError("Invalid init array")

            for i in range(self.nsites()):
                self._tensors[i] = np.zeros(self._tensors[i].shape, dtype=dtype)
                ind = int(init[i])
                if(ind >= self._tensors[i].shape[1] or ind < 0):
                    raise IndexError("Index out of bounds for state vector initialisation")

                self._tensors[i][0, ind, 0] = 1.0
            if(do_orthog):
                self.orthogonalise()


    #access tensor by index
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[ii] for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            if self._is_conjugated:
                return np.conj(self._tensors[i])
            else:
                return self._tensors[i]
        else:
            raise TypeError("Invalid argument type")
            return None

    def __getslice__(self, i):
        return self.__getitem__(i)

    #set tensor by index
    def __setitem__(self, i, v):
        #first we check that v is a valid type
        if not isinstance(v, np.ndarray):
            raise TypeError("Invalid type for setting item.")
        if not v.ndim == 3:
            raise TypeError("Invalid type for setting item.")

        if isinstance(i, slice):
            for ii in range(*i.indices(len(self))):
                self._tensors[ii] = v
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            self._tensors[i] = v
        else:
            raise TypeError("Invalid argument type")

        self._orth_centre = None
        self._is_valid = None
        self.is_valid()

    def __setslice__(self, i, v):
        self.__setitem__(i, v)

    def is_ortho(self):
        return not self._orth_centre == None

    def __len__(self):
        return len(self._tensors)

    def nsites(self):
        return len(self)

    def nbonds(self):
        return len(self)-1

    def sweep_order(self):
        return list(range(len(self)))

    def conj(self):
        ret = mps()
        ret._tensors = self._tensors
        ret._orth_centre = self._orth_centre
        ret._is_valid = self._is_valid
        ret._is_conjugated = not self._is_conjugated
        return ret

    def isconjugated(self):
        return self._is_conjugated

    def shape(self, i):
        if isinstance(i, slice):
            return [self.shape(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            return self._tensors[i].shape
        else:
            raise TypeError("Invalid argument type")

    def local_dimension(self, i):
        if isinstance(i, slice):
            return [self.local_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, len(self))
            return self._tensors[i].shape[1]
        else:
            raise TypeError("Invalid argument type")

    def expand_local_dimension(self, i, d):
        if isinstance(i, int):    
            i = mapint(i, self.nsites())
            if(d > self._tensors[i].shape[1]):
                npad = d - self._tensors[i].shape[1]
                self._tensors[i] = np.pad(self._tensors[i], ((0,0), (0, npad), (0,0) ), 'constant', constant_values=(0))
        else:
            raise TypeError("Invalid argument type")

    def bond_dimension(self, i):
        if isinstance(i, slice):
            return [self.bond_dimension(ii) for ii in range(*i.indices(len(self)))]
        elif isinstance(i, int):    
            i = mapint(i, self.nbonds())
            return self._tensors[i].shape[2]
        else:
            raise TypeError("Invalid argument type")

    def maximum_bond_dimension(self):
        bd = 0
        for i in range(self.nbonds()):
            if self._tensors[i].shape[2] > bd:
                bd = self._tensors[i].shape[2]
        return bd

    def expand_bond(self, i, chi):
        if isinstance(i, int):    
            i = mapint(i, self.nbonds())
            if not self.shape(i)[2] == self.shape(i+1)[0]:
                raise RuntimeError("Cannot expand bond, the specified bond is invalid.")
            if(chi > self.shape(i)[2]):
                npad = chi - self.shape(i)[2]
                self._tensors[i] = np.pad(self._tensors[i], ((0,0), (0, 0), (0,npad) ), 'constant', constant_values=(0))
                self._tensors[i+1] = np.pad(self._tensors[i+1], ((0,npad), (0, 0), (0,0) ), 'constant', constant_values=(0))
        else:
            raise TypeError("Invalid argument type")

    #a function for taking an invalid object and sanitising the bonds to make sure that the object is a valid size.
    def sanitise(self):
        #if we are valid then we don't need to do anything.  
        if not self.is_valid():
            for bi in range(self.nbonds()):
                chil = self.shape(bi)[2]
                chir = self.shape(bi+1)[0]
                if not chil == chir:
                    chi = max(chil, chir)
                    if(chi > chil):
                        npad = chi - chil
                        self._tensors[bi] = np.pad(self._tensors[i], ((0,0), (0, 0), (0,npad) ), 'constant', constant_values=(0))
                    else:
                        npad = chi - chir
                        self._tensors[i+1] = np.pad(self._tensors[i+1], ((0,npad), (0, 0), (0,0) ), 'constant', constant_values=(0))
            self._is_valid = True
        self.truncate(nbond=self.maximum_bond_dimension())


    #function for ensuring that the mps is currently valid
    def is_valid(self):
        if not self._is_valid == None:
            return self._is_valid
        else:
            iv = True
            for i in range(len(self._tensors)):
                if not self.shape(i-1)[2] == self.shape(i)[0]:
                    iv = False
            self._is_valid = iv
            return self._is_valid

    def orthogonalise(self):
        for i in reversed(range(self.nbonds())):
            self.shift_bond(i, dir='left')
        self._orth_centre = 0

    def truncate(self, tol = None, nbond=None):
        self.shift_orthogonality(-1)
        self.shift_orthogonality(0, tol=tol, nbond=nbond)

    def shift_orthogonality(self, i, tol = None, nbond = None):
        i = mapint(i, self.nsites())
        if self._orth_centre == None:
            self.orthogonalise()

        if i < self._orth_centre:
            for bi in reversed(range(i, self._orth_centre)):
                nb = None
                if isinstance(nbond, (list, np.ndarray)):
                    nb = nbond[bi]
                else:
                    nb = nbond
                self.shift_left(tol = tol, nbond = nb)

        elif i > self._orth_centre:
            for bi in range(self._orth_centre, i):
                nb = None
                if isinstance(nbond, (list, np.ndarray)):
                    nb = nbond[bi]
                else:
                    nb = nbond
                self.shift_right(tol = tol, nbond = nb)

        if(i != self._orth_centre):
            raise RuntimError("the orthogonality centre has not been shifted to the correct position.")

    def norm(self):
        norm = None
        if(self._orth_centre != None):
            oc = self._orth_centre
            norm = np.dot(np.conj(self._tensors[oc].flatten()), self._tensors[oc].flatten())
        else:
            norm = mps.overlap(self.conj(), self)
        return norm

    def normalise(self):
        norm = self.norm()
        if(self._orth_centre != None):
            oc = self._orth_centre
            self._tensors[oc] /= np.sqrt(norm)
        else:
            self._tensors[0] /= np.sqrt(norm)
        return norm


    def expand(self, T, boundary, is_orthogonalised = False):
        if not isinstance(T, np.ndarray):
            raise RuntimeError("Cannot expand mps as input tensor is invalid.")
        if T.ndim != 3:
            raise RuntimeError("Input tensors is not the correct dimension.")

        if(boundary == 'left'):
            if(T.shape[2] != self._tensors[0].shape[0]):
                raise RuntimeError("Cannot expand MPS boundary tensor has incompatible shape.")
            self._tensors.insert(0, T)
            
            if not is_orthogonalised:
                self._orth_centre = None
            else: 
                self._orth_centre += 1

        elif(boundary == 'right'):
            if(T.shape[0] != self._tensors[self.nbonds()].shape[2]):
                raise RuntimeError("Cannot expand MPS boundary tensor has incompatible shape.")

            self._tensors.append(T)
        else:
            raise RuntimeError("Failed to expand MPS.  Boundary type not recognised")

    def pop(self, boundary):
        if(boundary == 'left'):
            if not self._orth_centre == None:
                if(self._orth_centre > 0):
                    self._orth_centre -= 1
            return self._tensors.pop(0)

        elif(boundary == 'right'):
            if not self._orth_centre == None:
                if(self._orth_centre == self.nbonds() ):
                    self._orth_centre -= 1
            return self._tensors.pop(-1)

        else:
            raise RuntimeError("Failed to pop boundary tensor from MPS.  Boundary type not recognised")


    def shift_left(self, tol = None, nbond = None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")
        if self._orth_centre == None:
            raise RuntimeError("The object does not have an orthogonality centre to shift")
        if self._orth_centre == 0:
            raise RuntimeError("Orthogonality Centre cannot be shifted left")
        self.shift_bond(self._orth_centre - 1, dir='left', tol = tol, nbond = nbond)

    def shift_right(self, tol = None, nbond = None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")
        if self._orth_centre == None:
            raise RuntimeError("The object does not have an orthogonality centre to shift")
        if self._orth_centre == self.nbonds():
            raise RuntimeError("Orthogonality Centre cannot be shifted left")
        self.shift_bond(self._orth_centre , dir='right', tol = tol, nbond = nbond)


    def shift(self, dir, tol = None, nbond = None):
        if dir == 'left':
            self.shift_left(tol=tol, nbond=nbond)
        elif dir == 'right':
            self.shift_right(tol=tol, nbond=nbond)
        else:
            raise RuntimeError("Failed to shift bond incorrect direction.")

    #updates the MPS so that the site tensors are isometries and return the non-orthogonal bond matrix
    def schmidt_decomposition(self, dir, tol=None, nbond=None, chimin=None):
        if self._orth_centre == None:
            raise RuntimeError("The schmidt decomposition function requires the MPS to be in a mixed canonical form.")
        if (self._orth_centre == 0 and dir  == 'left') or (self._orth_centre + 1 == len(self) and dir == 'right'):
            raise RuntimeError("Unable to perform specified decomposition we are at the bounds of the MPS.")
        
        bind = None
        if dir == 'left':
            bind = self._orth_centre-1
        else:
            bind = self._orth_centre

        il = bind
        ir = bind+1
        
        self._tensors[il], R, self._tensors[ir] = local_canonical_form(self._tensors[il], self._tensors[ir], dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        return R

    #updates the MPS so that the site tensors are isometries and return the non-orthogonal bond matrix
    def schmidt_spectrum(self, dir, tol=None, nbond=None, chimin=None):
        if self._orth_centre == None:
            raise RuntimeError("The schmidt decomposition function requires the MPS to be in a mixed canonical form.")
        if (self._orth_centre == 0 and dir  == 'left') or (self._orth_centre + 1 == len(self) and dir == 'right'):
            raise RuntimeError("Unable to perform specified decomposition we are at the bounds of the MPS.")
        
        bind = None
        if dir == 'left':
            bind = self._orth_centre-1
        else:
            bind = self._orth_centre

        il = bind
        ir = bind+1
        
        self._tensors[il], R, self._tensors[ir] = local_canonical_form(self._tensors[il], self._tensors[ir], dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        return R

    def shift_bond(self, bind, dir='right', tol=None, nbond=None, chimin=None):
        if not self.is_valid():
            raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")

        bind = mapint(bind, self.nbonds())

        #get the indices of the two sites that will be modified by the operation
        il = bind
        ir = bind+1

        self._tensors[il], self._tensors[ir] = shift_mps_bond(self._tensors[il], self._tensors[ir], dir, il, self._orth_centre, tol = tol, nbond = nbond, chimin = chimin)
        self._orth_centre = update_mps_ortho_centre(il, ir, self._orth_centre, dir)

    def state_internal(self, init):
        if isinstance(init, (np.ndarray, list)):
            if(len(init) != self.nsites()):
                raise ValueError("Invalid init array")
            M = None
            for i in range(self.nsites()):
                ind = int(init[i])
                if(ind >= self._tensors[i].shape[1] or ind < 0):
                    raise IndexError("Index out of bounds for state vector initialisation")
                if not isinstance(M, np.ndarray):
                    M = self._tensors[i][:, ind, :]
                else:
                    M = M @ self._tensors[i][:, ind, :]
            return M
        else:
            raise RuntimeError("Not enough components for state")

    def state(self, init):
        M = self.state_internal(init)
        if(M.shape[0] == 1 and M.shape[1] == 1):
            return M[0, 0]
        else:
            return M

    def __str__(self):
        if(self._orth_centre != None):
            return 'MPS: tensors: %s \n orth centre: %d'%(self._tensors, self._orth_centre)
        else:
            return 'MPS: tensors: %s'%(self._tensors)
    
    def __imul__(self, x):
        if(self._orth_centre != None):
            self._tensors[self._orth_centre] *= x
        else:
            self._tensors[0] *= x
        return self

    def __itruediv__(self, x):
        if(self._orth_centre != None):
            self._tensors[self._orth_centre] /= x
        else:
            self._tensors[0] /= x
        return self

    #function for applying a general one-site operator to the MPS
    def apply_one_site(self, M, i, shift_orthogonality = False):
        if self._is_conjugated:
            M = np.conj(M)
        i = mapint(i, self.nsites())
        
        dM = M.shape[1]
        dims = self._tensors[i].shape
        if(dM != dims[1]):
            raise RuntimeError("The one site operator and MPS site tensor do not have compatible dimensions.")

        if(self._orth_centre != None and shift_orthogonality):
            self.shift_orthogonality(i)

        self._tensors[i] = np.transpose(np.tensordot(M, self._tensors[i], axes=([1], [1])), axes=[1, 0, 2])

        if(self._orth_centre != i):
            self._orth_centre = None
        #self._tensors[i] = np.einsum('ij, ajb -> aib', M, self._tensors[i])

    #performs a swap operation of physical indices across a bond
    def swap_bond_indices(self, bi, dir='right', tol=None, nbond=None):
        #ensure that the orthogonality centre is at site bi (the left most of the two sites involved in the bond)
        self.shift_orthogonality(bi)
        M = np.transpose(self.contract_bond(bi), axes=[0, 2, 1, 3])
        self.decompose_two_site(M, bi, dir=dir, tol=tol, nbond=nbond)
        if dir  == 'right':
            self._orth_centre = bi+1
        else:
            self._orth_centre = bi

    #contracts an MPS across a bond to the two-site tensor object
    def contract_bond(self, bi):
        bi = mapint(bi, self.nbonds())
        return np.tensordot(self._tensors[bi], self._tensors[bi+1], axes=([2], [0]))

    def decompose_two_site(self, M, bi, dir = 'right', tol = None, nbond = None):
        bi = mapint(bi, self.nbonds())
        dims = M.shape

        il = bi
        ir = bi+1

        Mm = M.reshape((dims[0]*dims[1], dims[2]*dims[3]))

        Q, S, Vh = np.linalg.svd(Mm, full_matrices=False, compute_uv=True)
        nsvd = determine_truncation(S, tol = tol, nbond = nbond)

        Q = Q[:, :nsvd]
        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        if dir == 'right':
            R = S @ Vh

            self._tensors[il] = Q.reshape( (dims[0], dims[1], nsvd))
            self._tensors[ir] = R.reshape( (nsvd, dims[2], dims[3]))

            if self._orth_centre == il:
                self._orth_centre = ir

        elif dir == 'left':
            R = Q @ S
            self._tensors[il] =  R.reshape( (dims[0], dims[1], nsvd))
            self._tensors[ir] = Vh.reshape( (nsvd, dims[2], dims[3]))

            if self._orth_centre == ir:
                self._orth_centre = il
        else:
            ValueError("Invalid dir argument")

    def apply_MPO_node(Mt, nt, order = 'mpo'):
        ret = None
        if(order == 'mpo'):
            ret = np.transpose(np.tensordot(Mt, nt, axes=([2], [1])), (0, 3, 1, 2, 4))
            #ret = np.einsum('aijb, mjn -> amibn', Mt, nt)
        else:
            ret = np.transpose(np.tensordot(Mt, nt, axes=([3], [1])), (0, 3, 1, 2, 4))
            #ret = np.einsum('aibj, mjn -> amibn', Mt, nt)

        d = ret.shape
        return (ret.reshape((d[0]*d[1], d[2], d[3]*d[4])))

    def zipup_left(A, M, tol=None, order='mpo'):
        dims = A.shape
        C = mps.apply_MPO_node(M, A, order=order)
        Cs = C.shape
        #now we swap the tensors around so that up and left point to the right 
        C = C.reshape((Cs[0]*Cs[1], Cs[2]))
        #compute the svd of the c matrix pointing towards the right

        Q, S, Vh = np.linalg.svd(C, full_matrices=False, compute_uv=True)

        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        return Q[:, :nsvd].reshape((Cs[0], Cs[1], nsvd)), S@Vh

    def zipup_internal(A, M, R, tol=None, order='mpo'):
        d = A.shape
        B = mps.apply_MPO_node(M, A, order=order)
        C = np.tensordot(R, B, axes=([1], [0]))
        #C = np.einsum('ij, jkl -> ikl', R, B)
        Cs = C.shape

        C = C.reshape((Cs[0]*Cs[1], Cs[2]))
        Q, S, Vh = np.linalg.svd(C, full_matrices=False, compute_uv=True)
        nsvd = determine_truncation(S, tol = tol, is_ortho=True)

        S = np.diag(S[:nsvd])
        Vh = Vh[:nsvd, :]

        return Q[:, :nsvd].reshape((Cs[0], Cs[1], nsvd)), S@Vh


    #function for applying a general two-site operator to the MPS
    def apply_two_site(self, M, i, j, method='naive', tol = None, nbond = None, optol=None, reorthogonalise=True):
        self._apply_nsite(M, [i, j], method=method, tol=tol, nbond=nbond, optol=optol, reorthogonalise=reorthogonalise)

    def _apply_nsite(self, M, inds, method='naive', tol = None, nbond = None, optol = None, reorthogonalise = True):
        mi = [mapint(x, len(self)) for x in inds]
    
        if len(mi) > len(self):
            raise ValueError("Index array too large.")
        if len(mi) != len(set(mi)):
            raise ValueError("Index array contained duplicates.")

        if(M.shape[0] != M.shape[1]):
            raise ValueError("Input operator is incorrect size.")

        mdim = 1
        dims = [self.local_dimension(m) for m in mi]
        for i, m in enumerate(mi):
            mdim = mdim * self.local_dimension(m)

        if(mdim != M.shape[0]):
            raise ValueError("Input operator and inds array are incompatible for this MPS.")

        if self._is_conjugated:
            M = np.conj(M)

        #now we have set up the state in a suitable form for constructing the 
        if(self._orth_centre == None):
            self.orthogonalise()

        #now we convert the n-site object into an MPO
        Mt, inds, ds = permute_nsite_dims(M, mi, dims)
        Mp = nsite_mpo(Mt, ds, order='clockwise', tol=optol)

        i = inds[0]
        j = inds[-1]

        c  = 0

        if method == "naive":
            nbd = 1
            for ind in range(i, j+1):
                if(ind == inds[c]):
                    self._tensors[ind] = mps.apply_MPO_node(Mp[c], self._tensors[ind], order='clockwise')
                    nbd = Mp[c].shape[2]
                    c = c+1
                else:
                    d = self._tensors[ind].shape
                    self._tensors[ind] = mps.apply_MPO_node(identity_pad_mpo(nbd, d[1], order='clockwise'), self._tensors[ind], order='clockwise')

            if(reorthogonalise):
                self.shift_orthogonality(j)

                #and shift back to the original orthogonality centre truncating all bonds
                self.shift_orthogonality(i, tol=tol, nbond=nbond)
            else:
                self._orth_center = None

        else:
            raise ValueError("Invalid two site mpo mps contraction scheme.")


    #function for applying a general two-site operator to the MPS
    def apply_nsite(self, M, inds, method='naive', tol = None, nbond = None, optol=None, reorthogonalise=True):
        if isinstance(inds, int):
            self.apply_one_site(M, inds)
        elif isinstance(inds, list):
            if len(inds) == 1:
                self.apply_one_site(M, inds[0])
            else:
                self._apply_nsite(M, inds, method=method, tol=tol, nbond=nbond, optol=optol, reorthogonalise=reorthogonalise)
        else:
            raise RuntimeError("Invalid index object passed to apply_nsite.")



    def apply_MPO_naive(self, M, tol = None, nbond = None, reorthogonalise = True):
        check_compatible(self, M, mps, mpo)
        for i in range(self.nsites()):
            self._tensors[i] = mps.apply_MPO_node(M[i], self._tensors[i])
    
        if(reorthogonalise):
            self.truncate(tol, nbond)
        else:
            self._orth_center = None

    def apply_MPO_zipup(self, M, tol=None, nbond=None):
        check_compatible(self, M, mps, mpo)

        self.shift_orthogonality(0)
        self._tensors[0], R = mps.zipup_left(self._tensors[0], M[0], tol=tol)

        for ind in range(1, len(self)-1):
            self._orth_centre = ind
            self._tensors[ind], R = mps.zipup_internal(self._tensors[ind], M[ind], R, tol=tol)

        self._orth_centre = len(self)-1
        B = mps.apply_MPO_node(M[-1], self._tensors[-1])
        self._tensors[-1] = np.tensordot(R, B, axes=([0], [0]))
        #self._tensors[-1] = np.einsum('ij, jkl -> ikl', R, B)
        self.shift_orthogonality(0, tol=tol, nbond=nbond)


    def apply_MPO(self, M, method="naive", tol = None, nbond = None, reorthogonalise=True):
        if method == "naive":
            self.apply_MPO_naive(M, tol=tol, nbond=nbond, reorthogonalise=reorthogonalise)

        elif method == "zipup":
            if(len(self) == 1):
                raise RuntimeError("zipup has not been implemented to handle length 1 mps.")
            self.apply_MPO_zipup(M, tol=tol, nbond=nbond)
        else:
            raise RuntimeError("method not recognised.")

    #function for applying a two site tensor across a bond and swapping the resultant output indices of the bond
    def apply_bond_tensor_and_swap(self, M, bi, dir='right', tol=None, nbond=None, optol=None):
        bi = mapint(bi, self.nbonds())

        mi = None
        #if we are going the 'right' direction then we have bi as the first index and bi+1 as the second index
        if dir == 'right':
            mi = [bi, bi+1]
        else:
            mi = [bi+1, bi]

        #now we have set up the state in a suitable form for constructing the 
        if(self._orth_centre == None):
            self.orthogonalise()

        self.shift_orthogonality(bi)
        dims = [self.local_dimension(m) for m in mi]

        #now we convert the n-site object into an MPO
        Mt, inds, ds = permute_nsite_dims(M, mi, dims)
        Mp = nsite_mpo(Mt, ds, order='clockwise', tol=optol)

        i = bi
        j = bi+1

        c=0 
        for ind in range(i, j+1):
            self._tensors[ind] = mps.apply_MPO_node(Mp[c], self._tensors[ind], order='clockwise')
            c = c+1

        Mv = np.transpose(self.contract_bond(bi), axes=[0, 2, 1, 3])
        self.decompose_two_site(Mv, bi, dir=dir, tol=tol, nbond=nbond)
        if dir  == 'right':
            self._orth_centre = bi+1
        else:
            self._orth_centre = bi

    #performs a swap operation of physical indices across a bond
    def swap_bond_indices(self, bi, dir='right', tol=None, nbond=None):
        #ensure that the orthogonality centre is at site bi (the left most of the two sites involved in the bond)
        bi = mapint(bi, self.nbonds())
        self.shift_orthogonality(bi)
        M = np.transpose(self.contract_bond(bi), axes=[0, 2, 1, 3])
        self.decompose_two_site(M, bi, dir=dir, tol=tol, nbond=nbond)
        if dir  == 'right':
            self._orth_centre = bi+1
        else:
            self._orth_centre = bi


    def apply_operator(self, op, method="naive", tol=None, nbond=None, optol=None):
        if isinstance(op, mpo):
            self.apply_MPO(op, method=method, tol=tol, nbond=nbond)
        elif isinstance(op, dict):
            if not "op" in op:
                raise ValueError("Invalid dictionary for applying operator")
            else:
                if "mode" in op:
                    self.apply_one_site(op["op"], op["mode"])
                elif "modes" in op:
                    if len(op["modes"]) == 1:
                        self.apply_one_site(op["op"], op["modes"][0])
                    elif len(op["modes"]) > 1:
                        if "tol" in op:
                            tol = op["tol"]
                        if "nbond" in op:
                            nbond = op["nbond"]
                        if "method" in op:
                            method = op["method"]
                        self.apply_nsite(op["op"], op["modes"], method=method, tol=tol, nbond=nbond, optol=optol)
                    else:
                        raise ValueError("Two body or higher operators only implemented through MPO")
                else:
                    raise ValueError("Failed to read information about modes")

    def __add__(self, other):
        check_compatible(self, other, mps, mps)

        if(self.shape(0)[0] != other.shape(0)[0] or self.shape(-1)[2] != other.shape(-1)[2]):
            raise ValueError("Unable to add mps objects with different exterior bond dimensions.")
        chil = self.shape(0)[0]
        chir = self.shape(-1)[2]

        bslice = slice(0, len(self)-1)
        chis = [x+y for x,y in zip(self.bond_dimension(bslice), other.bond_dimension(bslice))]
        ds = self.local_dimension(slice(0, len(self)))

        ret = mps(chi=chis, d=ds, chil=chil, chir=chir, dtype = self.dtype, do_orthog=False)

        cs = self.bond_dimension(0)
        ret[0][:, :, 0:cs] = self[0]
        ret[0][:, :, cs:chis[0]] = other[0]
        #for all interior tensors we set up the matrices as required 
        for i in range(1, len(self)-1):
            cs = self.bond_dimension(i)
            ret[i][0:cs, :, 0:cs] = self[i]
            ret[i][cs:chis[i], :, cs:chis[i]] = other[i]

        cs = self.bond_dimension(-1)
        ret[-1][0:cs, :, :] = self[-1]
        ret[-1][cs:chis[-1], :, :] = other[-1]

        return ret

    def __sub__(self, other):
        check_compatible(self, other, mps, mps)

        if(self.shape(0)[0] != other.shape(0)[0] or self.shape(-1)[2] != other.shape(-1)[2]):
            raise ValueError("Unable to add mps objects with different exterior bond dimensions.")
        chil = self.shape(0)[0]
        chir = self.shape(-1)[2]

        bslice = slice(0, len(self)-1)
        chis = [x+y for x,y in zip(self.bond_dimension(bslice), other.bond_dimension(bslice))]
        ds = self.local_dimension(slice(0, len(self)))

        ret = mps(chi=chis, d=ds, chil=chil, chir=chir, dtype = self.dtype, do_orthog=False)

        cs = self.bond_dimension(0)
        ret[0][:, :, 0:cs] = self[0]
        ret[0][:, :, cs:chis[0]] = -other[0]

        #for all interior tensors we set up the matrices as required 
        for i in range(1, len(self)-1):
            cs = self.bond_dimension(i)
            ret[i][0:cs, :, 0:cs] = self[i]
            ret[i][cs:chis[i], :, cs:chis[i]] = other[i]

        cs = self.bond_dimension(-1)
        ret[-1][0:cs, :, :] = self[-1]
        ret[-1][cs:chis[-1], :, :] = other[-1]

        return ret

    def overlap(A, B):
        check_compatible(A, B, mps, mps);

        #now that we have checked that the sizes are correct we can perform the contraction of the MPS
        res = None
        for i in range(A.nsites()):
            Ai = A[i]
            Bi = B[i]
            #contract the first node to form the res object
            if(i == 0):
                #res = np.einsum('ajb, cjd -> acbd', A[i], B[i])
                res = np.transpose(np.tensordot(Ai, Bi, axes=([1], [1])), axes=[0, 2, 1, 3])
            else:
                #temp = np.einsum('acbd, bjm -> acmdj', res, A[i])
                temp = np.transpose(np.tensordot(res, Ai, axes=([2], [0])), axes=[0, 1, 4, 2, 3])
                res = np.tensordot(temp, Bi, axes=([3, 4], [0, 1]))
                #res = np.einsum('acmdj, djn -> acmn', temp, B[i])
        if(np.prod(res.shape)== 1):
            return res[0, 0, 0, 0]
        else:
            return res

    def matrix_element(A, M, B):
        check_compatible(A, B, mps, mps);
        check_compatible(A, M, mps, mpo)

        #now that we have checked that the sizes are correct we can perform the contraction of the MPS
        res = None
        for i in range(A.nsites()):
            #contract the first node to form the res object
            if(i == 0):
                temp = np.transpose(np.tensordot(A[i], M[i], axes=([1], [1])), axes=(0, 2, 3, 1, 4))
                #temp = np.einsum('ajb, cjkd -> ackbd', A[i], M[i])
                res = np.transpose(np.tensordot(temp, B[i], axes=([2], [1])), axes=(0, 1, 4, 2, 3, 5))
                #res = np.einsum('ackbd, ekf -> acebdf', temp, B[i])
            else:
                temp = np.einsum('acebdf, bjm -> acemjdf', res, A[i])
                temp2 = np.einsum('acemjdf, djkn -> acemnfk', temp, M[i])
                res = np.einsum('acemnfk, fko -> acemno', temp2, B[i])

        if(np.prod(res.shape)== 1):
            return res[0, 0, 0, 0, 0, 0]
        else:
            return res

    def contract(A, B=None, MPO=None):
        if B is None:
            if MPO is None:
                if A.is_ortho():
                    Ao = A[A._orth_centre]
                    return np.tensordot(np.conj(Ao), Ao, axes=([0,1,2], [0, 1, 2]))
                    #return np.einsum('ijk, ijk', np.conj(Ao), Ao)
                else:
                    return mps.overlap(A.conj(), A)
            else:
                if isinstance(MPO, mpo):
                    return mps.matrix_element(A.conj(), MPO, A)
                #allow for the evaluation of one site matrix elements through the contract function.
                #elif isinstance(MPO, dict):
        else:
            if MPO is None:
                return mps.overlap(B, A)
            else:
                return mps.matrix_element(B, MPO, A)

    def save(self, filename, tag="psi", ftype='h5'):
        if ftype == 'h5':
            import h5py
            with h5py.File(filename, 'w') as f:
                grp = f.create_group(tag)
                for i, A in enumerate(self._tensors):
                    grp.create_dataset(str(i), shape=A.shape, data=A, compression='gzip')
                grp.attrs['orthcentre'] = self._orth_centre
                grp.attrs['conj'] = self._is_conjugated
                grp.attrs['valid'] = self._is_valid
                
        elif ftype == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump({tag :  {"Ar" : [np.real(x).tolist() for x in self._tensors], "Ai" : [np.imag(x).tolist() for x in self._tensors], "orthcentre" : self._orth_centre, "conj" : self._is_conjugated, "valid" : self._is_valid}}, f)

    def load(self, filename, tag="psi", ftype = 'h5'):
        if ftype == 'h5':
            import h5py
            with h5py.File(filename, 'r') as f:
                self._tensors = [np.array(f[tag][str(i)]) for i in range(len(f[tag]))]
                self._orth_centre = f[tag].attrs['orthcentre']
                self._is_conjugated = f[tag].attrs['conj']
                self._is_valid = f[tag].attrs['valid']
        elif ftype == 'json':
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
                dv = data[tag]

                self._tensors = [(np.array(ar) + 1.0j*np.array(ai)) for ar, ai in zip(dv["Ar"], dv["Ai"])]
                self._orth_centre = dv["orthcentre"]
                self._is_conjugated = dv["conj"]
                self._is_valid = dv["valid"]

    #returns the subsystem density matrix between sites i and j
    def subsystem_density(self, i, j): 
        i = mapint(i, self.nsites())
        j = mapint(j, self.nsites())
        if  i > j:
            t = j
            j = i
            i = t
        self.shift_orthogonality(Nl)                           
        chil = A[Nl].shape[0]                               
        chir = A[Nr-1].shape[2]                             
        t = np.reshape(np.identity(chil, dtype=A.dtype), (1, 1, chil, chil)) 
        for x in range(i, j):                                 
            temp = np.tensordot(t, self._tensors[x], axes=([2], [0]))       
            t2 = np.transpose(np.tensordot(temp, np.conj(self._tensors[x]), axes=([2], [0])), axes=[0, 2, 1, 4, 3, 5])                  
            t2s = t2.shape                                      
            t = t2.reshape((t2s[0]*t2s[1], t2s[2]*t2s[3], t2s[4], t2s[5]))   
        return np.tensordot(t, np.identity(chir, dtype=A.dtype), axes=([2, 3], [0, 1]))  


    def mutual_information(self, i, j):
        i = mapint(i, self.nsites())
        j = mapint(j, self.nsites())
        if i == j:
            return 0
        if  i > j:
            t = j
            j = i
            i = t

        #now we have i < j lets set up the mutual information calculation

        #shift the orthogonality centre to i.  This means that all of the tensor network to the left of i is trivial, similarly so is all of the network to the right of j
        self.shift_orthogonality(i)

        #now lets build the reduced density operator of the two sitse
        rhoij = None
        #if site i neighbours site j - then it is trivial to construct this element
        if i + 1 == j:
            M = self.contract_bond(i)
            rhoij = np.tensordot(M, np.conj(M), axes= ([0, 3], [0, 3]))
        #if site i and j are not neighbours then we will need to contract the intermediate contracted indices together until we connect the two sites
        else:
            t = np.transpose(np.tensordot(self._tensors[i], np.conj(self._tensors[i]), axes=([0], [0])), axes=[0, 2, 1, 3])
            for k in range(i+1, j):
                t2 = np.tensordot(t, self._tensors[k], axes=([2], [0]))
                t = np.tensordot(t2, np.conj(self._tensors[k]), axes=([2, 3], [0, 1]))

            #now finally contract on the j tensors to finish everything off
            t2 = np.tensordot(t, self._tensors[j], axes=([2], [0]))
            rhoij = np.transpose(np.tensordot(t2, np.conj(self._tensors[j]), axes=([2, 4], [0, 2])), axes=[0, 2, 1, 3])


        rhoi = np.einsum('mini -> mn', rhoij)
        rhoj = np.einsum('imin -> mn', rhoij)

        rs = rhoij.shape
        rhoij = rhoij.reshape((rs[0]*rs[1], rs[2]*rs[3]))

        wij, Vij = np.linalg.eigh(rhoij)
        wi, Vi = np.linalg.eigh(rhoi)
        wj, Vj = np.linalg.eigh(rhoj)

        Sij = - np.sum(wij*np.log(wij))
        Si = - np.sum(wi*np.log(wi))
        Sj = - np.sum(wj*np.log(wj))

        return Si+Sj-Sij


def overlap(A, B):
    return mps.overlap(A, B)

def matrix_element(A, M, B):
    return mps.matrix_element(A, M, B)

def contract(A, B=None, MPO=None):
    return mps.contract(A, B=B, MPO=MPO)


#implement MPS unit tests to ensure that the implementations work correctly

    #def svd_external_bond(self, dir='right', tol = 1e-15):
    #    if not self.is_valid():
    #        raise RuntimeError("The object does not represent a valid MPS.  Unable to perform transformation operations to the MPS.")

    #    #if dir == 'right' we need to reshape and svd the left tensor
    #    if dir == 'right':
    #        il = self.nbonds()
    #        dims = self._tensors[-1].shape
    #        A = self._tensors[il].reshape((dims[0]*dims[1], dims[2]))

    #        Q, S, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)

    #        nsvd = len(S)

    #        Q = Q[:, :nsvd]
    #        S = S[:nsvd]
    #        Vh = Vh[:nsvd, :]

    #        Sinv = S
    #        for i in range(nsvd):
    #            if(S[i] > tol):
    #                Sinv[i] = 1.0/S[i]
    #            else:
    #                Sinv[i] = 0
    #        QS = Q @ np.diag(S)
    #        VS = np.diag(S) @ Vh

    #        #check to see if we truncate the bond in any way

    #        self._tensors[il] = QS.reshape((dims[0], dims[1], nsvd))
    #        return Sinv, VS


    #    #otherwise we reshape and svd the right tensor
    #    elif dir == 'left':
    #        ir = 0
    #        dims = self._tensors[ir].shape
    #        B = self._tensors[ir].reshape((dims[0], dims[1]*dims[2]))

    #        Q, S, Vh = np.linalg.svd(B, full_matrices=False, compute_uv=True, lapack_driver='gesvd')

    #        nsvd = len(S)

    #        Q = Q[:, :nsvd]
    #        S = S[:nsvd]
    #        Vh = Vh[:nsvd, :]

    #        Sinv = S
    #        for i in range(nsvd):
    #            if(S[i] > tol):
    #                Sinv[i] = 1.0/S[i]
    #            else:
    #                Sinv[i] = 0
    #        QS = Q @ np.diag(S)
    #        VS = np.diag(S) @ Vh

    #        #check to see if we truncate the bond in any way

    #        self._tensors[ir] = VS.reshape( (nsvd, dims[1], dims[2]))

    #        return Sinv, QS
    #    else:
    #        ValueError("Invalid dir argument")


    ##function for applying a matrix to the boundary of an MPS.  This is used for inverse canonical gauge based parallelism of larger MPSs 
    #def apply_boundary_matrix(self, M, boundary):
    #    if(boundary == 'left'):
    #        if(M.shape[1] != self._tensors[0].shape[0]):
    #            raise RuntimeError("Cannot apply boundary matrix to MPS.")

    #        self._tensors[0] = np.tensordot(M, self._tensors[0], axes=([1], [0]))
    #        #self._tensors[0] = np.einsum('ij, jab -> iab', M, self._tensors[0])
    #        if(self._orth_centre != 0):
    #            self._orth_centre = None

    #    elif(boundary == 'right'):
    #        if(M.shape[0] != self._tensors[self.nbonds()].shape[2]):
    #            raise RuntimeError("Cannot apply boundary matrix to MPS.")
    #        self._tensors[self.nbonds()] = np.tensordot(self._tensors[self.nbonds()], M, axes=([2], [0]))
    #        #self._tensors[self.nbonds()] = np.einsum('ij, abi -> abj', M, self._tensors[self.nbonds()])
    #        if(self._orth_centre != self.nbonds()):
    #            self._orth_centre = None

    #    else:
    #        raise RuntimeError("Failed to apply boundary matrix.  Boundary type not recognised")

