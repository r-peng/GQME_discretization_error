import numpy as np
import scipy.linalg 
import time,itertools,pickle
sx = np.array([[0, 1], [1, 0]], dtype = np.complex128)
sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype = np.complex128)
def H2L(H,thresh=None):
    n = H.shape[0]
    L = np.einsum('ij,kl->ikjl',H,np.eye(n))
    L -= np.einsum('ij,lk->ikjl',np.eye(n),H)
    L = L.reshape((n**2,)*2)
    # check
    if thresh is not None:
        rho = random_rho(n,True)
        rho1 = np.dot(H,rho)-np.dot(rho,H)
        rho2 = np.dot(L,rho.flatten())
        assert np.linalg.norm(np.diag(rho1).sum())<thresh
        assert np.linalg.norm(rho1.flatten()-rho2)<thresh
    return L 
def onorm(M):
    _,s,_ = np.linalg.svd(M)
    return s[0]
def trace_norm(X):
    XX = np.dot(X.T.conj(),X)
    w = np.linalg.eigvalsh(XX)
    return np.sqrt(w).sum()
def choi(M,n,check=1e-10):
    nsq = n**2
    M = M.reshape(n,n,n,n).transpose(0,2,1,3).reshape(nsq,nsq)
    err = np.linalg.norm(M-M.T.conj())
    if err>check:
        print('hermitian error=',err)
    M = (M+M.T.conj())/2
    w,v = np.linalg.eigh(M)
    v = v.reshape(n,n,nsq)
    vv = np.einsum('kia,kja,a->ij',v.conj(),v,w)
    M = M.reshape(n,n,n,n).transpose(0,2,1,3).reshape(nsq,nsq)
    return M,w,vv
def fixed_point(U,n,check=1e-10):
    w,v = np.linalg.eig(U)
    w = np.absolute(w)
    idx = np.argsort(w)
    assert np.fabs(w[idx[-1]]-1)<check
    rho = v[:,idx[-1]].reshape(n,n)
    assert np.linalg.norm(rho-rho.T.conj())/n<check
    return w[idx],rho/np.trace(rho)
def process_trajectory(rhos,method='cvx'):
    if len(rhos.shape)==2:
        stop,nsq = rhos.shape
        n = int(np.sqrt(nsq+1e-6)) 
        rhos = rhos.reshape(stop,n,n)
    else:
        stop,n,_ = rhos.shape
    for i in range(stop):
        rhos[i] = (rhos[i]+rhos[i].T.conj())/2
        if method is not None:
            fit = FitDM(rhos[i],method)
            rhos[i] = fit.solve() 
    return rhos
def rho2U(rhos,mitigate=True,check=True):
    rhos = np.array(rhos)
    if len(rhos.shape)==4:
        nsq,stop,n,_ = rhos.shape
        rhos = rhos.reshape(nsq,stop,nsq)
    else:
        nsq,stop,_ = rhos.shape
        n = int(np.sqrt(nsq+1e-6)) 
    rhos = rhos.transpose(1,2,0)

    U = np.zeros_like(rhos)
    U[0] = np.eye(nsq)
    u,s,v = np.linalg.svd(rhos[0])
    if s[-1]<1e-4:
        print('Linear dependecy in intial trajectories=',s)
        exit()
    rho0_inv = np.linalg.inv(rhos[0])
    for i in range(1,stop):
        U[i] = np.dot(rhos[i],rho0_inv)
        if mitigate:
            fit = FitChannel(U[i],n)
            U[i] = fit.solve()
        if check:
            U[i],w,vv = choi(U[i],n)
            assert np.linalg.norm(vv-np.eye(n))<1e-10
    return U
def U2T(U,check=1e-10):
    #print('computing T...')
    stop,nsq,_ = U.shape
    n = int(np.sqrt(nsq+1e-6)) 
    T = U.copy() 
    for N in range(2,stop):
        for m in range(1,N):
            T[N] -= np.dot(T[N-m],U[m])
        # check TK
        if check is None:
            continue
        T[N],w,vv = choi(T[N],n,check=check)
        assert np.linalg.norm(vv)<check
    T[0] = 0
    Tn = np.array([onorm(Ti) for Ti in T])
    return T,Tn
def T2U(U,T,stop,mitigate=False,check=1e-10):
    #print('computing U...')
    U = list(U)
    start = len(U)
    M,nsq,_ = T.shape
    n = int(np.sqrt(nsq+1e-6)) 
    for N in range(start,stop):
        U.append(np.zeros_like(U[-1]))
        for m in range(1,min(M,N+1)):
            U[N] += np.dot(T[m],U[N-m])
        if mitigate:
            fit = FitChannel(U[N],n)
            U[N] = fit.solve()
        if check is not None:
            U[N],w,vv = choi(U[N],n)
            if np.linalg.norm(vv-np.eye(n))>check:
                print(N,np.linalg.norm(vv-np.eye(n)))
            #assert np.linalg.norm(vv-np.eye(n))<check
    return np.array(U)
def T2U_MPDI(U,T,G1,stop,mitigate=False,check=True):
    U = list(U) 
    start = len(U)
    M,nsq,_ = T.shape
    n = int(np.sqrt(nsq+1e-6)) 

    U[0] = np.eye(4)
    if start==1:
        U.append(.5*G1+.5*T[1])
    for N in range(max(2,start),stop):
        U.append(np.zeros_like(U[-1]))
        for m in range(1,min(M,N+1)):
            if m==N:
                U[N] += .5*T[m]
            else:
                U[N] += np.dot(T[m],U[N-m])
        if mitigate:
            fit = FitChannel(U[N],n)
            U[N] = fit.solve()
        if check:
            U[N],w,vv = choi(U[N],n)
            assert np.linalg.norm(vv-np.eye(n))<1e-10
    return np.array(U)
class FitDM:
    def __init__(self,rho_raw,method):
        self.rho_raw = rho_raw
        self.n = rho_raw.shape[0]
        self.nsq = self.n**2
        self.xsz = self.nsq
        self.sh = self.n,self.n
        self.method = method
        self.method = 'proj'

        #self.check_derivative()
    def preprocess(self):
        w,v = np.linalg.eigh(self.rho_raw)
        if w[0]>-1e-15:
            return self.rho_raw/w.sum(),None
        w[w<0] = 0
        w /= w.sum() 
        v = v*np.sqrt(w).reshape(1,self.n)
        if self.method=='proj':
            return np.dot(v,v.T.conj()),None
        return None,v
    def postprocess(self,V):
        rho = np.dot(V,V.T.conj())
        return rho/np.trace(rho)
    def solve(self,method='BFGS'):
        rt,x0 = self.preprocess()
        if rt is not None:
            return rt
        x0 = self.V2x(x0)
        if method=='BFGS':
            options = {'xrtol':1e-12,'maxiter':500}
            jac = True
        if method=='Nelder-Mead':
            options = {'ftol':1e-12,'maxiter':500}
            jac = False
        self.method = method
        res = scipy.optimize.minimize(self.fun,x0,method=method,jac=jac,callback=None,tol=1e-10,options=options)
        #if res['status']!=0:
        #    print(res)
        return self.postprocess(self.x2V(res['x']))
    def x2V(self,x):
        xr,xi = x[:self.xsz],x[self.xsz:]
        xr = xr.reshape(*self.sh) 
        xi = xi.reshape(*self.sh) 
        return xr+1j*xi
    def V2x(self,V):
        V = V.flatten()
        return np.concatenate([V.real,V.imag])
    def fun(self,x):
        self.f,g = self.Vfun(self.x2V(x))
        self.g = self.V2x(g)
        return self.f,self.g
    def Vfun(self,V):
        # V -> rho_ -> rho
        rho_ = np.dot(V,V.T.conj())
        tr = np.trace(rho_)
        rho = rho_/tr
        D = rho-self.rho_raw
        N = np.linalg.norm(D)
        # dN/drho*
        g = D/N
        # dN/drho_*
        g -= np.eye(self.n)*np.einsum('ij,ij->',g,rho.conj())
        g /= tr
        # dN/dV*
        g = np.dot(g.T.conj()+g,V)
        return N,g
    def check_derivative(self):
        n = self.n
        V = np.random.rand(n,n)*2-1
        V = V + 1j*(np.random.rand(n,n)*2-1)

        rho_raw = np.random.rand(n,n)*2-1
        rho_raw = rho_raw + 1j*(np.random.rand(n,n)*2-1)
        rho_raw += rho_raw.T.conj()
        self.rho_raw = rho_raw

        # norm derivative
        _,g = self.Vfun(V)
        
        import torch
        rho_raw = torch.tensor(rho_raw,requires_grad=False)
        V = torch.tensor(V,requires_grad=True)
        rho_ = torch.matmul(V,V.T.conj())
        rho = rho_/torch.trace(rho_)
        D = rho-rho_raw
        L = torch.norm(D)
        L.backward()
        assert np.linalg.norm(g-(V.grad).numpy(force=True))<1e-10
        exit()
class FitChannel(FitDM):
    def __init__(self,Uraw,n):
        Uraw = Uraw.reshape(n,n,n,n).transpose(0,2,1,3).reshape(n**2,n**2)
        assert np.linalg.norm(Uraw-Uraw.T.conj())<1e-10
        self.Uraw = Uraw
        self.n = n
        self.nsq = n**2
        self.xsz = self.nsq**2
        self.sh = n,n,self.nsq

        #self.check_derivative()
    def preprocess(self):
        n,nsq = self.n,self.nsq
        w,v = np.linalg.eigh(self.Uraw)
        pos = False
        if w[0]>-1e-15:
            pos = True
        w[w<0] = 0
        K = v.reshape(n,n,nsq)*np.sqrt(w)
        if pos:
            M = np.einsum('kia,kja->ij',K.conj(),K)
            if np.linalg.norm(M-np.eye(n))<1e-10:
                return self.Uraw.reshape(n,n,n,n).transpose(0,2,1,3).reshape(nsq,nsq),None
        return None,K 
    def postprocess(self,K):
        n,nsq = self.n,self.nsq
        M = np.einsum('kia,kja->ij',K.conj(),K)
        w,u = np.linalg.eigh(M)
        wsqrt = np.sqrt(w)
        Q = np.dot(u/wsqrt.reshape(1,n),u.T.conj())
        V = np.einsum('ika,kj->ija',K,Q).reshape(nsq,nsq)
        U = np.dot(V,V.T.conj())
        return U.reshape(n,n,n,n).transpose(0,2,1,3).reshape(nsq,nsq)
    def Vfun(self,K):
        M = np.einsum('kia,kja->ij',K.conj(),K)
        w,u = np.linalg.eigh(M)
        wsqrt = np.sqrt(w)
        Q = np.dot(u/wsqrt.reshape(1,self.n),u.T.conj())
        V = np.einsum('ika,kj->ija',K,Q).reshape(self.nsq,self.nsq)
        U = np.dot(V,V.T.conj())
        D = U-self.Uraw
        N = np.linalg.norm(D)
        # dN/dU*
        g = D/N
        # dN/dV*
        g = np.dot(g.T.conj()+g,V).reshape(self.n,self.n,self.nsq)
        # dN/dK* direct
        gK = np.einsum('mja,nj->mna',g,Q.conj())
        gQ = np.einsum('ina,ima->mn',g,K.conj())
        # dN/du*, dN/dw
        gu = np.dot(gQ+gQ.T.conj(),u)/wsqrt.reshape(1,self.n)
        tmp = np.einsum('in,jn->ijn',u,u.conj())
        gw = np.einsum('ij,ijn->n',gQ.real,tmp.real)
        gw += np.einsum('ij,ijn->n',gQ.imag,tmp.imag)
        gw /= -2*wsqrt*w
        # dN/dM*
        F = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                F[i,j] = 1/(w[j]-w[i])
                F[j,i] = -F[i,j]
        gM = F*np.dot(u.T.conj(),gu)
        gM += gM.T.conj()
        gM /= 2
        np.fill_diagonal(gM,gw)
        gM = np.linalg.multi_dot([u,gM,u.T.conj()])
        # dN/dK*
        gK += np.einsum('mib,in->mnb',K,gM+gM.T.conj())  
        return N,gK
    def check_derivative(self):
        nsq = self.nsq
        n = self.n
        K = np.random.rand(n,n,nsq)*2-1
        K = K + 1j*(np.random.rand(n,n,nsq)*2-1)

        Uraw = np.random.rand(nsq,nsq)*2-1
        Uraw = Uraw + 1j*(np.random.rand(nsq,nsq)*2-1)
        Uraw += Uraw.T.conj()
        self.Uraw = Uraw

        # norm derivative
        t = time.time()
        _,g = self.Vfun(K)
        
        import torch
        Uraw = torch.tensor(Uraw,requires_grad=False)
        K = torch.tensor(K,requires_grad=True)

        M = torch.einsum('kia,kja->ij',K.conj(),K)
        w,u = torch.linalg.eigh(M)
        Q = torch.matmul(u/torch.sqrt(w).reshape(1,n),u.T.conj())
        V = torch.einsum('ika,kj->ija',K,Q).reshape(nsq,nsq)
        U = torch.matmul(V,V.T.conj())
        D = U-Uraw
        L = torch.norm(D)
        L.backward()
        assert np.linalg.norm(g-(K.grad).numpy(force=True))<1e-10
        exit()
