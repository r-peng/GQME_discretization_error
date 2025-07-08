import numpy as np
import scipy as sp

def lanczos_iter(A, v0, m=3):
    vt = np.zeros((m, *v0.shape), dtype=v0.dtype)
    c = np.linalg.norm(v0)
    vt[0] = v0/c
    Hm=np.zeros((m, m), dtype=v0.dtype)

    w = A@vt[0]
    a = np.dot(np.conj(w), vt[0])
    w = w - a*vt[0]

    #_a = np.zeros(m)
    #_b = np.zeros(m-1)
    Hm[0, 0] = a
    #_a[0] = a

    for j in range(1, m):
        b = np.dot(np.conj(w), w)**0.5

        vt[j] = w /b
        w = A@vt[j]
        a = np.dot(np.conj(w), vt[j])
        w = w-a*vt[j] - b*vt[j-1]

        #_a[j] = a
        #_b[j-1] = b
        Hm[j, j] = a
        Hm[j-1, j] = b
        Hm[j, j-1] = np.conj(b)
    return Hm, vt, c

def lanczos(A, v0, m=3):
    Hm, vt, c = lanczos_iter(A, v0, m=m)
    u, v = sp.linalg.eigh(Hm)
    ind = np.argmin(u)
    v = np.dot(v[:,ind], vt)
    return u[ind], v*c#/np.sqrt(np.dot(np.conj(v),v))


def lanczos_expmv(A, v0, coeff = 1.0, m=15):
    Hm, vt, c = lanczos_iter(A, v0, m=m)
    u, v = sp.linalg.eigh(Hm)

    expu = np.exp(coeff*u)
    v = vt.T@((v@np.diag(expu)@np.conj(v).T)[0, :])    
    return v*c


def arnoldi_iter(A, v0, m=3):
    Q = np.zeros((m, *v0.shape), dtype=v0.dtype)
    c = np.linalg.norm(v0)
    Q[0] = v0/c
    Hm=np.zeros((m, m), dtype=v0.dtype)

    for k in range(m):
        u = A@Q[k]
        for j in range(k+1):
            Hm[j, k] = Q[j]@u
            u = u - Hm[j, k]*Q[j]
        
        if k+1< m:
            Hm[k+1, k] = np.linalg.norm(u)
            if np.abs(Hm[k+1, k]) < 1e-20:
                return Hm[:k, :k], Q[:k, :]
            Q[k+1] = u/Hm[k+1, k]

    return Hm, Q, c

def arnoldi_expmv(A, v0, coeff = 1.0, m=15):
    Hm, vt, c = arnoldi_iter(A, v0, m=m)
    v = vt.T@(sp.linalg.expm(coeff*Hm)[0, :])
    return v*c
