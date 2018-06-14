# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:52:56 2018

@author: Ours Noir

"""
import numpy as np

#Jacobi Method
def jacobi(A,b,tol,kmax,test=False):
    """
    This function solve a linear equation system using Jacobi Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        error [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        error_array [if test=true]: iteration error vector.
    """
    N = len(b)
    xnew = np.zeros(N)
    xold = np.zeros(N)
    error = 10
    error_array = np.zeros(kmax)
    k = 0
    while(error > tol and k < kmax) :
        #print(k)
        for i in range(0,N):
            xnew[i] = 0
            for j in range(0,i):
                xnew[i] += A[i,j] * xold[j]
            for j in range(i+1,N):
                xnew[i] += A[i,j] * xold[j]                
            xnew[i] = (b[i] - xnew[i]) / A[i,i]
        error = np.linalg.norm(xnew-xold)
        error_array[k] = error
        k += 1
        xold[:] = xnew[:]
#        print(k, error)
    if test:
        return xnew, error, k, error_array
    else:
        return xnew

#Gauss-Seidel Method
def gauss_seidel(A,b,tol,kmax,test=False):
    """
    This function solve a linear equation system using Gauss-Seidel Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        error [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        error_array [if test=true]: iteration error vector.
    """
    N = len(b)
    xnew = np.zeros(N)
    xold = np.zeros(N)
    error = 10
    error_array = np.zeros(kmax)
    k = 0
    while(error > tol and k < kmax) :
        #print(k)
        for i in range(0,N):
            xnew[i] = 0
            for j in range(0,i):
                xnew[i] += A[i,j] * xnew[j]
            for j in range(i+1,N):
                xnew[i] += A[i,j] * xold[j]                
            xnew[i] = (b[i] - xnew[i]) / A[i,i]
        error = np.linalg.norm(xnew-xold)
        error_array[k] = error
        k += 1
        xold[:] = xnew[:]
#        print(k, error)
    if test:
        return xnew, error, k, error_array
    else:
        return xnew

#Succesive Over-Relaxation Method
def sor(A,b,tol,kmax,w,test=False):
    """
    This function solve a linear equation system using Succesive Over-Relaxation Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        w: relaxation paramter.\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        error [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        error_array [if test=true]: iteration error vector.
    """
    N = len(b)
    xnew = np.zeros(N)
    xold = np.zeros(N)
    error = 10
    error_array = np.zeros(kmax)
    k = 0
    while(error > tol and k < kmax) :
        #print(k)
        for i in range(0,N):
            sigma = 0
            for j in range(0,i):
                sigma += A[i,j] * xnew[j]
            for j in range(i+1,N):
                sigma += A[i,j] * xold[j]                
            sigma = (b[i] - sigma) / A[i,i]
            xnew[i] = xold[i] + w * (sigma -xold[i])
        error = np.linalg.norm(xnew-xold)
        error_array[k] = error
        k += 1
        xold[:] = xnew[:]
#        print(k, error)
    if test:
        return xnew, error, k, error_array
    else:
        return xnew
    
#Conjugated Gradient Method
def cgm(A,b,x,tol,kmax,test=False):
    """
    This function solve a linear equation system using Conjugated Gradient Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        x: precondition vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        rnorm [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        rnormv [if test=true]: iteration error vector.
    """
    r = b - np.dot(A,x)
    #to resolve Mz=r0:
    #z,e,it = gauss_seidel(M,r,tol,kmax) #general (M must be defined)
    z = r #idea1
    #z = -r #idea2
    d = z
    k = 0
    rnorm = tol+1
    rnormv = np.zeros(kmax)
    while (rnorm > tol and k < kmax) :
        c = np.dot(A,d)
        alpha = float(np.dot(r.T,z)) / float(np.dot(d.T,c))
        x += alpha * d
        rp1 = r - alpha * c
        #to resolve Mz_{k+1} = r_{k+1}:
        #zp1,e,it = gauss_seidel(M,rp1,tol,kmax) #general (M must be defined)
        zp1 = rp1 #idea1
        #zp1 = -rp1 #idea2
        beta = float(np.dot(rp1.T,zp1)) / float(np.dot(r.T,z))
        dp1 = zp1 + beta * d
        rnorm = np.linalg.norm(r)
        rnormv[k] = rnorm
        k += 1
        z = zp1
        d = dp1
        r = rp1
    if test:
        return x,rnorm, k, rnormv
    else:
        return x    

#Biconjugated Stabilized Method
def bicgstab(A,b,x,tol,kmax,test=False):
    """
    This function solve a linear equation system using Biconjugated Gradient Stabilized Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        x: precondition vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        snorm [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        snormv [if test=true]: iteration error vector.
    """
    r = b - np.dot(A,x)
    rc = r
    p = rc-r
    v = rc-r
    rho = 1
    alpha = 1
    omega = 1
    k = 0
    snormv = np.zeros(kmax)
    while (k < kmax) :
        #print("iter: " + str(k))
        rhop1 = float(np.dot(rc.T,r))
        beta = (rhop1/rho)*(alpha/omega)
        p = r + beta*(p - omega*v)
        v = np.dot(A,p)
        alpha = rhop1/float(np.dot(rc.T,v))
        s = r - alpha*v
        t = np.dot(A,s)
        omega = float(np.dot(t.T,s))/float(np.dot(t.T,t))
        x += alpha*p + omega*s
        snorm = np.linalg.norm(s)
        snormv[k] = snorm
        k += 1
        if snorm < tol:
            break
        r = s-omega*t
        rho = rhop1
    
    if test:
        return x, snorm, k, snormv
    else:
        return x

#Generalized Minimal Residual Method
def gmres(A,b,x0,tol,kmax,restart,test=False):
    """
    This function solve a linear equation system using Generalized Minimal Residual Method.
    
    @param:
        A: matrix with the linear equation system.\n
        b: constant terms vector.\n
        x0: precondition vector.\n
        tol: tolerance limit.\n
        kmax: iterations upper limit.\n
        restart: upper limit for restart(useless).\n
        test: flag to indicate which values need to be returned.
        
    @return:
        xnew: solution vector.\n
        rnorm [if test=true]: minimum obtained error.\n
        k [if test=true]: iterations realized.\n
        rnormv [if test=true]: iteration error vector.
    """

    k = 1
    rnormv = np.zeros(kmax+1) 
    # Iteration k = 1
    b0 = A * x0
    r0 = b - b0
    p1 = 1/np.linalg.norm(r0) * r0
    b1 = A * p1
    t = float((b1.T * r0) / (b1.T * b1))
    x1 = x0 + t * p1
    r1 = r0 - t * b1 # b - A * x1    
    # Iteration k = 2
    k += 1    
    P = p1
    B = b1
    x = x1
    r = r1    
    rnorm = np.linalg.norm(r)
    rnormv[0] = rnorm    
    while rnorm > tol and k < kmax:
    
        beta = [float(p * r) for p in P.T]
        pp = r
        for b, p in zip(beta, P.T):
            pp = pp - b * p.T
        pnorm = 1 / np.linalg.norm(pp) * pp
        b = A * pnorm
        P = np.concatenate((P, pnorm), axis=1)
        B = np.concatenate((B, b), axis=1)
        Q, R = np.linalg.qr(B)
        inter = Q.T * r
        rows, cols = R.shape
        t = np.zeros((rows, 1))
        for i in range(0, rows):
            row = rows - 1 - i
            col = row
            tp1 = inter[row, 0]
            for j in range(row + 1, cols):
                tp1 = tp1 - R[row, j] * t[j, 0]
            t[row, 0] = tp1 / R[row, col]
        x_next = x + P * t
        r_next = r - B * t
        x = x_next.copy()
        r = r_next.copy()
        rnorm = np.linalg.norm(r)
        rnormv[k] = rnorm
        k += 1
    if test:
        return x,rnorm, k, rnormv
    else:
        return x
