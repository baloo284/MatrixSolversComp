#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:52:56 2018

@author: Ours Noir

Final Project:

Implement Jacobi, Gauss-Seidel, SOR, CGM, BICGSTAB and GMRES Methods. Test their convergence
for a 2D problem and different parameters (MEsh size, iterations, tolerance, etc.)

This example is for a dominant diagonal matrix builded through Finite Difference Method
to find the solution for the Laplace ecuation in two dimentions.

Boundary conditions are given by:
    
                                  T3 
                          ______________________
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                     T4  |                      |  T2
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |                      |
                         |______________________|
                                
                                  T1
                        

"""
#----------------------------------------------------------------------------------------
                               #Library Imports
#----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import Solvers as sol
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import FDM as fdm

#----------------------------------------------------------------------------------------
                           #Diffusion parameter definition
#----------------------------------------------------------------------------------------
T1 = 100
T2 = -5
T3 = 10
T4 = -34
Nxy = 25
tol = 1e-8
max_iter = 200
w = 0.1

#----------------------------------------------------------------------------------------
                     #Variables Separation to avoid overwriting
#----------------------------------------------------------------------------------------
A, b = fdm.generateSystem(Nxy, Nxy,-4,T1,T2,T3,T4) # Matriz del sistema
As = A.copy()
Aj = A.copy()
Ags = A.copy()
Asor = A.copy()
Agcm = A.copy()
Abic = A.copy()
Agmr = A.copy()

bs = b.copy()
bj = b.copy()
bgs = b.copy()
bsor = b.copy()
bgcm = np.matrix(b.copy())
bbic = np.matrix(b.copy())
bgmr = np.matrix(b.copy())
x0gcm = bgcm.copy() 
x0bic = bbic.copy()
x0gmr = bgmr.copy()

A1 = sp.csr_matrix(A)
A1gcm = A1.copy()
A1bic = A1.copy()
A1gmr = A1.copy()
b1gcm = b.copy()
b1bic = b.copy()
b1gmr = b.copy()
x01gcm = b.copy()
x01bic = b.copy()
x01gmr = b.copy()

#----------------------------------------------------------------------------------------
                               #atrix content print
#----------------------------------------------------------------------------------------
plt.figure(figsize=(10,10))
plt.matshow(A, cmap=plt.cm.flag, fignum=1)
plt.show()

#----------------------------------------------------------------------------------------
                               #Gram matrix solution
#----------------------------------------------------------------------------------------
if Nxy < 50:
#----------------------------------------------------------------------------------------
                               #Gram matrix solution Linalg
#----------------------------------------------------------------------------------------
    t1 = time.clock()
    ut = np.linalg.solve(As,bs)
    t2 = time.clock()
    te = t2 - t1
    fdm.plotSolution(ut,Nxy,Nxy,'linalg.solve({})'.format(0),'E. time : {:>3.5e}'.format(te),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Jacobi
#----------------------------------------------------------------------------------------
    t1 = time.clock()
    ut,error,it, eaJ = sol.jacobi(Aj,bj,tol,max_iter,True)
    t2 = time.clock()
    te = t2 - t1
    fdm.plotSolution(ut,Nxy,Nxy,'Jacobi ({:>8.5e} - {})'.format(error, it),'E. time : {:>3.5e}'.format(te),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Gauss-Seidel
#----------------------------------------------------------------------------------------
    t1 = time.clock()
    ut,error,it, eaGS = sol.gauss_seidel(Ags,bgs,tol,max_iter,True)
    t2 = time.clock()
    te = t2 - t1
    fdm.plotSolution(ut,Nxy,Nxy,'Gauss Seidel ({:>8.5e} - {})'.format(error, it),'E. time : {:>3.5e}'.format(te),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution SOR
#----------------------------------------------------------------------------------------
    t1 = time.clock()
    ut,error,it, eaSOR = sol.sor(Asor,bsor,tol,max_iter,w,True)
    t2 = time.clock()
    te = t2 - t1
    fdm.plotSolution(ut,Nxy,Nxy,'SOR ({:>8.5e} - {})'.format(error, it),'E. time : {:>3.5e}'.format(te),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution implemented CGM
#----------------------------------------------------------------------------------------
t1 = time.clock()
ut,residual,it, raCG = sol.cgm(Agcm,bgcm.T,x0gcm.T,tol,max_iter,True)
t2 = time.clock()
te = t2 - t1
#----------------------------------------------------------------------------------------
                               #Gram matrix solution Scipy CGM
#----------------------------------------------------------------------------------------
t1 =  time.time()
x = spla.cg(A1gcm,b1gcm,x01gcm,tol,maxiter=max_iter)
t2 = time.time()
tes = t2 - t1
fdm.plotSolution(ut,Nxy,Nxy,'CGM ({:>8.5e} - {})'.format(residual, it),'E. time : {:>2.4e} / {:>2.4e}'.format(te,tes),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution implemented BICGSTAB
#----------------------------------------------------------------------------------------
t1 = time.clock()
ut,residual,it, rabicgstab = sol.bicgstab(Abic,bbic.T,x0bic.T,tol,max_iter,True)
t2 = time.clock()
te = t2 - t1

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Scipy BICGSTAB
#----------------------------------------------------------------------------------------
t1 =  time.time()
x = spla.bicgstab(A1bic,b1bic,x01bic,tol,maxiter=max_iter)
t2 = time.time()
tes = t2 - t1
fdm.plotSolution(ut,Nxy,Nxy,'BICGSTAB ({:>8.5e} - {})'.format(residual, it),'E. time : {:>2.4e} / {:>2.4e}'.format(te,tes),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution implemented GMRES
#----------------------------------------------------------------------------------------
t1 = time.clock()
ut,residual,it, ragmres = sol.gmres(Agmr,bgmr.T,x0gmr.T,tol,max_iter,15,True)
t2 = time.clock()
te = t2 - t1

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Scipy GMRES
#----------------------------------------------------------------------------------------
t1 =  time.time()
x = spla.gmres(A1gmr,b1gmr,x01gmr,tol,maxiter=max_iter)
t2 = time.time()
tes = t2 - t1
fdm.plotSolution(ut,Nxy,Nxy,'GMRES ({:>8.5e} - {})'.format(residual, it),'E. time : {:>2.4e} / {:>2.4e}'.format(te,tes),T1,T2,T3,T4)

#----------------------------------------------------------------------------------------
                           #Solution and point cloud plotting
#----------------------------------------------------------------------------------------
plt.subplots_adjust(bottom=0.1, right=2, top=1.5, wspace=0.5, hspace=0.5)
plt.show()

#----------------------------------------------------------------------------------------
                               #Convergence comparison for all methods
#----------------------------------------------------------------------------------------
plt.figure(figsize=(10,5))
if Nxy < 50:
    plt.plot(eaJ, label='Jacobi')
    plt.plot(eaGS, label='Gauss-Seidel')
    plt.plot(eaSOR, label='SOR')
plt.plot(raCG, label='CGM')
plt.plot(rabicgstab, label='BICGSTAB')
plt.plot(ragmres, label='GMRES')
plt.semilogy()
plt.legend()
plt.title('Mesh: {} X {}, Tolerance: {}'.format(Nxy,Nxy,tol))
plt.xlabel('Iterations')
plt.ylabel('error/residual')
plt.grid()
plt.show()