#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:35:21 2018

@author: alex, Ours Noir

Final Project:

Implement Jacobi, Gauss-Seidel, SOR, CGM, BICGSTAB and GMRES Methods. Test their convergence
for a 2D problem and different parameters (MEsh size, iterations, tolerance, etc.)

This example is for a dominant diagonal matrix builded through Radial Basis Function Method
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
from Knots import RegularMesh2D
from RBF import Multiquadric2D
from RBF import GrammMatrix
from RBF import Solver
from Plotter import plotter

#----------------------------------------------------------------------------------------
                           #Diffusion parameter definition
#----------------------------------------------------------------------------------------
D = 1
Nxy = 25
T1 = 100
T2 = -5
T3 = 10
T4 = -34
tol = 1e-8

#----------------------------------------------------------------------------------------
                          #Creation of points cloud (knots)
#----------------------------------------------------------------------------------------
mesh = RegularMesh2D(Nxy,Nxy,1,2)
mesh.create2Dmesh()

#----------------------------------------------------------------------------------------
                               #Kernel selection
#----------------------------------------------------------------------------------------
kernel = Multiquadric2D(1/np.sqrt(mesh.N()))

#----------------------------------------------------------------------------------------
                             #Gramm matrx allocation
#----------------------------------------------------------------------------------------
matrix = GrammMatrix(mesh)
matrix.fillMatrixLaplace2D(kernel,D)

#----------------------------------------------------------------------------------------
                          #Dirichlet boundary condition 
#----------------------------------------------------------------------------------------
matrix.setDirichletRegular(T1,1)
matrix.setDirichletRegular(T2,2)
matrix.setDirichletRegular(T3,3)
matrix.setDirichletRegular(T4,4)

#----------------------------------------------------------------------------------------
                               #atrix content print
#----------------------------------------------------------------------------------------
plt.figure(figsize=(10,10))
plt.matshow(matrix.getMatrix(), cmap=plt.cm.bone, fignum=1)

plt.show()

#----------------------------------------------------------------------------------------
                               #Gram matrix solution
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Linalg
#----------------------------------------------------------------------------------------

solvl = Solver(matrix,'linalg')
solvl.solve()
laml = solvl.lam()
solvl.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Jacobi
#----------------------------------------------------------------------------------------

solvj = Solver(matrix,'jacobi',test=True)
solvj.solve()
lamj = solvj.lam()
errj = solvj.err()
kj = solvj.k()
errvj = solvj.errv()
solvj.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution Gauss-Seidel
#----------------------------------------------------------------------------------------

solvgs = Solver(matrix,'gs',test=True)
solvgs.solve()
lamgs = solvgs.lam()
errgs = solvgs.err()
kgs = solvgs.k()
errvgs = solvgs.errv()
solvgs.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution SOR
#----------------------------------------------------------------------------------------

solvsor = Solver(matrix,'sor',test=True)
solvsor.solve()
lamsor = solvsor.lam()
errsor = solvsor.err()
ksor = solvsor.k()
errvsor = solvsor.errv()
solvsor.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution GCM
#----------------------------------------------------------------------------------------

solvgcm = Solver(matrix,'cgm',max_iter=290,test=True)
solvgcm.solve()
lamgcm = solvgcm.lam()
errgcm = solvgcm.err()
kgcm = solvgcm.k()
errvgcm = solvgcm.errv()
solvgcm.evaluate(kernel)


#----------------------------------------------------------------------------------------
                               #Gram matrix solution BICGSTAB
#----------------------------------------------------------------------------------------

solvbi = Solver(matrix,'bicgstab',max_iter=290,test=True)
solvbi.solve()
lambi = solvbi.lam()
errbi = solvbi.err()
kbi = solvbi.k()
errvbi = solvbi.errv()
solvbi.evaluate(kernel)

#----------------------------------------------------------------------------------------
                               #Gram matrix solution GMRES
#----------------------------------------------------------------------------------------

solvgmr = Solver(matrix,'gmres',max_iter=290,test=True)
solvgmr.solve()
lamgmr = solvgmr.lam()
errgmr = solvgmr.err()
kgmr = solvgmr.k()
errvgmr = solvgmr.errv()
solvgmr.evaluate(kernel)

#----------------------------------------------------------------------------------------
                           #Solution and point cloud plotting
#----------------------------------------------------------------------------------------

print('Steady State Adv.-Diff. Heat Transfer with RBF')
xlabel = 'Lx [m]'
ylabel = 'Ly [m]'
barlabel = 'Temparature °C'
plotl = plotter(solvl,kernel)
plotl.levelplot(title = 'linalg.solve()', xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_l.png')
#plotj = plotter(solvj,kernel)
#plotj.levelplot(title = 'Jacobi - error : %g, iteraciones : %d' % (errj, kj), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_j.png')
#plotgs = plotter(solvgs,kernel)
#plotgs.levelplot(title = title+'Gauss-Seidel - error : %g, iteraciones : %d' % (errgs, kgs), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_gs.png')
#plotsor = plotter(solvsor,kernel)
#plotsor.levelplot(title = title+'SOR - error : %g, iteraciones : %d' % (errsor, ksol), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_sor.png')
plotgcm = plotter(solvgcm,kernel)
plotgcm.levelplot(title = 'CGM - error : %g, iteraciones : %d' % (errgcm, kgcm), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_gcm.png')
plotbi = plotter(solvbi,kernel)
plotbi.levelplot(title = 'BICGSTAB - error : %g, iteraciones : %d' % (errbi, kbi), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_bi.png')
plotgmr = plotter(solvgmr,kernel)
plotgmr.levelplot(title = 'GMRES - error : %g, iteraciones : %d' % (errgmr, kgmr), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryL_gmr.png')


#----------------------------------------------------------------------------------------
                               #Convergence comparison for all methods
#----------------------------------------------------------------------------------------
plt.figure(figsize=(10,5))
if Nxy < 50:
    plt.plot(errvj, label='Jacobi')
    plt.plot(errvgs, label='Gauss-Seidel')
    plt.plot(errvsor, label='SOR')
plt.plot(errvgcm, label='CGM')
plt.plot(errvbi, label='BICGSTAB')
plt.plot(errvgmr, label='GMRES')
plt.semilogy()
plt.legend()
plt.title('Mesh: {} X {}'.format(Nxy,Nxy))
plt.xlabel('Iterations')
plt.ylabel('error/residual')
plt.grid()
plt.show()

#----------------------------------------------------------------------------------------
                               #Convergence comparison for Krylov methods
#----------------------------------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(errvgcm, label='CGM')
plt.plot(errvbi, label='BICGSTAB')
plt.plot(errvgmr, label='GMRES')
plt.semilogy()
plt.legend()
plt.title('Mesh: {} X {}'.format(Nxy,Nxy))
plt.xlabel('Iterations')
plt.ylabel('error/residual')
plt.grid()
plt.show()