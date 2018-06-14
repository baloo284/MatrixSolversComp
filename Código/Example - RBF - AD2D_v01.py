#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:20:39 2018

@author: alex, Ours Noir

Final Project:

Implement Jacobi, Gauss-Seidel, SOR, CGM, BICGSTAB and GMRES Methods. Test their convergence
for a 2D problem and different parameters (MEsh size, iterations, tolerance, etc.)

This example is for a sparse matrix builded through Radial Basis Function Method
to find the solution for the Advection-Diffusion problem in two dimentions in stationary state:

\frac{{\partial }}{{\partial x}}( \rho u  T) +\frac{{\partial }}{{\partial y}}( \rho u  T)= \frac{{\partial }}{{\partial x}}\left(\Gamma\frac{{\partial  T}}{{\partial x}}\right) + \frac{{\partial }}{{\partial y}}\left(\Gamma\frac{{\partial  T}}{{\partial y}}\right)

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
Dx = .02
Dy = .02
T1 = 100
T2 = 80
T4 = 20
A = 2
alfa = 1
rho = 1
Nxy = 25

#----------------------------------------------------------------------------------------
                           #Velocity functions definition
#----------------------------------------------------------------------------------------
def u(x,y,A,alfa):
    return -A*np.cos(alfa*np.pi*y)*np.sin(alfa*np.pi*x)

def v(x,y,A,alfa):
    return A*np.sin(alfa*np.pi*y)*np.cos(alfa*np.pi*x)

#----------------------------------------------------------------------------------------
                          #Creation of points cloud (knots)
#----------------------------------------------------------------------------------------

mesh = RegularMesh2D(Nxy,Nxy,1,2)
mesh.create2Dmesh()

#----------------------------------------------------------------------------------------
                           #Velocity vectors
#----------------------------------------------------------------------------------------
Ax = mesh.Ax()
Ay = mesh.Ay()
Ux = np.zeros(mesh.N())
Uy = np.zeros(mesh.N())

for i in range(mesh.N()):
    Ux[i] = u(Ax[i],Ay[i],A,alfa)
    Uy[i] = v(Ax[i],Ay[i],A,alfa)

#----------------------------------------------------------------------------------------
                               #Kernel selection
#----------------------------------------------------------------------------------------

kernel = Multiquadric2D(1/np.sqrt(mesh.N()))

#----------------------------------------------------------------------------------------
                             #Gramm matrx allocation
#----------------------------------------------------------------------------------------

matrix = GrammMatrix(mesh)
matrix.fillMatrixAdvDiff2D(kernel,Dx,Dy,Ux,Uy,rho)

#----------------------------------------------------------------------------------------
                          #Dirichlet boundary condition 
#----------------------------------------------------------------------------------------

#matrix.setDirichletRegular(T1,1)
matrix.setDirichletRegular(T2,2)
matrix.setDirichletRegular(T4,4)
fb= matrix.getfv()
#matrix.setFv(100,17,18)


#----------------------------------------------------------------------------------------
                               #atrix content print
#----------------------------------------------------------------------------------------

plt.figure(figsize=(10,10))
plt.matshow(matrix.getMatrix(), cmap=plt.cm.Blues, fignum=1)
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
plotl.levelplot(title = 'linalg.solve()', xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_l.png')
#plotj = plotter(solvj,kernel)
#plotj.levelplot(title = 'Jacobi - error : %g, iteraciones : %d' % (errj, kj), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_j.png')
#plotgs = plotter(solvgs,kernel)
#plotgs.levelplot(title = title+'Gauss-Seidel - error : %g, iteraciones : %d' % (errgs, kgs), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_gs.png')
#plotsor = plotter(solvsor,kernel)
#plotsor.levelplot(title = title+'SOR - error : %g, iteraciones : %d' % (errsor, ksol), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_sor.png')
plotgcm = plotter(solvgcm,kernel)
plotgcm.levelplot(title = 'CGM - error : %g, iteraciones : %d' % (errgcm, kgcm), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_gcm.png')
plotbi = plotter(solvbi,kernel)
plotbi.levelplot(title = 'BICGSTAB - error : %g, iteraciones : %d' % (errbi, kbi), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_bi.png')
plotgmr = plotter(solvgmr,kernel)
plotgmr.levelplot(title = 'GMRES - error : %g, iteraciones : %d' % (errgmr, kgmr), xlabel = xlabel, ylabel = ylabel, barlabel = barlabel,fileName = 'StationaryAD_gmr.png')

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