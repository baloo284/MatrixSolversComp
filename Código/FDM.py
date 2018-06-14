# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:55:25 2018

@author: Ours Noir
"""

import numpy as np
import matplotlib.pyplot as plt

def plotSolution(ut,Nx,Ny,title,xlabel,T1,T2,T3,T4):
    """
    This function plot the solution for a given linear equation system, using contourplot.
    
    @param:
        ut: matrix with the solution.\n
        Nx: number of columns.\n
        Ny: number of rows.\n
        title: label for the plot.\n
        xlabel: label for the horizontal axe.\n
        T1: first boundary value.\n
        T2: second boundary value.\n
        T3: third boundary value.\n
        T4: fourth boundary value.
    """
    # contador is a static attribute that count the calls to this functions
    plotSolution.contador = getattr(plotSolution, 'contador', 0) + 1

    u = np.zeros((Ny+2, Nx+2))

    u[Ny+1,:   ] = T3 
    u[:   ,0   ] = T4
    u[:   ,Nx+1] = T2
    u[0   ,:   ] = T1

    ut.shape = (Ny, Nx)
    u[1:Ny+1,1:Nx+1] = ut

    x = np.linspace(0,1,Nx+2)
    y = np.linspace(0,2,Ny+2)
    xg, yg = np.meshgrid(x,y)

    plt.subplot(2,4,plotSolution.contador)    
    plt.contourf(xg, yg, u, 10, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(xg, yg, u, 10, colors='black')
    plt.clabel(C, inline=1, fontsize=7.5)
    plt.title(title)
    plt.xlabel(xlabel)    
    
def generateSystem(Nx, Ny, diagonal,T1,T2,T3,T4):
    """
    This function fill a matrix through Finie Difference Method.
    
    @param:
        Nx: number of columns.\n
        Ny: number of rows.\n
        diagonal: value of the dominant diagonal.\n
        T1: first boundary value.\n
        T2: second boundary value.\n
        T3: third boundary value.\n
        T4: fourth boundary value.\n
    """
    N = Nx * Ny
    A = np.zeros((N,N))

#Fill tridiagonal blocks
    for j in range(0,Ny):
        ofs = Nx * j
        A[ofs, ofs] = diagonal; 
        A[ofs, ofs + 1] = 1
        for i in range(1,Nx-1):
            A[ofs + i, ofs + i]     = diagonal
            A[ofs + i, ofs + i + 1] = 1
            A[ofs + i, ofs + i - 1] = 1
            A[ofs + Nx - 1, ofs + Nx - 2] = 1; 
            A[ofs + Nx - 1, ofs + Nx - 1] = diagonal 

#Fill the two external diagonals
    for k in range(0,N-Nx):
        A[k, Nx + k] = 1
        A[Nx + k, k] = 1

    f = np.zeros((Ny,Nx)) # RHS
#Apply Dirichlet boundary conditions
    f[0   ,:] -= T1 # Bottom wall    
    f[Ny-1,:] -= T3 # Upper wall
    f[:,0   ] -= T4 # Left wall 
    f[:,Nx-1] -= T2 # Right wall
    f.shape = f.size     # Cambiamos los arreglos a formato unidimensional

    return A, f