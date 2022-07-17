"""
This module contains core classes and functions of the project.

Classes:
    Ising2D

Functions:
    boltzmann(float, float) - float
    split_square_matrix(np.ndarray, int) - np.ndarray
    cogr(np.ndarray, int) - np.ndarray

Constants:
    k - Boltzmann constant
    J - coupling constant
"""

import numpy as np
import pickle
from scipy.stats import mode

#constants
k=1
J=1

def boltzmann(energy: float, T: float) -> float:
    """Boltzmann distribution"""
    return np.exp(-energy/(T*k))

def split_square_matrix(matrix: np.ndarray, N: int):
    """
    Splits a given L*L matrix into N equal matrices and returns
    an array of these matrices

    Arguments
    ---------
    matrix: numpy.ndarray
    N: int
        size of matrices (N*N) to split matrix into

    Returns
    -------
    numpy.ndarray
        array of matrices
    """
    shape = matrix.shape
    if shape[0] % N != 0: 
        raise ValueError('Unable to split matrix of size: {} into equal parts of size: {}'.format(shape, N))
    elif shape[0] != shape[1]:
        raise ValueError('Incorrect shape: {} for square matrix.'.format(shape))

    split = shape[0]/N
    return np.array([np.vsplit(i,split) for i in np.hsplit(matrix,split)])

def cogr(state: np.ndarray,N: int) -> list:
    """ 
    Perform coarse-graining on a given matrix

    Arguments
    ---------
    state: numpy.ndarray
        matrix to perform coarse-graining on
    N: int
        size of single cells

    Returns
    -------
    numpy.ndarray
        coarse-grained matrix
    """
    matrices = split_square_matrix(state, N)
    return np.squeeze(mode(mode(matrices,axis=2).mode,axis=3).mode)

class Ising2D:
    """
    Implementation of 2D Ising model using Monte Carlo Metropolis method. Represents
    N*N square lattice, where each point takes either +1 or -1 (up or down magnetic spin).

    Attributes
    ----------
    N: int
        size of the matrix (N*N)
    T: float
        temperature of the system
    N_iter: int
        number of Monte Carlo iterations performed on the system
    init_state: list
        N*N initial matrix (default random set of -1 and 1)
    h: float
        strength of the magnetic field applied (default 0)
    J: float
        coupling constant, defines magnetic nature of the system (default 1)

    Methods
    -------
    get_total_energy()
        returns total energy of the system
    spin_energy(pos)
        returns energy of spin at position pos
    reset()
        sets system to initial state
    step()
        performs a single Metropolis step and updates the system
    """

    def __init__(self, N: int, T: float, init_state=None, h=0,J=1):
        """
        Parameters
        ----------
        N: int
            size of the matrix (N*N)
        T: float
            temperature of the system
        init_state: list, optional
            N*N initial matrix (default random set of -1 and 1)
        h: float, optional
            strength of the magnetic field applied (default 0)
        J: float, optional
            coupling constant, defines magnetic nature of the system (default 1)
        """

        self.init_state = np.random.choice([-1,1] ,(N,N)) if init_state is None else init_state
        self.state = np.copy(self.init_state)
        self.T =T
        self.N = N
        self.N_iter = 0
        self.h=h

    def get_total_energy(self) -> float:
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                energy += self.spin_energy((i,j))
        return energy/2

    def reset(self):
        self.state = np.copy(self.init_state)
        self.N_iter = 0

    def spin_energy(self, pos: tuple) -> float:
        """
        Computes total energy of a single spin given its position.

        Boundary conditions: neighbours of edge spins are spins
         on opposing side

        Parameters
        ----------
        pos: tuple
            position of the spin on matrix
        
        Returns
        -------
        float
            energy of a single spin
        """
        n0, m0 = pos
        return J*self.state[pos]* (self.state[(n0+1)%self.N,m0] +
                                    self.state[(n0-1)%self.N,m0] +
                                    self.state[n0, (m0+1)%self.N] +
                                    self.state[n0, (m0-1)%self.N]) + self.state[pos]*self.h

    def step(self):
        """Performs a single Monte Carlo step on matrix and updates it"""
        self.N_iter +=1
        for _ in range(self.N):
            for _ in range(self.N):
                i = np.random.randint(0,self.N)                 #choose random spin
                j = np.random.randint(0,self.N)
                diff = self.spin_energy((i,j)) * 2              #calculate energy difference
                if diff < 0:                                    #update matrix
                    self.state[i,j] *= -1
                elif boltzmann(diff,self.T) > np.random.rand():
                    self.state[i,j] *= -1
