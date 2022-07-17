"""
This module contains tools for examining properties of ferromagnetic systems
using utilities from model.py module.

Functions:
    variance_with_t
    coarse_graining
    variance_with_h
"""

import numpy as np

from model import Ising2D, boltzmann, cogr, k,J

def variance_with_t(start, stop, num, N_iter=1001, equil=100, N=50, init_state=None, h=0):
    """
    Classic Metropolis Implementation, describes variance of parameters of Ising Model with
    temperature.

    Parameters
    ----------
    start, stop, num: float
        range of temperature points to use
    N_iter, equil: int (optional)
        number of iterations for Metropolis algorithm and for equilibration respectively
    N: int, init_state: np.ndarray, h: float (optional)
        parameters for Ising2D class initialisation
    
    Returns
    -------
    M0, E0, X0, C0: tuple
        average magnetisation, average energy,
        susceptibility and heat capacity of the Ising Model with respect to T

    See also
    --------
    variance with h
    """

    E0 = np.ones(num)
    M0 = np.ones(num)

    for i in range(5):
        E = np.array([],dtype=float)
        M = np.array([],dtype=float)

        for k in np.linspace(start, stop, num):
            np.random.seed(i)
            lat = Ising2D(N, round(k,2), init_state=init_state, h=h)
            M1 = np.array([0],dtype=float)
            E1 = np.array([0],dtype=float)

            for _ in range(equil):
                lat.step()
            for j in range(1, N_iter):
                lat.step()
                if j%100==0:
                    M1 += np.mean(lat.state)
                    E1 += lat.get_total_energy()
            E = np.append(E,E1/(N*N*N_iter//100))
            M = np.append(M,M1)
        E0 = np.vstack((E0,E))
        M0 = np.vstack((M0,M))
    M0 = M0[1:]
    E0 = E0[1:]
    X0 = np.var(M0,axis=0)
    C0 = np.var(E0,axis=0)

    return (M0,E0,X0,C0)

def variance_with_h(start, stop, num, N, T, init_state=np.ones((50,50)), equil=100, N_iter=1001):
    """
    Function for studying variance of properties of Ising Model with the strength
    of external magnetic field with constant temperature.
    Returns average magnetisation with respect to h.

    Parameters
    ----------
    start, stop, num: float
        range of magnetic field points to use
    N_iter, equil: int (optional)
        number of iterations for Metropolis algorithm and for equilibration respectively
    N: int, init_state: np.ndarray (optional)
        parameters for Ising2D class initialisation
    T: float
        temperature of the system
    
    Returns
    -------
    M0: np.ndarray
        average magnetisation with respect to h

    See also
    --------
    variance_with_t
    """

    M0 = np.array([],dtype=float)
    for h in np.linspace(start,stop,num):
        M=np.array([],dtype=float)
        lat = Ising2D(N=N,T=T,init_state=init_state,h=h)
        for _ in range(equil):
            lat.step()
        for j in range(N_iter):
            lat.step()
            if j%100==0:
                M = np.append(M,np.mean(lat.state))
        M0 = np.append(M0,np.mean(M))    
    return M0
  
def coarse_graining(t, N, N_iter, num, size):
    """
    Coarse graining algorithm performed on Ising2D class instance for
    a particular temperature value.

    Parameters
    ----------
    t: float
        temperature of the system
    N_iter: int
        number of iterations for Metropolis algorithm
    N: int
        parameters for Ising2D class initialisation
    num: int
        number of coarse-graining iterations
    size: int
        size of single cells
    
    Returns
    -------
    states: list
        state of Ising Model after N_iter iterations of Metropolis algorithm and
        an array of coarse-grained states
    """
 
    lat = Ising2D(N,T=t)

    for j in range(1,N_iter): 
        lat.step()
    a = np.copy(lat.state)
    states = [a]

    for i in np.arange(num):
        a = cogr(a, size)
        states.append(a)

    return states
   
