import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
import math



class Mechanic():

    def __init__(self, config):
        self.N = config["N"]
        self.R = config["R"]
        self.sigma = config["TENSION"]
        self.rho = config["DENSITY"]
        self.e = config["THICKNESS"]

        self.tol = config["TOLERANCE"]
        self.n_modes = config["N_MODES"]


    def ij2n(self, i, j):
        return j + i*self.N[0]
    
    def find_square_area(self):
        return (( 2/(self.N[0] - 1) ) * ( 2/(self.N[1] - 1) ))
    
    @staticmethod
    def distance(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def get_index_outside_circus(self):

        dx = 2/(self.N[0] - 1)
        dy = 2/(self.N[1] - 1)
 
        index_outside_circus = []
 
        for i in range(self.N[1]):        
            for j in range(self.N[0]):   
                position = [(j * dx) - 1, (i * dy) -1]
                if self.distance(position, (0,0)) >= 1.0:
                    index_outside_circus.append(self.ij2n(i, j))
 
        self.index_outside_circus = np.array(index_outside_circus)
        return self.index_outside_circus
 
    def BuildMatrizesEigen(self):
        N0, N1 = self.N   
        nunk = N0 * N1
 
        h_squared = self.find_square_area()
 
        # Matriz de rigidez K
        d1 = 4.0 * np.ones(nunk)
        d2 = -np.ones(nunk - 1)
        d3 = -np.ones(nunk - N0)
        K = (self.sigma / h_squared) * sparse.diags(
            [d3, d2, d1, d2, d3],
            [-N0, -1, 0, 1, N0],
            format="csr"
        )

        big_number = 1e7
        Iden = big_number * sparse.identity(nunk, format="csr")
 
        for k in range(N0):
            Ic = self.ij2n(0, k)
            K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]
 
            Ic = self.ij2n(N1 - 1, k)
            K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]
 

        for k in range(N1):
            Ic = self.ij2n(k, 0)
            K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]
 
            Ic = self.ij2n(k, N0 - 1)
            K[Ic, :], K[:, Ic] = Iden[Ic, :], Iden[:, Ic]
 
        # Máscara circular
        index_out_of_circus = self.get_index_outside_circus()
        K[index_out_of_circus, :] = Iden[index_out_of_circus, :]
        K[:, index_out_of_circus] = Iden[:, index_out_of_circus]
 
        M = sparse.identity(nunk, format="csr")
 
        return K, M
    
    def SolveEigenWithoutForce(self):
        K, M = self.BuildMatrizesEigen()
 
        lam, modes = eigsh(K, k=self.n_modes, M=M, which="SM", tol=self.tol)
 
        ordered = np.argsort(lam)
        lam   = lam[ordered]
        modes = modes[:, ordered]
 
        freq_scale = np.sqrt(self.sigma/(self.rho * self.e * self.R * self.R))

        omega    = freq_scale*np.sqrt(np.abs(lam))
        freq = omega / (2.0 * np.pi)

 
        return freq, omega, modes

