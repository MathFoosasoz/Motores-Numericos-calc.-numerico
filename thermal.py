import numpy as np
from plotting import PlotaMaxPressao, PlotaRede, PlotaPlaca
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
import math

class Thermal():

    def __init__(self, config, method = "cholesky"):
        self.N = config["N"]
        self.L = config["L"]
        self.f = config["SOURCE"]
        self.K = config["CONDUCTIVITY"]
        self.h2 = self.find_square_area()

        self.method = method

        self.temps = config["BORDER_TEMPS"]

    def  ij2n(self, i, j):
        return j + i*self.N[0]
    
    def find_square_area(self):
        return (( self.L[0]/(self.N[0] - 1) ) * ( self.L[1]/(self.N[1] - 1) ))

    def assembly(self):
        nunk = self.N[0]*self.N[1]

        A = np.zeros(shape=(nunk,nunk))
        b = np.zeros(shape=(nunk,))

        for i in range(1, self.N[0] - 1):
            for j in range(1, self.N[1] - 1):
                Ic = self.ij2n(j, i)
                Ie = self.ij2n(j+1, i)
                Iw = self.ij2n(j-1, i)
                In = self.ij2n(j, i+1)
                Is = self.ij2n(j, i-1)

                A[Ic,[Ic,Ie,Iw,In,Is]] = 4, -1, -1, -1, -1
                b[Ic] = (self.h2 * self.f / self.K)

        #print(A)
        return A, b
    
    @staticmethod
    def is_symetric(M):
        shape = M.shape

        for i in range(shape[0]):
            for j in range(i):
                if M[i][j] != M[j][i]:
                    print("assymetric!")
                    return

        print("symetric!")

    def solve_system_cholesky(self):
        nunk = self.N[0]*self.N[1]

        A, b = self.assembly()

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        b = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        #dicionario dos indices das bordas e seus valores pra faciltar
        boundary={}

        for i in kx:
            boundary[self.ij2n(self.N[1]-1,i)] = TT(i)
            boundary[self.ij2n(0,i)] = TB(i)

        for i in ky:
            boundary[self.ij2n(i,self.N[0]-1)] = TR
            boundary[self.ij2n(i,0)] = TL

        # PEGUE Atilde AQ (PROBLEMA DIFÍCIL 1)!!! 

        for index, value in boundary.items():
            for i in range(nunk):
                if i != index:
                    b[i] -= A[i, index]*value

                    Atilde[i, index] = 0
                    Atilde[index, i] = 0

            Atilde[index, index] = 1
            b[index] = value

        #aplicação do método cholesky 
        try: 
            L = np.linalg.cholesky(Atilde)
        except np.linalg.LinAlgError:
            print("erro na montagem da matriz (não positiva ou não simétrica)")
            return None
        
        y = np.linalg.solve(L, b)
    
        Temperature = np.linalg.solve(L.T, y)
        return Temperature
    
    def solve_system_sparse(self):
        nunk = self.N[0]*self.N[1]

        A, b = self.assembly()

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        b = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        Iden = np.identity(nunk)

        # PAREDE DIREITA
        Id = self.ij2n(ky, self.N[0] -1,)
        Atilde[Id,:], b[Id] = Iden[Id,:], TR 

        # PAREDE TOPO
        It = self.ij2n(self.N[1] - 1, kx)
        Atilde[It,:], b[It] = Iden[It,:], TT(kx)

        # PAREDE ESQUERDA
        Ie = self.ij2n(ky, 0)
        Atilde[Ie,:], b[Ie] = Iden[Ie,:], TL

        # PAREDE BAIXO
        Ib = self.ij2n(0, kx)
        Atilde[Ib,:], b[Ib] = Iden[Ib,:], TB(kx)

        Atildesp = sparse.csr_matrix(Atilde)
        Temperature = spsolve(Atildesp, b)

        return Temperature


    
    def print_temp(self, temp):
        for i in range(self.N[1]-1, -1, -1):
            for j in range(self.N[0]):
                global_index = self.ij2n(i,j)
                t = self.round_up_to_nearest_00_1(temp[global_index])
                print(f"{t:.2f}  ", end="")

            print('\n')

    @staticmethod 
    def round_up_to_nearest_00_1(n):
        return math.ceil(n * 100) / 100
    
    def run(self, print_info = False, plot = False):

        if self.method == "cholesky":
            temps = self.solve_system_cholesky()

        elif self.method == "sparse":
            temps = self.solve_system_sparse()

        else:
            raise ValueError()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Resolvido por: {self.method}")
            print(f"solução das temperaturas do sistema:")
            self.print_temp(temps)

        if plot:
            PlotaPlaca(*self.N, *self.L, temps)




