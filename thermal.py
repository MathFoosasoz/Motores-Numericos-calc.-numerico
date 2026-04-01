import numpy as np
from plotting import PlotaMaxPressao, PlotaRede
import matplotlib.pyplot as plt
import time
import math

class Thermal():

    def __init__(self, config, N):
        self.N = N
        self.k = config["CONDUCTIVITY"]

        self.temperatures = config["BORDER_TEMPS"]

    def  ij2n(self, i, j):
        return j +i*self.N

    def assembly(self):
        nunk = self.N*self.N
        A = np.zeros(shape=(nunk,nunk))

        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                Ic = self.ij2n(j, i)
                Ie = self.ij2n(j+1, i)
                Iw = self.ij2n(j-1, i)
                In = self.ij2n(j, i+1)
                Is = self.ij2n(j, i-1)

                A[Ic,[Ic,Ie,Iw,In,Is]] = 4, -1, -1, -1, -1

        #print(A)
        return A
    
    @staticmethod
    def is_symetric(M):
        shape = M.shape

        for i in range(shape[0]):
            for j in range(i):
                if M[i][j] != M[j][i]:
                    print("assymetric!")
                    return

        print("symetric!")

    def SolveSystem(self):

        A = self.assembly()

        TR, TT, TL, TB = self.temperatures

        Atilde = A.copy()
        nunk = (self.N)**2

        b = np.zeros(nunk)

        k = np.arange(self.N)
        Iden = np.identity(nunk)

        # PAREDE DIREITA
        Id = self.ij2n(k, self.N -1,)
        Atilde[Id,:], b[Id] = Iden[Id,:], TR 

        # PAREDE TOPO
        It = self.ij2n(self.N - 1, k)
        Atilde[It,:], b[It] = Iden[It,:], TT

        # PAREDE ESQUERDA
        Ie = self.ij2n(k, 0)
        Atilde[Ie,:], b[Ie] = Iden[Ie,:], TL

        # PAREDE BAIXO
        Ib = self.ij2n(0, k)
        Atilde[Ib,:], b[Ib] = Iden[Ib,:], TB

        Temperature = np.linalg.solve(Atilde, b)
        return Temperature
    
    def print_temp(self, temp):
        for i in range(self.N-1, -1, -1):
            for j in range(self.N):
                global_index = self.ij2n(i,j)
                t = self.round_up_to_nearest_0_1(temp[global_index])
                print(f"{t} ", end="")

            print('\n')

    @staticmethod 
    def round_up_to_nearest_0_1(n):
        return math.ceil(n * 10) / 10


