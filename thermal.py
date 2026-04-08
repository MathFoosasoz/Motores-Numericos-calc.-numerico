import numpy as np
from shapely import boundary
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

        #dicionario dos indices das bordas e seus valores pra faciltar
        k = np.arange(self.N)
        boundary={}

        for i in k:
            boundary[self.ij2n(i,self.N-1)] = TR
            boundary[self.ij2n(self.N-1,i)] = TT
            boundary[self.ij2n(0,i)] = TL
            boundary[self.ij2n(i,0)] = TB   

        #NOTA IMPORTANTE PARA O FUTURO: os quatro cantos da borda são "disputados" (por ex, o ponto (0,0) é parte da borda esquerda 
        # e da borda inferior) não faço ideia de como isso pode afetar os resultados, mas é algo pra gente ter atenção

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


