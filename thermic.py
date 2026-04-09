import numpy as np
from plotting import PlotaPlaca
import matplotlib.pyplot as plt
import math

class Thermal():

    def __init__(self, border_temps):
        self.N = (101, 51)
        self.L = (0.02, 0.01)
        self.f = 100000
        self.K = 0.2
        self.h2 = ( self.L[0]/(self.N[0] - 1) ) * ( self.L[1]/(self.N[1] - 1) )

        self.temps = border_temps["BORDER_TEMPS"]

    def ij2n(self, i, j):
        return j + i*self.N[0]
    

    def assembly(self):
        nunk = self.N[0]*self.N[1]

        A = np.zeros(shape=(nunk,nunk))

        for i in range(self.N[1]):
            for j in range(self.N[0]):

                Ic = self.ij2n(i, j)

                if i == 0 or j == 0 or i == (self.N[1]-1) or j == (self.N[0]-1):
                    A[Ic][Ic] == 1
                    continue
                
                Ie = self.ij2n(i+1, j)
                Iw = self.ij2n(i-1, j)
                In = self.ij2n(i, j+1)
                Is = self.ij2n(i, j-1)

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

    def solve_system(self):
        nunk = self.N[0]*self.N[1]

        A= self.assembly()
        b = np.full((nunk,), ((self.h2 * self.f / self.K)))

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        Iden = np.identity(nunk)

        for i in range(self.N[1]):
            for j in range(self.N[0]):
                
                Ic = self.ij2n(i, j)
                
                # PAREDE BASE
                if i == 0:
                    Atilde[Ic, :] = Iden[Ic, :]
                    b[Ic] = TB(j)
                
                # PAREDE TOPO
                elif i == self.N[1]-1:
                    Atilde[Ic, :] = Iden[Ic, :]
                    b[Ic] = TT(j)
                
                # PAREDE ESQUERDA
                elif j == 0:
                    Atilde[Ic, :] = Iden[Ic, :]
                    b[Ic] = TL
                
                # PAREDE DIREITA
                elif j == self.N[0]-1:
                    Atilde[Ic, :] = Iden[Ic, :]
                    b[Ic] = TR
       
        Temperature  = np.linalg.solve(Atilde, b)
    
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
            
        temps = self.solve_system()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Resolvido por: {self.method}")
            print(f"solução das temperaturas do sistema:")
            
            if self.N[0]>21:
                a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
            
            else:
                a = "y"
            
            if a in "yY":
                self.print_temp(temps)


        if plot:
            PlotaPlaca(*self.N, *self.L, temps, filename="img.png")