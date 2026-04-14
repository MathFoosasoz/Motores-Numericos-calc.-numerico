import numpy as np
from plotting import PlotaPlaca, PlotaEixoTemps
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

        self.method = method
        self.temps = config["BORDER_TEMPS"]

    def  ij2n(self, i, j):
        return j + i*self.N[0]
    
    def find_square_area(self):
        return (( self.L[0]/(self.N[0] - 1) ) * ( self.L[1]/(self.N[1] - 1) ))

    def assembly(self):
        nunk = self.N[0]*self.N[1]

        h2 = self.find_square_area()

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
                b[Ic] = (h2 * self.f / self.K)

        #print(A)
        return A, b
    
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
        btilde = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        Iden = np.identity(nunk)

        # PAREDE DIREITA
        Id = self.ij2n(ky, self.N[0] -1,)
        Atilde[Id,:], btilde[Id] = Iden[Id,:], TR 

        # PAREDE TOPO
        It = self.ij2n(self.N[1] - 1, kx)
        Atilde[It,:], btilde[It] = Iden[It,:], TT(kx)

        # PAREDE ESQUERDA
        Ie = self.ij2n(ky, 0)
        Atilde[Ie,:], btilde[Ie] = Iden[Ie,:], TL

        # PAREDE BAIXO
        Ib = self.ij2n(0, kx)
        Atilde[Ib,:], btilde[Ib] = Iden[Ib,:], TB(kx)

        Atildesp = sparse.csr_matrix(Atilde)
        Temperature = spsolve(Atildesp, btilde)

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
            print(f"Número de discretizações: {self.N}")
            print(f"solução das temperaturas do sistema:")
            
            if self.N[0]>21:
                a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
            
            else:
                a = "y"
            
            if a in "yY":
                self.print_temp(temps)


        if plot:
            PlotaPlaca(*self.N, *self.L, temps, filename="img.png")


class Thermal_P2(Thermal):

    def __init__(self, config, method = "cholesky"):
        super().__init__(config)

        self.N = config["MULTI_N"]
        circle_dict = config["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"]
        self.circle_radius = circle_dict["R"]
        self.circle_coords = circle_dict["coords"]
        self.circle_temp = circle_dict["T"]

        self.method = method

    @staticmethod
    def distance(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    def get_index_inside_circus(self):

        dx = self.L[0]/(self.N[0]-1)
        dy = self.L[1]/(self.N[1]-1)

        center_x = self.circle_coords[0]*self.L[0]
        center_y = self.circle_coords[1]*self.L[1]

        center_coords = (center_x, center_y)

        index_inside_circus = []
        position = []

        for i in range(self.N[1]):
            for j in range(self.N[0]):

                position = [j*dx, i*dy]

                if self.distance(position, center_coords)<= self.circle_radius:
                    index_inside_circus.append(self.ij2n(i, j))

        self.index_inside_circus = np.array(index_inside_circus)

        #print(self.index_inside_circus)
        return
    
    def assembly(self):
        nunk = self.N[0]*self.N[1]

        self.get_index_inside_circus()
        h2 = self.find_square_area()

        A = np.zeros(shape=(nunk,nunk))
        b = np.zeros(shape=(nunk,))

        for j in range(1, self.N[0] - 1):
            for i in range(1, self.N[1] - 1):
                
                Ic = self.ij2n(i, j)

                if Ic not in self.index_inside_circus:
                    Ie = self.ij2n(i+1, j)
                    Iw = self.ij2n(i-1, j)
                    In = self.ij2n(i, j+1)
                    Is = self.ij2n(i, j-1)


                    A[Ic,[Ic,Ie,Iw,In,Is]] = 4, -1, -1, -1, -1
                    b[Ic] = (h2 * self.f / self.K)


        #print(A)
        return A, b
    
    def solve_system_cholesky(self):
        nunk = self.N[0]*self.N[1]

        A, b = self.assembly()

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        b_tilde = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        #dicionario dos indices das bordas e seus valores pra faciltar
        boundary={}

        for i in self.index_inside_circus:
            boundary[i] = self.circle_temp

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
                    b_tilde[i] -= A[i, index]*value

                    Atilde[i, index] = 0
                    Atilde[index, i] = 0

            Atilde[index, index] = 1
            b_tilde[index] = value

        #aplicação do método cholesky 
        try: 
            L = np.linalg.cholesky(Atilde)
        except np.linalg.LinAlgError:
            print("erro na montagem da matriz (não positiva ou não simétrica)")
            return None
        
        y = np.linalg.solve(L, b_tilde)
    
        Temperature = np.linalg.solve(L.T, y)
        return Temperature
    
    def solve_system_sparse(self):
        nunk = self.N[0]*self.N[1]

        A, b = self.assembly()

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        btilde = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        Iden = np.identity(nunk)

        # CÍRCULO
        Ic = self.index_inside_circus
        Atilde[Ic,:], btilde[Ic] = Iden[Ic,:], self.circle_temp 

        # PAREDE DIREITA
        Id = self.ij2n(ky, self.N[0] -1,)
        Atilde[Id,:], btilde[Id] = Iden[Id,:], TR 

        # PAREDE TOPO
        It = self.ij2n(self.N[1] - 1, kx)
        Atilde[It,:], btilde[It] = Iden[It,:], TT(kx)

        # PAREDE ESQUERDA
        Ie = self.ij2n(ky, 0)
        Atilde[Ie,:], btilde[Ie] = Iden[Ie,:], TL

        # PAREDE BAIXO
        Ib = self.ij2n(0, kx)
        Atilde[Ib,:], btilde[Ib] = Iden[Ib,:], TB(kx)

        Atildesp = sparse.csr_matrix(Atilde)
        Temperature = spsolve(Atildesp, btilde)

        return Temperature
    
    def run(self, print_info = False, plot = False):

        preserve_self_N = self.N

        for n in preserve_self_N:
            self.N = n

            if self.method == "cholesky":
                temps = self.solve_system_cholesky()

            elif self.method == "sparse":
                temps = self.solve_system_sparse()

            else:
                raise ValueError()

            if print_info:
                print(f"Resultados para classe: {self.__class__.__name__}")
                print(f"Resolvido por: {self.method}")
                print(f"Número de discretizações: {self.N}")
                print(f"Solução das temperaturas do sistema:")
                
                if self.N[0]>21:
                    a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
                
                else:
                    a = "y"
                
                if a in "yY":
                    self.print_temp(temps)


            if plot:
                PlotaPlaca(*self.N, *self.L, temps, Tmax= True)

                kx = np.arange(self.N[0]-1)
                y = (self.N[1]-1)//2

                Ic = self.ij2n(y, kx)
                temps_centrais = temps[Ic]

                PlotaEixoTemps(self.N[0]-1, self.L[0], temps_centrais)


class Thermal_P4(Thermal_P2):

    def __init__(self, config, method="sparse"):
        super().__init__(config, method)
        self.config = config
        self.N = config["N"]

    def definir_reta(self):
        import numpy as np

        original_TC = self.circle_temp

        self.circle_temp = 0
        T0 = self.solve_system_sparse()

        self.circle_temp = 1
        T1 = self.solve_system_sparse()

        self.circle_temp = original_TC

        dT = T1 - T0

        TC_values = np.linspace(0, 60, 200)
        T_max = np.empty_like(TC_values)
        T_mean = np.empty_like(TC_values)

        for idx, TC in enumerate(TC_values):
            T_field = T0 + TC * dT
            T_max[idx] = np.max(T_field)
            T_mean[idx] = np.mean(T_field)

        return TC_values, T_max, T_mean

    def run(self, plot=False):

        TC, T_max, T_mean = self.definir_reta()

        if plot:
            from plotting import plot_problem4
            plot_problem4(TC, T_max, T_mean, filename="p4.png")


class Thermal_P5(Thermal_P2):

    def __init__(self, config, k_node, method="sparse"):
        super().__init__(config, method)
        self.config = config
        self.k_node = k_node
        self.N = config["N"]

    def solve(self):
        import numpy as np

        def make_config(TR, TC):
            cfg = self.config.copy()

            cfg["MULTI_N"] = [self.config["N"]]

            cfg["BORDER_TEMPS"] = [
                TR,
                self.config["BORDER_TEMPS"][1],
                self.config["BORDER_TEMPS"][2],
                self.config["BORDER_TEMPS"][3],
            ]

            cfg["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"] = {
                **self.config["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"],
                "T": TC,
            }

            return cfg

        def solve_single(TR, TC):
            sim = Thermal_P2(make_config(TR, TC), method="sparse")
            sim.N = self.config["N"] 
            return sim.solve_system_sparse()

        pairs = [(30.0, 0.0), (30.0, 30.0), (0.0, 0.0)]

        T_k_vals = []
        for TR, TC in pairs:
            T_field = solve_single(TR, TC)
            T_k_vals.append(T_field[self.k_node])

        M = np.array([
            [pairs[0][0], pairs[0][1], 1.0],
            [pairs[1][0], pairs[1][1], 1.0],
            [pairs[2][0], pairs[2][1], 1.0],
        ])

        a, b, c = np.linalg.solve(M, np.array(T_k_vals))

        return a, b, c

    def run(self, print_info=False):

        a, b, c = self.solve()

        if print_info:
            print("\n--- Problema 5 ---")

            print(f"\nCoeficientes encontrados (nó k={self.k_node})")
            print(f"  a = {a:.8f}   (sensibilidade a T_R)")
            print(f"  b = {b:.8f}   (sensibilidade a T_C)")
            print(f"  c = {c:.8f}   (termo independente)")
            print(f"\nEquação linear\n  T_{self.k_node} = {a:.6f}·T_R  +  {b:.6f}·T_C  +  {c:.6f}")    
    

if __name__ == "__main__":
    from env import CONFIG_T

    p4 = Thermal_P4(CONFIG_T)
    p4.run(plot=True)

    p5 = Thermal_P5(CONFIG_T, k_node=233)
    p5.run(print_info=True)
    