import numpy as np
from plotting import PlotaPlaca, PlotaEixoTemps, plot_problem4, plot_p1_extra_subdivisions, plot_p1_extra_tolerance, plot_p1_complex_analysis
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

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        #dicionario dos indices das bordas e seus valores pra faciltar
        boundary={}

        for i in kx:
            boundary[self.ij2n(self.N[1]-1,i)] = TT(i, self.N)
            boundary[self.ij2n(0,i)] = TB(i, self.N)

        for i in ky:
            boundary[self.ij2n(i,self.N[0]-1)] = TR
            boundary[self.ij2n(i,0)] = TL

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
        nunk = self.N[0] * self.N[1]
        Nx, Ny = self.N
        h2 = self.find_square_area()
        
        TR, TT, TL, TB = self.temps

        rows, cols, data = [], [], []
        b = np.zeros(nunk)

        # i (linhas) corresponde ao eixo Y. j (colunas) corresponde ao eixo X.
        
        for i in range(Ny): 
            for j in range(Nx): 
                Ic = self.ij2n(i, j)

                if j == Nx - 1:     
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TR
                elif j == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TL
                elif i == Ny - 1:   
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TT(j, self.N)
                elif i == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TB(j, self.N)

                else:
                    # Mapeamento dos vizinhos:
                    Ie = self.ij2n(i, j+1)  # Leste (+x)
                    Iw = self.ij2n(i, j-1)  # Oeste (-x)
                    In = self.ij2n(i+1, j)  # Norte (+y)
                    Is = self.ij2n(i-1, j)  # Sul (-y)


                    # Preenche os coeficientes na matriz esparsa
                    rows.append(Ic); cols.append(Ic); data.append(4.0)
                    rows.append(Ic); cols.append(Ie); data.append(-1.0)
                    rows.append(Ic); cols.append(Iw); data.append(-1.0)
                    rows.append(Ic); cols.append(In); data.append(-1.0)
                    rows.append(Ic); cols.append(Is); data.append(-1.0)

                    # O lado direito agora é apenas f * h^2 (pois o k já está na matriz A)
                    b[Ic] = (h2 * self.f)/self.K

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        return spsolve(A_sparse, b)

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
            print(f"Condutividade térmica utilizada: {self.K} W/mK")
            print(f"Fonte térmica utilizada: {self.f:.1e} W/m³")

            """print(f"solução das temperaturas do sistema:")
            if self.N[0]>21:
                a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
            
            else:
                a = "y"
            if a in "yY":
                self.print_temp(temps)
            """


        if plot:
            PlotaPlaca(*self.N, *self.L, temps)
            
class Thermal_P1(Thermal):
    def __init__(self, config, method="cholesky"):
        super().__init__(config, method)
        self.MULTI_N = config["MULTI_N"]

    def complexity_analysis(self, print_info=True, plot=True):
        resultados = []

        for n in self.MULTI_N:
            self.N = n
            nunk = n[0] * n[1]

            # 1. Medir o tempo do método Esparso
            self.method = "sparse"
            t_start = time.time()
            self.solve_system_sparse()
            t_sparse = time.time() - t_start

            # 2. Medir o tempo do método Denso (Cholesky)
            # Trava de segurança: se for maior que 15.000 nós
            if nunk < 15000:
                self.method = "cholesky"
                t_start = time.time()
                self.solve_system_cholesky()
                t_dense = time.time() - t_start
            else:
                t_dense = None

            # Guardar os resultados
            resultados.append({
                "Malha": f"{n[0]}x{n[1]}",
                "Nós": nunk,
                "Esparsa (s)": t_sparse,
                "Densa (s)": t_dense
            })

            if print_info:
                print(f"Malha {n} analisada. Esparsa: {t_sparse:.4f}s | Densa: {'Pulada' if t_dense is None else f'{t_dense:.4f}s'}")

        if plot:
            # GERAR TABELA
            colunas = ["Malha", "Nós", "Esparsa (s)", "Densa (s)"]

            dados = []
            for r in resultados:
                tempo_denso = f"{r['Densa (s)']:.6f}" if r['Densa (s)'] is not None else "Pulada"
                
                dados.append([
                    r["Malha"],
                    r["Nós"],
                    f"{r['Esparsa (s)']:.6f}",
                    tempo_denso
                ])

            plot_p1_complex_analysis(dados, colunas)                       
    
    def run(self, print_info=True, plot=True):
        resultados_tabela = []

        for n in self.MULTI_N:
            self.N = n
            Nx = n[0]
            nunk = n[0] * n[1]
        
            # Medição Esparsa (Executada para todas as malhas)
            t_0 = time.time()
            temps_final = self.solve_system_sparse()
            # PLOTAGEM DENTRO DO LAÇO: Gera gráficos para cada discretização
            if plot:
                # 1. Mapa de Calor (Curvas de Nível)
                PlotaPlaca(*self.N, *self.L, temps_final, Tmax=True)
                
                # 2. Temperatura no Eixo Central
                kx = np.arange(self.N[0])
                y_idx = (self.N[1] - 1) // 2
                indices_centrais = [self.ij2n(y_idx, i) for i in kx]
                temps_centrais = temps_final[indices_centrais]
                PlotaEixoTemps(self.N, self.L[0], temps_centrais)

class Thermal_P2(Thermal):

    def __init__(self, config, method = "cholesky"):
        super().__init__(config, method)

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
            boundary[self.ij2n(self.N[1]-1,i)] = TT(i, self.N)
            boundary[self.ij2n(0,i)] = TB(i, self.N)

        for i in ky:
            boundary[self.ij2n(i,self.N[0]-1)] = TR
            boundary[self.ij2n(i,0)] = TL


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

        Nx, Ny = self.N
        h2 = self.find_square_area()
        
        TR, TT, TL, TB = self.temps

        rows, cols, data = [], [], []
        b = np.zeros(nunk)
        self.get_index_inside_circus()

        for i in range(Ny): 
            for j in range(Nx): 
                Ic = self.ij2n(i, j)

                if j == Nx - 1:     
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TR
                elif j == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TL
                elif i == Ny - 1:   
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TT(j, self.N)
                elif i == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TB(j, self.N)

                elif Ic in self.index_inside_circus:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = self.circle_temp

                else:
                    # Mapeamento dos vizinhos:
                    Ie = self.ij2n(i, j+1)  # Leste (+x)
                    Iw = self.ij2n(i, j-1)  # Oeste (-x)
                    In = self.ij2n(i+1, j)  # Norte (+y)
                    Is = self.ij2n(i-1, j)  # Sul (-y)


                    # Preenche os coeficientes na matriz esparsa
                    rows.append(Ic); cols.append(Ic); data.append(4.0)
                    rows.append(Ic); cols.append(Ie); data.append(-1.0)
                    rows.append(Ic); cols.append(Iw); data.append(-1.0)
                    rows.append(Ic); cols.append(In); data.append(-1.0)
                    rows.append(Ic); cols.append(Is); data.append(-1.0)

                    b[Ic] = (h2 * self.f)/self.K

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        return spsolve(A_sparse, b)

    
    def run(self, print_info = False, plot = False):

        preserve_self_N = self.N

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Resolvido por: {self.method}")
            print(f"Raio do círculo: {self.circle_radius} m")
            print(f"Temperatura no círculo: {self.circle_temp} °C")

        for n in preserve_self_N:
            self.N = n

            if self.method == "cholesky":
                temps = self.solve_system_cholesky()

            elif self.method == "sparse":
                temps = self.solve_system_sparse()

            else:
                raise ValueError()

                """
                print(f"Solução das temperaturas do sistema:")
                
                if self.N[0]>21:
                    a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
                
                else:
                    a = "y"
                
                if a in "yY":
                    self.print_temp(temps)
                """

            if plot:
                PlotaPlaca(*self.N, *self.L, temps, Tmax= True)

                kx = np.arange(self.N[0])
                y = (self.N[1]-1)//2

                Ic = self.ij2n(y, kx)
                temps_centrais = temps[Ic]

                PlotaEixoTemps(self.N, self.L[0], temps_centrais)

class Thermal_P3(Thermal):
    def __init__(self, config, method="sparse"):
        super().__init__(config, method)
        self.MULTI_N = config["MULTI_N"]

    def k_cond(self, x, y):
        #Função de condutividade variável k(x,y)
        #k(x,y) = 0.2 + 0.05 * sin(3*pi*x/Lx) * sin(3*pi*y/Ly)
        Lx, Ly = self.L
        return 0.2 + 0.05 * math.sin(3 * math.pi * x / Lx) * math.sin(3 * math.pi * y / Ly)

    def solve_system_sparse(self):
        # Montagem esparsa com condutividade variável.
        nunk = self.N[0] * self.N[1]
        Nx, Ny = self.N
        Lx, Ly = self.L
        
        # O problema pressupõe malha quadrada h = dx = dy.
        # Para N = (101, 51) e L = (0.02, 0.01), dx = dy = 0.0002
        h = Lx / (Nx - 1) 
        h2 = h ** 2
        
        TR, TT, TL, TB = self.temps

        rows, cols, data = [], [], []
        b = np.zeros(nunk)

        # i (linhas) corresponde ao eixo Y. j (colunas) corresponde ao eixo X.
        for i in range(Ny): 
            for j in range(Nx): 
                Ic = self.ij2n(i, j)

                # Coordenadas físicas do nó central
                x = j * h
                y = i * h

                if j == Nx - 1:     
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TR
                elif j == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TL
                elif i == Ny - 1:   
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TT(j, self.N)
                elif i == 0:               
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TB(j, self.N)

                else:
                    # Mapeamento dos vizinhos:
                    Ie = self.ij2n(i, j+1)  # Leste (+x)
                    Iw = self.ij2n(i, j-1)  # Oeste (-x)
                    In = self.ij2n(i+1, j)  # Norte (+y)
                    Is = self.ij2n(i-1, j)  # Sul (-y)

                    # Calcula a condutividade k nas faces (pontos médios)
                    ke = self.k_cond(x + h/2, y)
                    kw = self.k_cond(x - h/2, y)
                    kn = self.k_cond(x, y + h/2)
                    ks = self.k_cond(x, y - h/2)

                    # Preenche os coeficientes na matriz esparsa
                    kc = ke + kw + kn + ks
                    rows.append(Ic); cols.append(Ic); data.append(kc)
                    rows.append(Ic); cols.append(Ie); data.append(-ke)
                    rows.append(Ic); cols.append(Iw); data.append(-kw)
                    rows.append(Ic); cols.append(In); data.append(-kn)
                    rows.append(Ic); cols.append(Is); data.append(-ks)

                    # O lado direito agora é apenas f * h^2 (pois o k já está na matriz A)
                    b[Ic] = h2 * self.f

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        return spsolve(A_sparse, b)

    def run(self, print_info=False, plot=True):
        for n in self.MULTI_N:
            self.N = n
            temps = self.solve_system_sparse()
            
            if print_info:
                print(f"Malha {self.N[0]:>3}x{self.N[1]:<3} resolvida. T_max = {np.max(temps):.4f}°C")
                
            if plot:
                PlotaPlaca(*self.N, *self.L, temps, Tmax=True)
                
                # Plot do eixo central
                kx = np.arange(self.N[0])
                y_idx = (self.N[1] - 1) // 2
                indices_centrais = [self.ij2n(y_idx, i) for i in kx]
                temps_centrais = temps[indices_centrais]
                
                PlotaEixoTemps(self.N, self.L[0], temps_centrais)

class Thermal_P4(Thermal_P2):

    def __init__(self, config, method="sparse"):
        super().__init__(config, method)
        self.config = config
        self.N = config["N"]

    def definir_reta(self):

        original_TC = self.circle_temp

        self.circle_temp = 0
        T0 = self.solve_system_sparse() if self.method == "sparse" else self.solve_system_cholesky()

        self.circle_temp = 1
        T1 = self.solve_system_sparse() if self.method == "sparse" else self.solve_system_cholesky()

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

    def run(self, print_info=False, plot=False):

        TC, T_max, T_mean = self.definir_reta()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Resolvido por: {self.method}")
            print(f"Número de discretizações: {self.N}")

        if plot:
            plot_problem4(TC, T_max, T_mean, filename="p4.png")

class Thermal_P5(Thermal_P2):

    def __init__(self, config, method="sparse"):
        super().__init__(config, method)
        self.config = config
        self.k_node = config["K_NODE"]
        self.N = config["N"]

    def solve(self):


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
    
class Thermal_P1_extra(Thermal):
    
    def __init__(self, config, method="jacobi"):
        super().__init__(config, method=method)
        
        self.method = method
        self.tol = config.get("TOLERANCE", 1e-5)
        self.max_iter = config.get("MAX_ITERATIONS", 15000)
        self.config = config
        
    def solve_system_iterative(self):
        
     # ESSA PARTE DE APLICAR AS CONDIÇÕES INICIAIS FOI COPIADA DO solve_system_cholesky
     
        nunk = self.N[0]*self.N[1]

        A, b = self.assembly()

        TR, TT, TL, TB = self.temps

        Atilde = A.copy()
        b = b.copy()

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])

        #dicionario dos indices das bordas e seus valores pra facilitar
        boundary={}

        for i in kx:
            boundary[self.ij2n(self.N[1]-1,i)] = TT(i, self.N)
            boundary[self.ij2n(0,i)] = TB(i, self.N)

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
            
     # FIM DA PARTE COPIADA
            
        if self.method == "jacobi":
            return self._jacobi(Atilde, b, self.tol, self.max_iter)
        elif self.method == "gauss_seidel":
            return self._gauss_seidel(Atilde, b, self.tol, self.max_iter)
        else:
            raise ValueError()

    def _jacobi(self, A, b, tol, max_iter, callback=None):
        n = len(b)
        x = np.zeros_like(b) # chute inicial vetor nulo
        D = np.diag(A)
        R = A - np.diag(D)
        
        for k in range(max_iter):
            
            x_new = (b - np.dot(R, x)) / D

            if callback: callback(k, x_new)
            
            # testa convergencia
            if np.linalg.norm(x_new - x, np.inf) < tol:
                print(f"Jacobi converge em {k+1} iterações")
                return x_new
            x = x_new
            
        print("Jacobi não converge no limite de iterações")
        return x

    def _gauss_seidel(self, A, b, tol, max_iter, callback=None):
        n = len(b)
        x = np.zeros_like(b) # chute inicial vetor nulo
        
        for k in range(max_iter):
            x_old = x.copy()
            if callback: callback(k, x)
            for i in range(n):
                s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x[i] = (b[i] - s) / A[i, i]

            
            # testa convergencia
            if np.linalg.norm(x - x_old, np.inf) < tol:
                print(f"Gauss-Seidel converge em {k+1} iterações")
                return x
        
        print("Gauss-Seidel não converge no limite de iterações")
        return x

    def run(self, print_info=False, plot=False, analyze_subdivisions=False, analyze_tolerance=False):

        if analyze_subdivisions:
            self.analyze_time_subdivisions()

        if analyze_tolerance:
            self.analyze_tolerance_subdivisions()
            
        if analyze_subdivisions or analyze_tolerance:
            return
            
        temps = self.solve_system_iterative()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Resolvido por: {self.method}")
            print("solução das temperaturas do sistema:")
            
            if self.N[0] > 21:
                a = input("WARNING! Não é recomendado printar as temperaturas para mais de 20 discretizações horizontais. Ainda quer que imprima? (y/n) ")
            else:
                a = "y"
            
            if a in "yY":
                self.print_temp(temps)

        if plot:
            from plotting import PlotaPlaca 
            PlotaPlaca(*self.N, *self.L, temps)
            
            
    def analyze_time_subdivisions(self):
        multi_n = self.config.get("MULTI_N", [])
        original_n = self.N
        original_method = self.method 
        
        nodes_list = []
        times_j = []
        times_gs = []

        print("Gerando gráfico de tempo vs subdivisões...")

        for n_tuple in multi_n:
            self.N = n_tuple
            nodes_list.append(self.N[0] * self.N[1])
            
            self.method = "jacobi"
            t0 = time.perf_counter()
            self.solve_system_iterative()
            t1 = time.perf_counter()
            times_j.append(t1 - t0)

            self.method = "gauss_seidel"
            t0 = time.perf_counter()
            self.solve_system_iterative()
            t1 = time.perf_counter()
            times_gs.append(t1 - t0)

        self.N = original_n
        self.method = original_method

        plot_p1_extra_subdivisions(nodes_list, times_j, times_gs)


    def analyze_tolerance_subdivisions(self):
        multi_t = self.config.get("MULTI_T", [])
        original_tol = self.tol
        original_method = self.method 
        
        tol_list = []
        times_j = []
        times_gs = []

        print("Gerando gráfico de tempo vs tolerância...")

        for t_val in multi_t:
            self.tol = t_val
            tol_list.append(t_val)
            
            self.method = "jacobi"
            t0 = time.perf_counter()
            self.solve_system_iterative()
            t1 = time.perf_counter()
            times_j.append(t1 - t0)

            self.method = "gauss_seidel"
            t0 = time.perf_counter()
            self.solve_system_iterative()
            t1 = time.perf_counter()
            times_gs.append(t1 - t0)

        self.tol = original_tol
        self.method = original_method

        plot_p1_extra_tolerance(tol_list, times_j, times_gs)

class Thermal_P2_Extra(Thermal_P1_extra):
    def __init__(self, config, method="jacobi"):
        super().__init__(config, method=method)
        self.histories = {"jacobi": [], "gauss_seidel": []}

    def prepare_system(self):
        A, b_orig = self.assembly()
        Atilde = A.copy()
        b = b_orig.copy()
        nunk = self.N[0] * self.N[1]

        kx = np.arange(self.N[0])
        ky = np.arange(self.N[1])
        boundary = {}
        
        #mesma lógica de aplicar as condições de contorno do solve_system_cholesky
        for i in kx:
            boundary[self.ij2n(self.N[1]-1, i)] = self.temps[1](i, self.N) 
            boundary[self.ij2n(0, i)] = self.temps[3](i, self.N)     

        for i in ky:
            boundary[self.ij2n(i, self.N[0]-1)] = self.temps[0]   
            boundary[self.ij2n(i, 0)] = self.temps[2]            

        for index, value in boundary.items():
            for i in range(nunk):
                if i != index:
                    b[i] -= A[i, index] * value
                    Atilde[i, index] = 0
                    Atilde[index, i] = 0
            Atilde[index, index] = 1
            b[index] = value
        return Atilde, b

    def run_comparison_history(self, frame_step=20):
        Atilde, b = self.prepare_system()
        
        self.histories["jacobi"] = []
        self.histories["gauss_seidel"] = []

        #callbacks para armazenar os históricos da iteração de jacobi e gauss-seidel
        def cb_j(k, current_x):
            if k % frame_step == 0:
                self.histories["jacobi"].append(current_x.copy())

        def cb_gs(k, current_x):
            if k % frame_step == 0:
                self.histories["gauss_seidel"].append(current_x.copy())


        print("Rodando Jacobi...")
        res_j = self._jacobi(Atilde, b, self.tol, self.max_iter, callback=cb_j)
        self.histories["jacobi"].append(res_j.copy())

        print("Rodando Gauss-Seidel...")
        res_gs = self._gauss_seidel(Atilde, b, self.tol, self.max_iter, callback=cb_gs)
        self.histories["gauss_seidel"].append(res_gs.copy())

        return res_j, res_gs

    def animate_comparison(self, interval=50):
        import matplotlib.animation as animation

        h_j = self.histories["jacobi"]
        h_gs = self.histories["gauss_seidel"]
        shape = (self.N[1], self.N[0])
        
        v_min = 0
        v_max = max(np.max(h_j[-1]), np.max(h_gs[-1]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.imshow(h_j[0].reshape(shape), cmap='coolwarm', origin='lower', vmin=v_min, vmax=v_max)
        im2 = ax2.imshow(h_gs[0].reshape(shape), cmap='coolwarm', origin='lower', vmin=v_min, vmax=v_max)
        
        ax1.set_title('Evolução: Jacobi')
        ax2.set_title('Evolução: Gauss-Seidel')

        fig.colorbar(im1, ax=ax1, label='Temp (°C)')
        fig.colorbar(im2, ax=ax2, label='Temp (°C)')

        max_frames = max(len(h_j), len(h_gs))

        def update(frame):
            im1.set_array(h_j[min(frame, len(h_j)-1)].reshape(shape))
            im2.set_array(h_gs[min(frame, len(h_gs)-1)].reshape(shape))
            return [im1, im2]

        ani = animation.FuncAnimation(fig, update, frames=max_frames, 
                                      blit=True, repeat=False, interval=interval)
        plt.tight_layout()
        plt.show()
        return ani

class SolucionadorTermicoIterativo(Thermal):
    def __init__(self, config):
        super().__init__(config)

        self.T_alvo = config["T_GOAL"]
        self.nos_circulo = None
        self.multi_n = config["MULTI_N"]

        circle_dict = config["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"]
        self.circle_radius = circle_dict["R"]
        self.circle_coords = circle_dict["coords"]

        self.A_base, self.b_base = self._montar_esparso()

    def _montar_esparso(self):
        Nx, Ny = self.N
        hx = self.L[0] / (Nx - 1)
        hy = self.L[1] / (Ny - 1)
        rx, ry = self.K / hx**2, self.K / hy**2
        diag = 2 * rx + 2 * ry
        TR, TT, TL, TB = self.temps

        linhas, cols, vals = [], [], []
        b = np.zeros(Nx * Ny)

        for j in range(Ny):
            for i in range(Nx):
                idx = self.ij2n(j, i)
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    linhas.append(idx); cols.append(idx); vals.append(1.0)
                    if   i == 0:      b[idx] = TL
                    elif i == Nx - 1: b[idx] = TR
                    elif j == 0:      b[idx] = TB(i, self.N) if callable(TB) else TB
                    else:             b[idx] = TT(i, self.N) if callable(TT) else TT
                else:
                    Ic = idx
                    Ie, Iw = self.ij2n(j, i+1), self.ij2n(j, i-1)
                    In, Is = self.ij2n(j+1, i), self.ij2n(j-1, i)
                    linhas += [Ic] * 5
                    cols   += [Ic, Ie, Iw, In, Is]
                    vals   += [diag, -rx, -rx, -ry, -ry]
                    b[Ic]   = self.f

        self.A_base = sparse.coo_matrix((vals, (linhas, cols)),shape=(Nx*Ny, Nx*Ny)).tocsr() 
        self.b_base = b

        return self.A_base, self.b_base

    def identificar_nos_circulo(self):
        Nx, Ny = self.N
        hx = self.L[0] / (Nx - 1)
        hy = self.L[1] / (Ny - 1)
        R2 = self.circle_radius ** 2

        cx = self.circle_coords[0]*self.L[0]
        cy = self.circle_coords[1]*self.L[1]

        self.nos_circulo = [
            self.ij2n(j, i)
            for j in range(1, Ny - 1)
            for i in range(1, Nx - 1)
            if (i * hx - cx) ** 2 + (j * hy - cy) ** 2 <= R2
        ]

    def _aplicar_condicao_circulo(self, T_C):
        A_lil = self.A_base.tolil()
        b = self.b_base.copy()
        for idx in self.nos_circulo:
            A_lil.rows[idx] = [idx]
            A_lil.data[idx] = [1.0]
            b[idx] = T_C
        return A_lil.tocsr(), b

    def encontrar_Tc(self, beta=1.0, tol=1e-4, max_it=500, T_C_init=None):
        T_C = T_C_init if T_C_init is not None else self.T_alvo
        T_max_final, n_iter, campo = None, 0, None

        for k in range(max_it):
            A_upd, b_upd = self._aplicar_condicao_circulo(T_C)
            campo = spsolve(A_upd, b_upd)
            T_max_final = float(np.max(campo))
            erro = T_max_final - self.T_alvo
            n_iter = k + 1
            if abs(erro) < tol:
                break
            T_C -= beta * erro

        return T_C, T_max_final, n_iter, campo

    def run(self):
        cab = f"{'Nx':>6} {'Ny':>6} {'N_nos':>8} {'t_mont(s)':>11} {'t_resolv(s)':>12} {'Iters':>7} {'T_max(C)':>10}   {'T_C(C)':>10}"
        print("\n" + cab)
        print("-" * len(cab))

        for n in self.multi_n:
            self.N = n

            t0 = time.time()
            self._montar_esparso()
            self.identificar_nos_circulo()
            t_mont = time.time() - t0

            t1 = time.time()
            T_c, T_max, n_iter, _ = self.encontrar_Tc()
            t_resolv = time.time() - t1

            print(f"{n[0]:>6} {n[1]:>6} {n[0]*n[1]:>8} {t_mont:>11.4f} {t_resolv:>12.4f} {n_iter:>7} {T_max:>10.4f}   {T_c:>10.4f}")

