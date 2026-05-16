import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from plotting import gera_grafico_energia_M_P5
import matplotlib.pyplot as plt
import time
import math

from plotting import plot_membrane_modes


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
    
    def run(self, print_info=False, plot=False):
        freq, omega, modes = self.SolveEigenWithoutForce()

        if print_info:
            print("Solução da membrana elástica:")
            print(f"N: {self.N[0]} x {self.N[1]}")
            print(f"freq: {freq}\n\n")
            print(f"omega: {omega}\n\n")
            print(f"modes: {modes}\n\n")

        if plot:
            plot_membrane_modes(self.N, self.n_modes, modes, freq)
            plt.show()


class Mechanic_P2(Mechanic):
    def __init__(self, config):
        super().__init__(config)

    
    def plot_convergence_table_image(self, discretizations):
        data = []
        for n_size in discretizations:
            self.N = (n_size, n_size)
            freqs, _, _ = self.SolveEigenWithoutForce()
            
            row = [f"{n_size}x{n_size}"]
            for i in range(10):
                row.append(f"{freqs[i]:.1f}")
            data.append(row)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        columns = ["Grade NxN"] + [f"f{i+1} (Hz)" for i in range(10)]

        table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9) 
        table.scale(1.1, 2.5) 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')

        plt.title(f"Tabela de Convergência (10 Modos) - Membrana Circular (R={self.R*100} cm)", pad=30, fontsize=14)
        
        plt.show()
    
    
    def plot_modes(self, discretizations):
        for n_size in discretizations:
            self.N = (n_size, n_size)
            
            freq, _, modes = self.SolveEigenWithoutForce()

            plot_membrane_modes(self.N, self.n_modes, modes, freq)
            plt.show() 


class Mechanic_P4(Mechanic):

    def __init__(self, config):
        super().__init__(config)

        #gera o vetor V definido pelo problema (*****atemporal*****) (sem o termo cosseno)
    def get_spatial_force_vector(self):
        
        N0, N1 = self.N
        dx = 2 / (N0 - 1)
        dy = 2 / (N1 - 1)
        
        V = np.zeros(N0 * N1)
        
        for i in range(N1):
            for j in range(N0):
                x = (j * dx) - 1
                y = (i * dy) - 1
                
                n = self.ij2n(i, j)

                V[n] = (x - 0.5)**2 + (y - 0.5)**2

        index_out = self.get_index_outside_circus()
        V[index_out] = 0.0
        
        return V
        
    def compute_modal_projection(self):

        freq, omega, modes = self.SolveEigenWithoutForce()

        V = self.get_spatial_force_vector()

        c = modes.T @ V

        return c, V, modes, freq 
    
    def run(self, print_info = True, plot = False):
        c, V, modes, freq = self.compute_modal_projection()

        if print_info:
            print("RESULTADO DA PROJEÇÃO MODAL:")
            for i in range (len(c)):
                print(f"    Modo {i+1} ({freq[i]:.2f} Hz): Coeficiente {i}: {freq[i]:.6f}")
    
            print("\n\nREPRESENTAÇÃO DO TERMO FORÇANTE NA BASE MODAL:")
            for i in range(len(c)):
                print(f"    Termo {i+1}: ({c[i]:.6f}) * Phi_{i+1} * cos(omega_s * t)")

        if plot:
            plot_membrane_modes(self.N, self.n_modes, modes, freq)
            plt.show()
    
    def P5_plot_average_elastic_energy(self):
        
        c, V, modes, freq = self.compute_modal_projection()

        omega_fisico = freq * 2.0 * np.pi 
        freq_scale = np.sqrt(self.sigma / (self.rho * self.e * self.R**2))
        
        lam = (omega_fisico / freq_scale)**2 
        omega_n_adim = np.sqrt(lam / self.sigma)

        gera_grafico_energia_M_P5(omega_n_adim, c, titulo="Energia Média - Problema 5")