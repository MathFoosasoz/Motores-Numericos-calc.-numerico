import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

from data_structures import GeraGrafo, CreateMapDistance
from hydraulics import Hydraulics_T
from thermal import Thermal_H
from plotting import PlotaPlaca, PlotaRede, plot_hydraulic_thermal_profiles, plota_perfis_vertical_horizontal
from themal_hydraulics import HydraulicThermal

class Hydraulic_to_Thermal():
    def __init__(self, config):
        self.config = config

        self.multi_n = config["MULTI_N"]
        self.r_cond_values = config["R_COND_VALUES"]

        levels = config["LEVELS"]
        Xno, conec = GeraGrafo(levels)

        Xno = Xno * 0.001
        Xno[:, 1] = Xno[:, 1] + config["L"][1] / 2

        self.hydraulics = Hydraulics_T(conec, Xno, config)
        self.thermal = Thermal_H(config)
        self.results = {}

        self._prepare_circle_center()

    def ij2n(self, i, j):
        return self.thermal.ij2n(i, j)

    def grid_spacing(self):
        Nx, Ny = self.thermal.N
        dx = self.thermal.L[0] / (Nx - 1)
        dy = self.thermal.L[1] / (Ny - 1)
        return dx, dy

    def _prepare_circle_center(self):
        # Thermal_H guarda as coordenadas originais; aqui convertemos para coordenadas fisicas.
        coords = np.array(self.thermal.circle_coords, dtype=float)

        if np.all(coords <= 1.0):
            self.circle_center = coords * np.array(self.thermal.L)
        elif np.all(coords <= self.thermal.L):
            self.circle_center = coords
        else:
            self.circle_center = coords * 0.01

    def _circle_mask(self):
        # Mascara dos nos onde a temperatura e imposta pela inclusao circular.
        Nx, Ny = self.thermal.N
        x = np.linspace(0.0, self.thermal.L[0], Nx)
        y = np.linspace(0.0, self.thermal.L[1], Ny)
        X, Y = np.meshgrid(x, y)

        dist = np.sqrt((X - self.circle_center[0]) ** 2 + (Y - self.circle_center[1]) ** 2)
        return (dist <= self.thermal.circle_radius).ravel()

    def _modified_conductivity_from_map(self, mapa_proximidade, node_a, node_b, r_cond):
        edge_distances = {}

        for edge_id, dist in mapa_proximidade[node_a] + mapa_proximidade[node_b]:
            if edge_id not in edge_distances or dist < edge_distances[edge_id]:
                edge_distances[edge_id] = dist

        if len(edge_distances) == 0:
            return self.thermal.K

        distances = np.array(list(edge_distances.values()))
        return self.thermal.K * (1 + np.sum(1 / (1 + distances / r_cond)))

    def calculate_interface_conductivities(self, r_cond):
        Nx, Ny = self.thermal.N

        # Mapa de proximidade fornecido pelo professor:
        # para cada no da malha, lista as arestas hidraulicas dentro do raio de corte.
        mapa_proximidade = CreateMapDistance(
            self.thermal.L[0],
            self.thermal.L[1],
            Nx,
            Ny,
            self.hydraulics.Xno,
            self.hydraulics.conec,
            r_cond
        )

        kx_faces = np.full((Ny, Nx - 1), self.thermal.K, dtype=float)
        ky_faces = np.full((Ny - 1, Nx), self.thermal.K, dtype=float)

        for i in range(Ny):
            for j in range(Nx - 1):
                node_a = self.ij2n(i, j)
                node_b = self.ij2n(i, j + 1)
                kx_faces[i, j] = self._modified_conductivity_from_map(mapa_proximidade, node_a, node_b, r_cond)

        for i in range(Ny - 1):
            for j in range(Nx):
                node_a = self.ij2n(i, j)
                node_b = self.ij2n(i + 1, j)
                ky_faces[i, j] = self._modified_conductivity_from_map(mapa_proximidade, node_a, node_b, r_cond)

        return kx_faces, ky_faces

    def solve_system_sparse(self, r_cond):
        """Resolve a placa usando os k modificados pela proximidade da rede hidraulica."""
        total_start = time.time()

        # Primeiro calcula os k modificados nas interfaces da malha.
        t0 = time.time()
        kx_faces, ky_faces = self.calculate_interface_conductivities(r_cond)
        time_k = time.time() - t0

        t0 = time.time()
        nunk = self.thermal.N[0] * self.thermal.N[1]
        Nx, Ny = self.thermal.N
        dx, dy = self.grid_spacing()

        TR, TT, TL, TB = self.thermal.temps
        circle_mask = self._circle_mask()

        rows, cols, data = [], [], []
        b = np.zeros(nunk)

        for i in range(Ny):
            for j in range(Nx):
                Ic = self.ij2n(i, j)

                if j == Nx - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TR
                elif j == 0:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TL
                elif i == Ny - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TT(j, self.thermal.N)
                elif i == 0:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TB(j, self.thermal.N)
                elif circle_mask[Ic]:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = self.thermal.circle_temp
                else:
                    Ie = self.ij2n(i, j + 1)
                    Iw = self.ij2n(i, j - 1)
                    In = self.ij2n(i + 1, j)
                    Is = self.ij2n(i - 1, j)

                    ke = kx_faces[i, j]
                    kw = kx_faces[i, j - 1]
                    kn = ky_faces[i, j]
                    ks = ky_faces[i - 1, j]

                    ce = ke * dy / dx
                    cw = kw * dy / dx
                    cn = kn * dx / dy
                    cs = ks * dx / dy

                    rows.append(Ic); cols.append(Ic); data.append(ce + cw + cn + cs)
                    rows.append(Ic); cols.append(Ie); data.append(-ce)
                    rows.append(Ic); cols.append(Iw); data.append(-cw)
                    rows.append(Ic); cols.append(In); data.append(-cn)
                    rows.append(Ic); cols.append(Is); data.append(-cs)

                    b[Ic] = self.thermal.f * dx * dy

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        time_assembly = time.time() - t0

        t0 = time.time()
        temps = spsolve(A_sparse, b)
        time_solve = time.time() - t0

        times = {
            "k_interfaces": time_k,
            "assembly": time_assembly,
            "solve": time_solve,
            "total": time.time() - total_start,
        }

        return temps, kx_faces, ky_faces, times

    def solve_case(self, n, r_cond):
        self.thermal.N = tuple(n)
        temps, kx_faces, ky_faces, times = self.solve_system_sparse(r_cond)

        result = {
            "N": self.thermal.N,
            "r_cond": r_cond,
            "temperature": temps,
            "kx_faces": kx_faces,
            "ky_faces": ky_faces,
            "T_max": np.max(temps),
            "times": times,
        }

        self.results[(self.thermal.N, r_cond)] = result
        return result

    def print_results_table(self, results):
        print("\nRESULTADOS - PROBLEMA 1 / PARTE 2\n")
        print(
            f"{'Nx':>6} {'Ny':>6} {'r_cond (m)':>12} {'Tmax (C)':>12} "
            f"{'k_faces (s)':>12} {'assembly (s)':>12} {'solve (s)':>10} {'total (s)':>10}"
        )
        print("-" * 88)

        for result in results:
            Nx, Ny = result["N"]
            times = result["times"]
            print(
                f"{Nx:>6} {Ny:>6} {result['r_cond']:>12.5g} {result['T_max']:>12.4f} "
                f"{times['k_interfaces']:>12.4f} {times['assembly']:>12.4f} "
                f"{times['solve']:>10.4f} {times['total']:>10.4f}"
            )

    def run_problem1(self, print_info=True, plot=True):
        results = []

        for n in self.multi_n:
            for r_cond in self.r_cond_values:
                result = self.solve_case(n, r_cond)
                results.append(result)

                if plot:
                    fig, ax = PlotaPlaca(*result["N"], *self.thermal.L, result["temperature"], Tmax=True, show=False)
                    PlotaRede(self.hydraulics.conec, self.hydraulics.Xno, fig=fig, ax=ax, show=False)
                    plot_hydraulic_thermal_profiles(result["N"], self.thermal.L, result["temperature"], result["r_cond"])
                    plt.show()

        if print_info:
            self.print_results_table(results)

        return results

    def run(self, print_info=True, plot=True):
        return self.run_problem1(print_info=print_info, plot=plot)


class Hydraulic_to_Thermal_P2(HydraulicThermal):
    
    #Problema 2: Cálculo do termo fonte (esquenta) e sumidouro (resfria) origininado pela rede de microcanais.

    def __init__(self, config):
        
        super().__init__(config)
        self.S0_values = [1e5, -1e5, 5e5, -5e5, 1e6, -1e6]
        self.cases = ["homogenea", "heterogenea"]
        
    def ij2n(self, i, j):
        return self.thermal.ij2n(i, j)

    def grid_spacing(self):
        Nx, Ny = self.thermal.N
        dx = self.thermal.L[0] / (Nx - 1)
        dy = self.thermal.L[1] / (Ny - 1)
        return dx, dy

    def identify_spine(self):
        
        #Identifica as arestas que compõem o canal central (y do meio da placa)
        
        y_center = self.thermal.L[1] / 2.0
        tol = 1e-4
        spine_edges = set()
        for idx, connection in enumerate(self.hydraulics.conec):
            n1, n2 = connection
            y1 = self.hydraulics.Xno[n1][1]
            y2 = self.hydraulics.Xno[n2][1]
            
            if abs(y1 - y_center) < tol and abs(y2 - y_center) < tol:
                spine_edges.add(idx)
        return spine_edges

    def map_source_term(self, S0, case_Ij, d_max):
        
        #Mapeia a influência da rede hidráulica na placa térmica
        #Usa o CreateMapDistance que o professor deu

        Nx, Ny = self.thermal.N
        f_map = np.zeros(Nx * Ny)
        
        # Parâmetro de espalhamento
        sigma = d_max / 2.0
        
        spine_edges = self.identify_spine()
        
        #Usa a função que o professor deu
        mapa_proximidade = CreateMapDistance(
            self.thermal.L[0], self.thermal.L[1], Nx, Ny,
            self.hydraulics.Xno, self.hydraulics.conec, d_max
        )
        
        for node_idx, arestas_proximas in enumerate(mapa_proximidade):
            
            if not arestas_proximas:
                continue
                
            S_p = 0.0
            
            for id_aresta, dist in arestas_proximas:
                
                if case_Ij == "homogenea":
                    Ij = 1.0
                elif case_Ij == "heterogenea":
                    Ij = 100.0 if id_aresta in spine_edges else 0.1
                    
                # formula do professor
                S_p += S0 * Ij * np.exp(-(dist**2) / (2.0 * sigma**2))
                
            f_map[node_idx] = S_p

        return f_map

    def solve_system_p2(self, S0, case_Ij, d_max=0.00025):
        
        #Resolve o sistema da placa usando mesmo conceito dos thermal anteriores
        #não teve como reutilizar diretamente pois os solve do thermal tem limitações dependendo do problema
        
        total_start = time.time()
        nunk = self.thermal.N[0] * self.thermal.N[1]
        Nx, Ny = self.thermal.N
        dx, dy = self.grid_spacing()

        TR, TT, TL, TB = self.thermal.temps
        
        # 1. Gera o mapa espacial
        f_map = self.map_source_term(S0, case_Ij, d_max)
        
        k = self.thermal.K
        ce, cw = k * dy / dx, k * dy / dx
        cn, cs = k * dx / dy, k * dx / dy
        c_central = ce + cw + cn + cs
        area_elemento = dx * dy

        b = f_map * area_elemento
        
        rows, cols, data = [], [], []

        for i in range(Ny):
            for j in range(Nx):
                Ic = self.ij2n(i, j)

                # Condições de Contorno
                if j == Nx - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TR
                elif j == 0:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TL
                elif i == Ny - 1:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TT(j, self.thermal.N)
                elif i == 0:
                    rows.append(Ic); cols.append(Ic); data.append(1.0); b[Ic] = TB(j, self.thermal.N)
                
                else:
                    Ie = self.ij2n(i, j + 1)
                    Iw = self.ij2n(i, j - 1)
                    In = self.ij2n(i + 1, j)
                    Is = self.ij2n(i - 1, j)

                    rows.append(Ic); cols.append(Ic); data.append(c_central)
                    rows.append(Ic); cols.append(Ie); data.append(-ce)
                    rows.append(Ic); cols.append(Iw); data.append(-cw)
                    rows.append(Ic); cols.append(In); data.append(-cn)
                    rows.append(Ic); cols.append(Is); data.append(-cs)

        A_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(nunk, nunk)).tocsr()
        temps = spsolve(A_sparse, b)
        
        return temps, time.time() - total_start

    def run_problem2(self, plot=True):
        print(f"\n{'='*70}")
        print(f"{'RESULTADOS - PROBLEMA 2':^70}")
        print(f"{'='*70}")
        print(f"{'S0':>10} | {'Distribuição':>15} | {'T_max (°C)':>15} | {'T_min (°C)':>15}")
        print(f"-{'-'*69}")

        all_results = []
        for S0 in self.S0_values:
            for case in self.cases:
                temps, exec_time = self.solve_system_p2(S0, case)
                
                T_max = np.max(temps)
                T_min = np.min(temps)
                print(f"{S0:>10.1e} | {case:>15} | {T_max:>15.4f} | {T_min:>15.4f}")
                
                Nx, Ny = self.thermal.N
                
                # Extrai perfil vertical e horizontal
                j_mid = Nx // 2
                perfil_v = [temps[self.ij2n(i, j_mid)] for i in range(Ny)]
                y_coords = np.linspace(0, self.thermal.L[1], Ny)
                x_mid_val = self.thermal.L[0] / 2.0
                
                i_mid = Ny // 2
                perfil_h = [temps[self.ij2n(i_mid, j)] for j in range(Nx)]
                x_coords = np.linspace(0, self.thermal.L[0], Nx)
                y_mid_val = self.thermal.L[1] / 2.0
                
                all_results.append({
                    "S0": S0, "case": case, "temps": temps, 
                    "T_max": T_max, "perfil_v": (y_coords, perfil_v), "perfil_h": (x_coords, perfil_h)
                })
                
                if plot:
                    
                    # mapa de Contorno e Rede Hidráulica Sobreposta
                    
                    fig, ax = PlotaPlaca(*self.thermal.N, *self.thermal.L, temps, Tmax=True, show=False)
                    PlotaRede(self.hydraulics.conec, self.hydraulics.Xno, fig=fig, ax=ax, show=False)
                    ax.set_title(f"Contours of temperature (P2: S0={S0:.0e} - {case})")
                    
                    
                    #corte vertical / horizontal comparação
                    
                    plota_perfis_vertical_horizontal(
                        x_coords, perfil_h, y_mid_val, 
                        y_coords, perfil_v, x_mid_val, 
                        S0, case
                    )
                    
                    plt.show()

        return all_results