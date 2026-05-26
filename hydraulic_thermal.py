import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

from data_structures import GeraGrafo, CreateMapDistance
from hydraulics import Hydraulics_T
from thermal import Thermal_H
from plotting import PlotaPlaca, PlotaRede, plot_hydraulic_thermal_profiles



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
