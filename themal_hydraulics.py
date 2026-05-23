import numpy as np
from plotting import PlotaMaxPressao, PlotaRede
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from data_structures import GeraGrafo
from hydraulics import Hydraulics_T
from thermal import Thermal_H
from plotting import plot_interpolation, plot_temp_hydraulics
import time


class HydraulicThermal():
    def __init__(self, config):

        levels = config["LEVELS"]
        Xno, conec = GeraGrafo(levels)

        Xno = Xno * 0.001
        Xno[:,1] = Xno[:, 1] + config["L"][1]/2

        self.hydraulics = Hydraulics_T(conec, Xno, config)
        self.thermal = Thermal_H(config)

    def interpolator(self, method):
        temps = self.thermal.solve_system_sparse()

        data = temps.reshape(self.thermal.N[1], self.thermal.N[0]).T
        x = np.linspace(0, self.thermal.L[0], self.thermal.N[0])
        y = np.linspace(0, self.thermal.L[1], self.thermal.N[1])

        self.interp = RegularGridInterpolator((x, y), data, method=method, bounds_error= False, fill_value= None)
    
    def evaluate_coarse(self, method):
        self.interpolator(method)

        Nx_coarse, Ny_coarse = 21, 11

        x_coarse = np.linspace(0, self.thermal.L[0], Nx_coarse)
        y_coarse = np.linspace(0, self.thermal.L[1], Ny_coarse)

        xx, yy = np.meshgrid(x_coarse, y_coarse, indexing='ij')
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        temps_coarse = self.interp(pts).reshape(Nx_coarse, Ny_coarse)

        return x_coarse, y_coarse, temps_coarse
    
    def evaluate_hydraulic_temps(self, method):
        self.interpolator(method)
        return self.interp(self.hydraulics.Xno)

    def run_interpolator(self, plot = False):
        
        N_thin = [241, 121]
        N_thick = [61,31]

        for n in (N_thin, N_thick):
            self.thermal.N = n
            results = {}
            for method in ['nearest', 'linear', 'cubic']:
                x_c, y_c, t_c = self.evaluate_coarse(method)

                results[method] = (x_c, y_c, t_c)

            if plot:
                plot_interpolation(results, n)
                plt.show()

        
        for n in (N_thin, N_thick):
            self.thermal.N = n
            for method in ['nearest', 'linear', 'cubic']:
                temp_nodes = self.evaluate_hydraulic_temps(method)

                if plot:
                    plot_temp_hydraulics(
                        self.hydraulics.conec,
                        self.hydraulics.Xno,
                        temp_nodes,
                        method,
                        n[0], n[1]
                    )

                    plt.show()

        



        