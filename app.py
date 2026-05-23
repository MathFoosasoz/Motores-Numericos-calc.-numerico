import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra
from mechanic import Mechanic, Mechanic_P2, Mechanic_P4

from themal_hydraulics import HydraulicThermal
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_ht = env.CONFIG_HT

    test = HydraulicThermal(config_ht)
    test.run_interpolator(plot=True)
    
    return


if __name__ == "__main__":
    main()
