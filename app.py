import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra
from mechanic import Mechanic, Mechanic_P2, Mechanic_P4
from mechanic_hydraulic import MechanicHydraulic
from hydraulic_thermal import Hydraulic_to_Thermal
from hydraulic_thermal import Hydraulic_to_Thermal_P2
from hydraulic_thermal import Hydraulic_to_Thermal_P2_circulo
from themal_hydraulics import HydraulicThermal
from data_structures import GeraGrafo
from analysis import complexity_analysis


def main():

    config_mh = env.CONFIG_MH

    test = MechanicHydraulic(config_mh)
    test.run(print_info=True, plot=False)
    
    return


if __name__ == "__main__":
    main()
    
