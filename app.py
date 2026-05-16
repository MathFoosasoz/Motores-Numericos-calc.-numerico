import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra
from mechanic import Mechanic, Mechanic_P2, Mechanic_P4
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_m = env.CONFIG_M

    #test = Mechanic(config_m)
    #test.run(print_info=True, plot=False)


    # PROBLEMA 2
    test_p2 = Mechanic_P2(config_m)
    
    # Gera a tabela
    test_p2.plot_convergence_table_image(config_m["MULTI_N"])
    
    # Plota os modos sequencialmente
    test_p2.plot_modes(config_m["MULTI_N"])

    
    test_p4 = Mechanic_P4(config_m)
    
    test_p4.P5_plot_average_elastic_energy()
    
    """
    # ==================== PROBLEMA 4 =========================
    test_p4 = Mechanic_P4(config_m)
    test_p4.run(print_info = True, plot = False)
    
    """
    
    return


if __name__ == "__main__":
    main()
