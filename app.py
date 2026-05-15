import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra
from mechanic import Mechanic, Mechanic_P4
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_m = env.CONFIG_M

    test = Mechanic(config_m)
    test.run(print_info=True, plot=False)
    
    test_p4 = Mechanic_P4(config_m)
    
    test_p4.P5_plot_average_elastic_energy()
    
    """
    # ==================== PROBLEMA 4 =========================
    test_p4 = Mechanic_P4(config_m)
    coefs, V_vector, phi, f_naturais = test_p4.compute_modal_projection()
    print("RESULTADO DA PROJEÇÃO MODAL:")
    for i in range (len(coefs)):
        print(f"Modo {i+1} ({f_naturais[i]:.2f} Hz): Coeficiente {i}: {coefs[i]:.6f}")
    
    print("REPRESENTAÇÃO DO TERMO FORÇANTE NA BASE MODAL:")
    for i in range(len(coefs)):
        print(f"  Termo {i+1}: ({coefs[i]:.6f}) * Phi_{i+1} * cos(omega_s * t)")
    
    """
    
    return


if __name__ == "__main__":
    main()
