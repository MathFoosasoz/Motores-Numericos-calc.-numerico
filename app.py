import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra 
from mechanic_hydraulic import MechanicHydraulic, gerar_todos_os_plots, plotar_perfil_membrana_corte, MH_Problema4, MH_Problema5
from data_structures import GeraGrafo
from analysis import complexity_analysis
from plotting import plot_relaxamento_problema3
from analise_falhas import RandomFail, resolver_vazao_estacionaria, avaliar_convergencia_monte_carlo, varredura_probabilidade_individual
from sensitivity_analysis import P2_3_GD, prob_base_3_GD

def main():
    
    config_base = env.CONFIG_MH

    #P2_PARTE_3
    config_base["TIME_END"] = 0.09 
    config_base["DT"] = 1.5 
    solver_p2 = P2_3_GD(config_base)
    solver_p2.resolver_P2(E_target=7.5, H0=1000.0e-6, Tc=25.0)

    return

if __name__ == "__main__":
    main()


