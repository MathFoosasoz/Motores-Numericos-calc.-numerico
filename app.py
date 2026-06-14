import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra 
from mechanic_hydraulic import MechanicHydraulic, gerar_todos_os_plots, plotar_perfil_membrana_corte, MH_Problema4, MH_Problema5
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_mh = env.CONFIG_MH

    #simulador = MechanicHydraulic(config_mh)

    #print("="*8 + "Iniciando a varredura transiente de todos os cenários" + "="*8)

    #todos_resultados = simulador.resolver_todos_cenarios(print_info=False)

    #print("Simulações finalizadas. Gerando arquivos de plotagem...")

    #gerar_todos_os_plots(todos_resultados)
    #print("Gráficos salvos")
    
    config_p4 = env.CONFIG_MH.copy()
    
    config_p4["N"] = (101, 101) 
    
    solver_p4 = MH_Problema4(config_p4)
    solver_p4.resolver_P4(dt=0.0125, tempo_final=12.0)

    solver_p5 = MH_Problema5(config_p4)
    solver_p5.resolver_P5()
    
    return

if __name__ == "__main__":
    main()
