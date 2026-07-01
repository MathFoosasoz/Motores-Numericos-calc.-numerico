import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra 
from mechanic_hydraulic import MechanicHydraulic, gerar_todos_os_plots, plotar_perfil_membrana_corte, MH_Problema4, MH_Problema5
from data_structures import GeraGrafo
from analysis import complexity_analysis
from plotting import plot_relaxamento_problema3
from analise_falhas import RandomFail, resolver_vazao_estacionaria, avaliar_convergencia_monte_carlo, varredura_probabilidade_individual, aprox_dados
from P2_PARTE_3_GD import P2_3_GD, prob_base_3_GD

def main():

    #============================= Monte Carlo ========================
    Xno, conec = GeraGrafo(env.CONFIG_FALHAS["LEVELS"]);
    Xno = Xno * 0.001
    config_base = env.CONFIG_FALHAS

    print("=== CALIBRANDO PONTO DE OPERAÇÃO OPERACIONAL ===")
    config_base["INLET_PRESSURE"] = 1.0e4 
    vazao_teste = resolver_vazao_estacionaria(conec, Xno, config_base, C_estocastico=None)
    
    fator_pressao = 2.0e-5 / vazao_teste
    config_base["INLET_PRESSURE"] = 1.0e4 * fator_pressao
    
    vazao_limpa = resolver_vazao_estacionaria(conec, Xno, config_base, C_estocastico=None)
    print(f"Pressão calibrada para o ensaio:       {config_base['INLET_PRESSURE']:.2f} Pa")
    print(f"Vazão calculada para a rede sem falhas: {vazao_limpa:.5e} m³/s")
    print(f"Limite crítico de falha do enunciado:  {config_base['V_CRITIC']:.5e} m³/s")
    print(f"A rede limpa falha? {vazao_limpa < config_base['V_CRITIC']}")
    print("================================================\n")
 
    print("Disparando loops estocásticos de Monte Carlo")
    N_convergido = avaliar_convergencia_monte_carlo(conec, Xno, config_base, p_O=0.35, f_obs=5, N_max=4000)
    varredura_probabilidade_individual(conec, Xno, config_base, N_estatistico=N_convergido)

    # ================================== Interpolação e Regressão ===============================

    config_mh = env.CONFIG_MH
    aprox = aprox_dados(config_mh)
    aprox.run() # 31 plots!!!

    #P2_PARTE_3
    solver_p2 = P2_3_GD(config_base)
    solver_p2.resolver_P2(E_target=7.5, H0=1000.0e-6, Tc=25.0)

    return

if __name__ == "__main__":
    main()


