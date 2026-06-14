import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra 
from mechanic_hydraulic import MechanicHydraulic, gerar_todos_os_plots, plotar_perfil_membrana_corte
from data_structures import GeraGrafo
from analysis import complexity_analysis
from plotting import plot_relaxamento_problema3

def main():

    config_t = env.CONFIG_T

    #FUNCIONA!!!
    #test_sparse = Thermal(config_t, method = "sparse")
    #test_sparse.run(print_info = True, plot = True)

    #FUNCIONA!!!
    #test_cholesky = Thermal(config_t, method = "cholesky")
    #test_cholesky.run(print_info = True, plot= True)

    #FUNCIONA!!! 
    #test_P1 = Thermal_P1(config_t, method="sparse")
    #test_P1.complexity_analysis(print_info=True, plot=True)
    #test_P1.run(print_info = True, plot = True)
    
    #FUNCIONA!!!
    #test_P2 = Thermal_P2(config_t, method="sparse")
    #test_P2.run(print_info = False, plot = True)

    #FUNCIONA!!!
    #test_P3 = Thermal_P3(config_t, method="sparse")
    #test_P3.run(print_info = False, plot = True)

    #FUNCIONA!!!
    #test_P4 = Thermal_P4(config_t)
    #test_P4.run(plot=True)

    #FUNCIONA!!!
    #test_P5 = Thermal_P5(config_t, k_node=233)
    #test_P5.run(print_info=True)
    
    #FUNCIONA !!!
    #test_jacobi = Thermal_P1_extra(config_t, method = "jacobi")
    #test_gauss_seidel = Thermal_P1_extra(config_t, method = "gauss_seidel")
    #test_jacobi.run(print_info = True, plot = True)
    #test_gauss_seidel.run(print_info = True, plot = True)
    
    #test_gauss_seidel.run(print_info = True, plot = True, analyze_subdivisions=True, analyze_tolerance=True)

    #FUNCIONA !!!
    #gsimulacao = Thermal_P2_Extra(config_t)
    #res_j, res_gs = simulacao.run_comparison_history(frame_step=20)
    #simulacao.animate_comparison(interval=50)

    #FUNCIONA !!!
    #test_P3_extra = Thermal_P3_Extra(config_t)
    #test_P3_extra.run(print_info = True, plot=False)

    config_mh = env.CONFIG_MH

    simulador = MechanicHydraulic(config_mh)

    # ==============================================================================
    # COMENTADO PARA NÃO RODAR A VARREDURA DEMORADA DO EX 2 TODA VEZ:
    # ==============================================================================
    # print("="*8 + "Iniciando a varredura transiente de todos os cenários" + "="*8)
   
    # todos_resultados = simulador.resolver_todos_cenarios(print_info=False)
   
    # print("Simulações finalizadas. Gerando arquivos de plotagem...")
   
    # gerar_todos_os_plots(todos_resultados)
    # print("Gráficos salvos")
    # ==============================================================================

    print("\n" + "="*8 + " [EXERCÍCIO 2] Obtendo o estado inicial inflado (Malha 51x51, dt=0.025) " + "="*8)
    #Roda o caso base uma vez com as restrições do enunciado para gerar o ponto de partida (estado inflado)
    estado_inflado_ex2 = simulador.resolver_caso_base(
        N=(51, 51),
        dt=0.025,
        tempo_final=config_mh["TIME_END"], 
        pressao_inlet=config_mh["INLET_PRESSURE"], 
        largura_canal=config_mh["CHANNEL_WIDTH"],
        print_info=True
    )

    print("\n" + "="*8 + " [EXERCÍCIO 3] Iniciando simulação de relaxamento (P_inlet = 0) " + "="*8)
    #Executa o problema 3 passando o dicionário com os estados finais coletados no passo anterior
    resultado_problema3 = simulador.resolver_relaxamento(
        estado_inicial_ex2=estado_inflado_ex2,
        dt=0.025,
        tempo_final=12.0, 
        largura_canal=config_mh["CHANNEL_WIDTH"],
        print_info=True
    )

    print("Gerando e salvando os gráficos temporais do Exercício 3...")
    plot_relaxamento_problema3(resultado_problema3, filename="relaxamento_malha51_dt0025.png")
    print("Gráficos salvos com sucesso!")


    return


if __name__ == "__main__":
    main()


