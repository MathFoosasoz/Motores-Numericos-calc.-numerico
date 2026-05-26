import numpy as np
from plotting import PlotaMaxPressao, PlotaRede
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from data_structures import GeraGrafo
from hydraulics import Hydraulics_T
from thermal import Thermal_H
from plotting import plot_interpolation, plot_temp_hydraulics, plot_arestas_cromaticas_hidraulics
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

    
    def integrar_todas_as_arestas(self, N_subdivisoes, regra='trapezio'):
        """Calcula a temperatura media em todas as arestas simultaneamente
        empregando as regras de quadratura simples (N=1) e compostas vetorizadas.
        """
        conec = self.hydraulics.conec
        Xno = self.hydraulics.Xno

        p_i = Xno[conec[:, 0]]  
        p_j = Xno[conec[:, 1]]  

        if N_subdivisoes == 1:
            if regra == 'ponto_medio':
                p_medio = 0.5 * (p_i + p_j)
                T_media = self.interp(p_medio)
            elif regra == 'trapezio':
                T_i = self.interp(p_i)
                T_j = self.interp(p_j)
                T_media = 0.5 * (T_i + T_j)
        else:
            if regra == 'ponto_medio':
                t = np.linspace(0.5 / N_subdivisoes, 1.0 - 0.5 / N_subdivisoes, N_subdivisoes)
                pesos = np.ones(N_subdivisoes) / N_subdivisoes
            elif regra == 'trapezio':
                t = np.linspace(0.0, 1.0, N_subdivisoes + 1)
                pesos = np.ones(N_subdivisoes + 1) * (1.0 / N_subdivisoes)
                pesos[0] *= 0.5
                pesos[-1] *= 0.5

            pts_interp = p_i[:, np.newaxis, :] + t[np.newaxis, :, np.newaxis] * (p_j - p_i)[:, np.newaxis, :]
            shape_original = pts_interp.shape
            pts_flat = pts_interp.reshape(-1, 2)
            T_flat = self.interp(pts_flat)
            
            T_grid = T_flat.reshape(shape_original[0], shape_original[1])
            T_media = np.dot(T_grid, pesos)

        return T_media

    def analisar_estudo_de_casos(self):
        """avalia as versões simples e compostas (N=1, 10, 100, 1000),
        compara os resultados numéricos (Richardson) e quantifica o tempo de computação.
        """
        # Texto explicativo da estrategia matematica 
        print("\n" + "="*85)
        print(" ESTRATÉGIA MATEMÁTICA PARA ESTIMATIVA DE ERRO DE INTEGRAÇÃO")
        print("="*85)
        print("Como a solução analítica contínua da temperatura ao longo dos canais é desconhecida,")
        print("adotou-se a solução altamente refinada com N = 1000 subintervalos como nossa")
        print("Referência Assintótica Exata de controle (<T_k>_ref).")
        print("\nO erro de integração cometido em cada cenário (N = 1, 10, 100) foi estimado através")
        print("do Erro Absoluto Máximo Global (Norma do Infinito) sobre todas as arestas da rede:")
        print("\n      E_max^(N) = max_k | <T_k>^(N) - <T_k>^(N=1000) |")
        print("\nOnde k representa o índice de cada aresta e <T_k> é a temperatura média calculada.")
        print("="*85)

        configuracoes = [1, 10, 100, 1000]
        regras = ['ponto_medio', 'trapezio']
        
        print("\n" + "="*75)
        print(f"{'Regra de Integração':<20} | {'Subdiv. (N)':<12} | {'Tempo Exec. (s)':<15} | {'Erro Máx. Estimado':<18}")
        print("="*75)

        T_ref = {regra: self.integrar_todas_as_arestas(N_subdivisoes=1000, regra=regra) for regra in regras}

        for regra in regras:
            for N in configuracoes:
                t0 = time.perf_counter()
                T_obtida = self.integrar_todas_as_arestas(N_subdivisoes=N, regra=regra)
                t_exec = time.perf_counter() - t0

                erro_max = np.max(np.abs(T_obtida - T_ref[regra])) if N != 1000 else 0.0
                erro_str = f"{erro_max:.4e} °C" if N != 1000 else "--- (Ref. Exata)"

                nome_regra = "Ponto Médio" if regra == 'ponto_medio' else "Trapézio"
                print(f"{nome_regra:<20} | {N:<12} | {t_exec:<15.6f} | {erro_str:<18}")
                
        print("="*75 + "\n")

    
    def evaluate_coupling(self, plot=False):
        self.interpolator(method='cubic')

        #Roda a análise de tabelas, tempos e erros exigida no terminal
        self.analisar_estudo_de_casos()

        #Configura a malha fina de referência [241, 121]
        N_thin = [241, 121]
        self.thermal.N = N_thin
        
        # Temperatura estática nos nós (padrão Cecília)
        temp_nodes = self.evaluate_hydraulic_temps(method='cubic')

        # loop para gerar o gráfico para cada N solicitado pelo enunciado
        if plot:
            for N in [1, 10, 100, 1000]:
                # Calcula a temperatura média das arestas com o N atual do loop
                T_arestas_N = self.integrar_todas_as_arestas(N_subdivisoes=N, regra='trapezio')
                
                # Define se é a versão simples ou composta para o título
                tipo_versao = "Versão Simples" if N == 1 else f"Versão Composta (N={N})"
                titulo_grafico = f"Mapeamento nas Arestas via Trapézio - {tipo_versao}"
                
                # Chama a função do plotting.py passando o título específico deste N
                plot_arestas_cromaticas_hidraulics(
                    self.hydraulics.conec,
                    self.hydraulics.Xno,
                    temp_nodes,
                    T_arestas_N,
                    'cubic',
                    N_thin[0], N_thin[1],
                    titulo=titulo_grafico
                )
                plt.show()

    def resolver_rede_acoplada(self, N_subdivisoes, regra='trapezio'):
           # resolve o circuito hidráulico atualizando as condutâncias com base
           # nas temperaturas integradas de cada aresta.
            
            T_medias_arestas = self.integrar_todas_as_arestas(N_subdivisoes, regra)

            hyd = self.hydraulics
            
            # calcula o diâmetro hidráulico e a constante base 
            hydraulic_diameter = (4 * hyd.pipe_area / np.pi) ** 0.5 
            const_K_base = np.pi * (hydraulic_diameter ** 4) / 128.0

            # atualiza o vetor de condutâncias C cano por cano baseado na viscosidade local
            C_atualizado = np.zeros(hyd.num_pipes)
            for idx, connection in enumerate(hyd.conec):
                node_start, node_end = connection
                x_start, y_start = hyd.Xno[node_start]
                x_end, y_end = hyd.Xno[node_end]
                Lk = ((x_start - x_end) ** 2 + (y_start - y_end) ** 2) ** 0.5

                # calcula a viscosidade do fluido para a temperatura média deste cano específico
                mu_local = hyd.viscosity_t(T_medias_arestas[idx])
                
                # Condutância real corrigida termicamente
                C_atualizado[idx] = const_K_base / (mu_local * Lk)

            # salva o novo vetor C na instância hidráulica para os cálculos de vazão subsequentes
            hyd.C = C_atualizado

            # remonta a matriz A global de condutâncias da rede
            A_tilde = np.zeros((hyd.num_nodes, hyd.num_nodes))
            for idx, connectivity in enumerate(hyd.C):
                from_node = hyd.conec[idx, 0]
                to_node = hyd.conec[idx, 1]

                A_tilde[from_node, from_node] += connectivity
                A_tilde[to_node, to_node] += connectivity
                A_tilde[to_node, from_node] -= connectivity
                A_tilde[from_node, to_node] -= connectivity

            # aplica as condições de contorno de pressao no outlet 
            A_tilde[hyd.node_outlet, :] = 0
            A_tilde[hyd.node_outlet, hyd.node_outlet] = 1

            b_vector = np.zeros(hyd.num_nodes)
            for node, flow in hyd.inlet_flow_dict.items():
                b_vector[node] = flow
            b_vector[hyd.node_outlet] = hyd.outlet

          
            pressures = np.linalg.solve(A_tilde, b_vector)
            hyd.results['P'] = pressures

           
            matriz_K = np.diag(hyd.C)
            matriz_D = np.zeros((hyd.num_pipes, hyd.num_nodes))
            for k in range(hyd.num_pipes):
                matriz_D[k, hyd.conec[k, 0]] = 1
                matriz_D[k, hyd.conec[k, 1]] = -1

            Q = matriz_K @ matriz_D @ pressures 
            W = pressures.T @ matriz_D.T @ Q
            
            hyd.results['Q'] = Q
            hyd.results['W'] = W

            return np.max(pressures), W       
     
    def avaliar_impacto_malha_e_quadratura(self):
            #quantifica como a escolha da malha térmica e as regras de quadratura
            #influenciam a pressão máxima e a potência consumida na rede.
            
            N_refinada = [241, 121]
            N_grosseira = [21, 11]
            
            regras = ['ponto_medio', 'trapezio']
            subdivisoes = [1, 10, 100, 1000]

            print("\n" + "="*85)
            print(" AVALIAÇÃO QUANTITATIVA DO ACOPLAMENTO TERMO-HIDRÁULICO")
            print("="*85)
            print(f"{'Malha Térmica':<20} | {'Regra':<12} | {'Subdiv. (N)':<12} | {'Pressão Máx (Pa)':<18} | {'Potência Total (W)':<18}")
            print("-" * 85)

            for malha_nome, malha_dim in [("Grosseira", N_grosseira), ("Refinada", N_refinada)]:
                # altera a malha do interpolador rodando o solver térmico novamente
                self.thermal.N = malha_dim
                self.interpolator(method='cubic') 
                
                for regra in regras:
                    for N in subdivisoes:
                        p_max, potencia = self.resolver_rede_acoplada(N_subdivisoes=N, regra=regra)
                        
                        nome_regra = "Ponto Médio" if regra == 'ponto_medio' else "Trapézio"
                        print(f"{malha_nome:<20} | {nome_regra:<12} | {N:<12} | {p_max:<18.2f} | {potencia:<18.6f}")
            
            print("="*85 + "\n")


        