import numpy as np
import matplotlib.pyplot as plt
import time
import env

from data_structures import GeraGrafo
from hydraulics import Hydraulics_p3

def RandomFail(C_original , p_0, f_obs):

    C_falho = C_original.copy()
    num_pipes = len(C_original)
    sorteio = np.random.rand(num_pipes)

    indices_obstruidos = sorteio < p_0
    C_falho[indices_obstruidos] /= f_obs

    return C_falho

def resolver_vazao_estacionaria(conec, Xno, config_base, C_estocastico):
    solver_hidraulico = Hydraulics_p3(conec, Xno, config_base)

    solver_hidraulico.calculate_conductancy()

    if C_estocastico is not None:
        solver_hidraulico.C = C_estocastico
    
    A_tilde = np.zeros(shape=(solver_hidraulico.num_nodes, solver_hidraulico.num_nodes))
    for index, conectivity in enumerate(solver_hidraulico.C):
        from_node = solver_hidraulico.conec[index, 0]
        to_node = solver_hidraulico.conec[index, 1]
        A_tilde[from_node, from_node] += conectivity
        A_tilde[to_node, to_node] += conectivity
        A_tilde[to_node, from_node] -= conectivity
        A_tilde[from_node, to_node] -= conectivity

    line_to_find_inlet_flow = A_tilde[solver_hidraulico.node_inlet, :].copy()

    A_tilde[solver_hidraulico.node_outlet, :] = 0
    A_tilde[solver_hidraulico.node_outlet, solver_hidraulico.node_outlet] = 1

    A_tilde[solver_hidraulico.node_inlet, :] = 0
    A_tilde[solver_hidraulico.node_inlet, solver_hidraulico.node_inlet] = 1

    b_vector = np.zeros(shape=(solver_hidraulico.num_nodes))
    b_vector[solver_hidraulico.node_inlet] = solver_hidraulico.inlet   
    b_vector[solver_hidraulico.node_outlet] = solver_hidraulico.outlet 

    pressures = np.linalg.solve(A_tilde, b_vector)
    
    inlet_flow = np.dot(line_to_find_inlet_flow, pressures)
    
    return inlet_flow


def avaliar_convergencia_monte_carlo(conec, Xno, config_base, p_O=0.35, f_obs=5, N_max=4000):
    limite_critico = config_base["V_CRITIC"]
    falhas_acumuladas = 0
    prob_historico = []
    
    modelo_base = Hydraulics_p3(conec, Xno, config_base)
    C_original = modelo_base.calculate_conductancy()
    
    print(f"Rodando convergencia assintotica de Monte Carlo (N={N_max}) para p_O={p_O} e f_obs={f_obs}")
    
    for n in range(1, N_max + 1):
        C_cenario = RandomFail(C_original, p_O, f_obs)
        q_inlet = resolver_vazao_estacionaria(conec, Xno, config_base, C_cenario)
        
        if q_inlet < limite_critico:
            falhas_acumuladas += 1
            
        prob_historico.append(falhas_acumuladas / n)
        
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, N_max + 1), prob_historico, color='tab:blue', linewidth=1.5)
    plt.axhline(y=prob_historico[-1], color='r', linestyle='--', label=f'Prob Convergida = {prob_historico[-1]:.4f}')
    plt.title(f"Comportamento Assintotico do Estimador de Monte Carlo\n$p_O$ = {p_O} | $f_{{obs}}$ = {f_obs}", fontsize=11, fontweight='bold')
    plt.xlabel("Numero Progressivo de Realizações ($N$)")
    plt.ylabel("Probabilidade Global de Falha ($Prob$)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.savefig("convergencia_assintotica_monte_carlo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return N_max

def varredura_probabilidade_individual(conec, Xno, config_base, N_estatistico=4000):
    inicio, fim, pontos = config_base["P_O_RANGE"]
    p_O_valores = np.linspace(inicio, fim, pontos)
    
    resultados = {5: [], 10: []}
    limite_critico = config_base["V_CRITIC"]
    
    modelo_base = Hydraulics_p3(conec, Xno, config_base)
    C_original = modelo_base.calculate_conductancy()
    
    print("\nIniciando Varredura do Domínio de Falhas Individuais...")
    
    for f_obs in config_base["SEVERITIES"]:
        print(f"Analisando fator de severidade de obstrução: f_obs = {f_obs}")
        for p_O in p_O_valores:
            falhas = 0
            for _ in range(N_estatistico):
                C_cenario = RandomFail(C_original, p_O, f_obs)
                q_inlet = resolver_vazao_estacionaria(conec, Xno, config_base, C_cenario)
                
                if q_inlet < limite_critico:
                    falhas += 1
            
            prob_global = falhas / N_estatistico
            resultados[f_obs].append(prob_global)
            print(f"  p_O = {p_O:.2f} -> Prob Global de Falha = {prob_global:.4f}")
            
    plt.figure(figsize=(9, 6))
    plt.plot(p_O_valores, resultados[5], 'o-', label='$f_{obs} = 5$ (Severidade Moderada)', color='tab:blue', linewidth=2)
    plt.plot(p_O_valores, resultados[10], 's-', label='$f_{obs} = 10$ (Severidade Alta)', color='tab:orange', linewidth=2)
    
    plt.title("Domínio de Vulnerabilidade Hidráulica do Gêmeo Digital\nProbabilidade Global de Falha vs Desgaste Operacional", fontsize=12, fontweight='bold')
    plt.xlabel("Probabilidade de Obstrução Individual dos Canais ($p_O$)")
    plt.ylabel("Probabilidade Global de Falha do Sistema ($Prob$)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.savefig("curva_vulnerabilidade_convergida.png", dpi=300, bbox_inches='tight')
    plt.show()