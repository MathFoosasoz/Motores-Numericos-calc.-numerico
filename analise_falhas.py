import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.sparse.linalg import factorized
import time

from data_structures import GeraGrafo
from hydraulics import Hydraulics_p3
from mechanic_hydraulic import MechanicHydraulic
from plotting import plot_aprox_dados, plot_fitting

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

class aprox_dados():

    def __init__(self, config):
        self.N = (81, 161)
        config["N"] = self.N

        self.dt = 0.05
        config["DT"] = self.dt

        self.time_end = 4
        config["TIME_END"] = self.time_end

        self.channel_width = config["CHANNEL_WIDTH"]
        self.inlet_pressure = config["INLET_PRESSURE"]

        self.mh = MechanicHydraulic(config)

        self.known = np.linspace(0, 100*self.time_end, int(self.time_end/self.dt) + 1)
        self.unknown_full = np.linspace(0, 100*self.time_end, 5 * int(self.time_end/self.dt) + 1)
        self.unknown = self.unknown_full[~np.isin(self.unknown_full, self.known)]

        self.known /= 100
        self.unknown /= 100

        self.potencia = self.pot()
        self.potencia_full = self.pot(dt = 0.01)

        self.potencia_com_ruido = self.pot(ruido = True)
        self.potencia_com_ruido_full = self.pot(ruido = True, dt = 0.01)

    def interpolação_linear(self, potencia = None):
        if potencia is None:
            potencia = self.potencia

        linear_interp = interp1d(self.known, potencia)
        potencia_full = linear_interp(self.unknown)

        return potencia_full
    
    def interpolação_cubica(self, potencia = None):
        if potencia is None:
            potencia = self.potencia

        cubic_interp = CubicSpline(self.known, potencia)
        potencia_full = cubic_interp(self.unknown)

        return potencia_full
    
    def regressão_polinomial(self, potencia_full = None, potencia = None):
        potencias = []
        errs = []

        if potencia_full is None:
            potencia_full = self.potencia_full

        if potencia is None:
            potencia = self.potencia
        
        for m in range(3, 16): 
            params = np.polynomial.polynomial.Polynomial.fit(self.known, potencia, m)
            pot_regression = params(self.unknown)
            err = 0.0

            sub = 0
            for i in range(len(potencia_full)):
                if i%5 == 0:
                    sub += 1
                    continue

                err += (pot_regression[i - sub] - potencia_full[i])**2

            potencias.append(pot_regression)
            errs.append(np.sqrt(err))

        return potencias, errs
    
    def pot(self, ruido = False, dt = None):

        if dt == None:
            dt = self.dt

        sistema = self.mh.montar_sistema_global(dt, self.channel_width)
        solver = factorized(sistema["A_global"].tocsc())

        n_m = self.mh.num_nodes_membrana
        n_steps = int(round(self.time_end / dt))
        idt = 1.0 / dt

        w = np.zeros(n_m)
        v = np.zeros(n_m)
        p = np.zeros(self.mh.num_nodes)

        potencia = [0.0]
        for _ in range(1, n_steps + 1):

            p_inlet = self.inlet_pressure*(0.85 + 0.3 * np.random.rand()) if ruido else self.inlet_pressure

            b_pressao = self.mh.montar_vetor_pressao_inlet(p_inlet)

            rhs = np.concatenate([
                idt * w,
                idt * (sistema["M"] @ v),
                b_pressao,
            ])

            solucao = solver(rhs)
            w = solucao[:n_m]
            v = solucao[n_m:2 * n_m]
            p = solucao[2 * n_m:]

            estado = self.mh.medir_estado(w, v, p)
            potencia.append(estado["potencia"])

        return np.array(potencia)
    

    def run(self):
    
        pot_L = self.interpolação_linear()
        pot_L_ruido = self.interpolação_linear(self.potencia_com_ruido)

        plot_aprox_dados(
            "Interpolação linear (sem ruido)",
            self.known,
            self.potencia,
            self.unknown,
            pot_L
        )

        plot_aprox_dados(
            "Interpolação linear (com ruido)",
            self.known,
            self.potencia_com_ruido,
            self.unknown,
            pot_L_ruido
        )

        pot_C = self.interpolação_cubica()
        pot_C_ruido = self.interpolação_cubica(self.potencia_com_ruido)

        plot_aprox_dados(
            "Interpolação cúbica (sem ruido)",
            self.known,
            self.potencia,
            self.unknown,
            pot_C
        )

        plot_aprox_dados(
            "Interpolação cúbica (com ruido)",
            self.known,
            self.potencia_com_ruido,
            self.unknown,
            pot_C_ruido
        )

        pots_poly, errs = self.regressão_polinomial()
        pots_poly_ruido, errs_ruido = self.regressão_polinomial(self.potencia_com_ruido_full, self.potencia_com_ruido)

        for index, pot_poly in enumerate(pots_poly):
            plot_aprox_dados(
                f"Regressão polinomial (sem ruido) m = {index+3} e = {errs[index]:.4f}",
                self.known,
                self.potencia,
                self.unknown,
                pot_poly
            )

        for index, pot_poly_ruido in enumerate(pots_poly_ruido):
            plot_aprox_dados(
                f"Regressão polinomial (com ruido) m = {index+3} e = {errs_ruido[index]:.4f}",
                self.known,
                self.potencia_com_ruido,
                self.unknown,
                pot_poly_ruido
            )

        plot_fitting(np.linspace(3, 15, 13), errs, errs_ruido)

