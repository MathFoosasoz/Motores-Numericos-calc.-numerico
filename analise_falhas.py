import numpy as np
import matplotlib.pyplot as plt
import time
import env
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator

from data_structures import GeraGrafo
from hydraulics import Hydraulics_p3
from hydraulic_thermal import Hydraulic_to_Thermal
from mechanic_hydraulic import MechanicHydraulic

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


class MechanicHydraulicMonteCarlo(MechanicHydraulic):

    def __init__(self, config):
        super().__init__(config)
        self.condutancias_amostra = None

    def definir_condutancias(self, condutancias):
        self.condutancias_amostra = np.asarray(condutancias, dtype=float).copy()

    def montar_matriz_rede_adimensional(self, largura_canal=None):
        if self.condutancias_amostra is None:
            return super().montar_matriz_rede_adimensional(largura_canal)

        escalas = self.calcular_escalas()
        C = self.condutancias_amostra

        A_fisico = sparse.lil_matrix((self.num_nodes, self.num_nodes))
        for index, conectivity in enumerate(C):
            from_node = self.conec[index, 0]
            to_node = self.conec[index, 1]

            A_fisico[from_node, from_node] += conectivity
            A_fisico[to_node, to_node] += conectivity
            A_fisico[to_node, from_node] -= conectivity
            A_fisico[from_node, to_node] -= conectivity

        A_fisico = A_fisico.tocsr()
        fator_adimensional = escalas["p_ref"] / (escalas["v_ref"] * self.R ** 2)
        A = (A_fisico * fator_adimensional).tolil()

        A[self.node_inlet, :] = 0.0
        A[self.node_inlet, self.node_inlet] = 1.0

        return A.tocsr(), A_fisico, C, escalas


class MonteCarloDinamico:

    def __init__(self, r_cond=0.0005, N_termico=(241, 121), N_quadratura=1000):
        config_mh = dict(env.CONFIG_MH)
        config_mh["N"] = (51, 51)
        config_mh["INLET_PRESSURE"] = 5.0e3
        config_mh["CHANNEL_WIDTH"] = 1000.0e-6
        config_mh["TIME_END"] = 4.0

        config_ht = dict(env.CONFIG_HT)
        config_ht["N"] = N_termico
        config_ht["SOURCE"] = 5.0e5
        config_ht["PIPE_AREA"] = (1000.0e-6) ** 2

        self.modelo_dinamico = MechanicHydraulicMonteCarlo(config_mh)
        self.modelo_termico = Hydraulic_to_Thermal(config_ht)
        self.r_cond = r_cond
        self.N_quadratura = N_quadratura

        self.condutancias_originais = self.calcular_condutancias_termicas()

    def calcular_condutancias_termicas(self):
        temperaturas, _, _, _ = self.modelo_termico.solve_system_sparse(self.r_cond)

        Nx, Ny = self.modelo_termico.thermal.N
        Lx, Ly = self.modelo_termico.thermal.L
        campo_temperaturas = temperaturas.reshape(Ny, Nx).T

        interpolador = RegularGridInterpolator(
            (np.linspace(0.0, Lx, Nx), np.linspace(0.0, Ly, Ny)),
            campo_temperaturas,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        conec = self.modelo_termico.hydraulics.conec
        Xno = self.modelo_termico.hydraulics.Xno
        pontos_iniciais = Xno[conec[:, 0]]
        pontos_finais = Xno[conec[:, 1]]

        t = np.linspace(0.0, 1.0, self.N_quadratura + 1)
        pesos = np.ones(self.N_quadratura + 1) / self.N_quadratura
        pesos[0] *= 0.5
        pesos[-1] *= 0.5

        pontos = (
            pontos_iniciais[:, np.newaxis, :]
            + t[np.newaxis, :, np.newaxis]
            * (pontos_finais - pontos_iniciais)[:, np.newaxis, :]
        )
        temperaturas_arestas = interpolador(
            pontos.reshape(-1, 2)
        ).reshape(len(conec), self.N_quadratura + 1) @ pesos

        viscosidades = 0.001791 / (
            1.0
            + 0.03368 * temperaturas_arestas
            + 0.000221 * temperaturas_arestas ** 2
        )

        area_canal = (1000.0e-6) ** 2
        diametro_hidraulico = (4.0 * area_canal / np.pi) ** 0.5
        constante_geometrica = np.pi * diametro_hidraulico ** 4 / 128.0
        comprimentos = np.linalg.norm(pontos_finais - pontos_iniciais, axis=1)

        return constante_geometrica / (viscosidades * comprimentos)

    def calcular_energia(self, dt, condutancias):
        self.modelo_dinamico.definir_condutancias(condutancias)
        resultado = self.modelo_dinamico.resolver_caso_base(
            N=(51, 51),
            dt=dt,
            tempo_final=4.0,
            pressao_inlet=5.0e3,
            largura_canal=1000.0e-6,
            print_info=False,
        )

        escalas = self.modelo_dinamico.calcular_escalas()
        potencia_referencia = (
            escalas["p_ref"]
            * escalas["v_ref"]
            * self.modelo_dinamico.R ** 2
        )
        potencia_adimensional = resultado["potencia"] / potencia_referencia

        return np.trapezoid(potencia_adimensional, resultado["time"])

    def executar(self, p_O=0.35, f_obs=5, N=2000, plot=True):
        dt_valores = [0.05, 0.1]
        limite_energia = 7.0

        sorteios = np.random.rand(N, self.modelo_dinamico.num_pipes)
        mascaras_obstrucao = sorteios < p_O
        resultados = {}

        print("\nIniciando análise dinâmica do Gêmeo Digital Completo...")
        for dt in dt_valores:
            eventos_acumulados = 0
            probabilidades = []
            energias = []

            print(f"Analisando passo de tempo dt = {dt}")
            for n, mascara in enumerate(mascaras_obstrucao, start=1):
                C_cenario = self.condutancias_originais.copy()
                C_cenario[mascara] /= f_obs

                energia = self.calcular_energia(dt, C_cenario)
                energias.append(energia)

                if energia < limite_energia:
                    eventos_acumulados += 1

                probabilidades.append(eventos_acumulados / n)

                if n % 100 == 0:
                    print(f"  {n}/{N} realizações concluídas")

            resultados[dt] = {
                "energias": np.array(energias),
                "probabilidades": np.array(probabilidades),
                "probabilidade_final": probabilidades[-1],
            }

        diferenca = (
            resultados[0.1]["probabilidade_final"]
            - resultados[0.05]["probabilidade_final"]
        )

        print("\nResultados do Problema 2:")
        for dt in dt_valores:
            print(
                f"dt = {dt}: Prob(E < 7.0) = "
                f"{resultados[dt]['probabilidade_final']:.5f}"
            )
        print(f"Diferença entre os estimadores: {diferenca:+.5f}")

        if plot:
            self.plotar_resultados(resultados)

        return resultados

    def plotar_resultados(self, resultados):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))

        for dt, resultado in resultados.items():
            realizacoes = np.arange(1, len(resultado["probabilidades"]) + 1)
            axs[0].plot(
                realizacoes,
                resultado["probabilidades"],
                label=f'dt = {dt}',
            )

        axs[0].set_title("Convergência do estimador de Monte Carlo")
        axs[0].set_xlabel("Número de realizações")
        axs[0].set_ylabel("Prob(E < 7.0)")
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend()

        dt_valores = list(resultados.keys())
        probabilidades = [
            resultados[dt]["probabilidade_final"] for dt in dt_valores
        ]
        axs[1].plot(dt_valores, probabilidades, 'o-', color='tab:purple')
        axs[1].set_title("Impacto do passo de tempo")
        axs[1].set_xlabel("Passo de tempo dt")
        axs[1].set_ylabel("Prob(E < 7.0)")
        axs[1].set_ylim(0.0, 1.0)
        axs[1].grid(True, linestyle=':', alpha=0.6)

        fig.suptitle("Análise Dinâmica do Gêmeo Digital Completo")
        plt.tight_layout()
        plt.savefig(
            "comparacao_passo_tempo_monte_carlo.png",
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()


def executar_monte_carlo_dinamico(p_O=0.35, f_obs=5, N=2000, plot=True):
    analise = MonteCarloDinamico(
        r_cond=0.0005,
        N_termico=(241, 121),
        N_quadratura=1000,
    )
    return analise.executar(p_O=p_O, f_obs=f_obs, N=N, plot=plot)
