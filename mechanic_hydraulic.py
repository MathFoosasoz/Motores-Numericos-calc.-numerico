import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks

from data_structures import GeraGrafo
from hydraulics import Hydraulics
from mechanic import Mechanic

class MechanicHydraulic():

    def __init__(self, config):
        self.config = config

        self.N = config["N"]
        self.R = config["R"]
        self.sigma = config["TENSION"]
        self.rho = config["DENSITY"]
        self.e = config["THICKNESS"]
        self.beta = config["BETA"]

        self.node_inlet = config["N_INLET"]
        self.node_outlet = config["N_OUTLET"]
        self.inlet_pressure = config["INLET_PRESSURE"]
        self.viscosity = config["VISCOSITY"]
        self.largura_canal = config["CHANNEL_WIDTH"]

        self.dt = config["DT"]
        self.time_end = config["TIME_END"]
        self.w0_factor = config["W0_FACTOR"]

        levels = config["LEVELS"]
        self.Xno, self.conec = GeraGrafo(levels)
        self.Xno = self.Xno * 0.001

        self.num_nodes = int(np.max(self.conec)) + 1
        self.num_pipes = self.conec.shape[0]

        self.mechanic = Mechanic({
            "N": self.N,
            "R": self.R,
            "TENSION": self.sigma,
            "DENSITY": self.rho,
            "THICKNESS": self.e,
            "TOLERANCE": config["TOLERANCE"],
            "N_MODES": config["N_MODES"],
        })

        self.hydraulics = Hydraulics(self.conec, self.Xno, {
            "N_INLET": self.node_inlet,
            "INLET_FLOW": 0.0,
            "N_OUTLET": self.node_outlet,
            "OUTLET": 0.0,
            "PIPE_AREA": self.largura_canal ** 2,
            "VISCOSITY": self.viscosity,
        })
        self.hydraulics.calculate_conductancy()

        self.results = {
            "time": None,
            "deslocamento_centro": None,
            "pressao_outlet": None,
            "vazao_outlet": None,
            "volume_reservatorio": None,
            "potencia": None,
            "deslocamentos_finais": None,
            "velocidades_finais": None,
            "pressoes_finais": None,
            "snapshots_deslocamento": None,
        }

    def ij2n(self, i, j):
        return self.mechanic.ij2n(i, j)

    def calcular_escalas(self):
        w_ref = self.w0_factor * self.R
        t_ref = self.R * np.sqrt(self.rho * self.e / self.sigma)
        v_ref = w_ref / t_ref
        p_ref = self.sigma * w_ref / (self.R ** 2)

        return {
            "w_ref": w_ref,
            "t_ref": t_ref,
            "v_ref": v_ref,
            "p_ref": p_ref,
        }

    def calcular_area_elemento_membrana(self):
        dx = 2.0 / (self.N[0] - 1)
        dy = 2.0 / (self.N[1] - 1)
        return dx * dy

    def get_indices_fora_do_circulo(self):
        self.indices_fora_do_circulo = self.mechanic.get_index_outside_circus()
        return self.indices_fora_do_circulo

    def montar_matrizes_membrana(self):
        N0, N1 = self.N
        nunk = N0 * N1
        h2 = self.calcular_area_elemento_membrana()

        # A mascara circular vem da classe Mechanic, mas a matriz usada aqui
        # segue a forma adimensional do sistema acoplado.
        d1 = 4.0 * np.ones(nunk)
        d2 = -np.ones(nunk - 1)
        d3 = -np.ones(nunk - N0)

        K = (1.0 / h2) * sparse.diags(
            [d3, d2, d1, d2, d3],
            [-N0, -1, 0, 1, N0],
            format="lil"
        )

        indices_restritos = self.get_indices_fora_do_circulo()
        big_number = 1.0e7

        for idx in indices_restritos:
            K.rows[idx] = [idx]
            K.data[idx] = [big_number]

        K = K.tocsc()
        for idx in indices_restritos:
            inicio = K.indptr[idx]
            fim = K.indptr[idx + 1]
            K.data[inicio:fim] = 0.0
            K[idx, idx] = big_number

        M = sparse.identity(nunk, format="csr")

        self.num_nodes_membrana = nunk
        self.indices_restritos = indices_restritos

        return K.tocsr(), M

    def montar_matriz_rede_adimensional(self, largura_canal=None):
        escalas = self.calcular_escalas()

        if largura_canal is None:
            largura_canal = self.largura_canal

        self.hydraulics.pipe_area = largura_canal ** 2

        C = self.hydraulics.calculate_conductancy()

        A_fisico = sparse.lil_matrix((self.num_nodes, self.num_nodes))
        for idx, c_k in enumerate(C):
            n1 = self.conec[idx, 0]
            n2 = self.conec[idx, 1]

            A_fisico[n1, n1] += c_k
            A_fisico[n2, n2] += c_k
            A_fisico[n1, n2] -= c_k
            A_fisico[n2, n1] -= c_k

        A_fisico = A_fisico.tocsr()
        C_fisico = C

        fator_adimensional = escalas["p_ref"] / (escalas["v_ref"] * self.R ** 2)
        A = (A_fisico * fator_adimensional).tolil()

        A[self.node_inlet, :] = 0.0
        A[self.node_inlet, self.node_inlet] = 1.0

        return A.tocsr(), A_fisico, C_fisico, escalas

    def montar_matriz_acoplamento(self):
        nunk = self.N[0] * self.N[1]
        U = sparse.lil_matrix((self.num_nodes, nunk))

        pesos_membrana = np.ones(nunk)
        pesos_membrana[self.indices_restritos] = 0.0
        U[self.node_outlet, :] = pesos_membrana

        return U.tocsr()

    def montar_sistema_global(self, dt, largura_canal=None):
        K, M = self.montar_matrizes_membrana()
        A, A_fisico, C_fisico, escalas = self.montar_matriz_rede_adimensional(largura_canal)
        U = self.montar_matriz_acoplamento()

        n_m = self.num_nodes_membrana
        n_p = self.num_nodes
        idt = 1.0 / dt
        h2 = self.calcular_area_elemento_membrana()

        Iden = sparse.identity(n_m, format="csr")
        zero_m_p = sparse.csr_matrix((n_m, n_p))
        zero_p_m = sparse.csr_matrix((n_p, n_m))

        blocks = [
            [idt * Iden, -Iden, zero_m_p],
            [K, (idt + self.beta) * M, -U.T],
            [zero_p_m, h2 * U, A],
        ]

        A_global = sparse.bmat(blocks, format="csr")

        self.sistema_atual = {
            "A_global": A_global,
            "A_fisico": A_fisico,
            "C_fisico": C_fisico,
            "U": U,
            "M": M,
            "K": K,
            "dt": dt,
            "h2": h2,
            "escalas": escalas,
        }

        return self.sistema_atual

    def montar_vetor_pressao_inlet(self, pressao_inlet):
        b = np.zeros(self.num_nodes)
        p_ref = self.sistema_atual["escalas"]["p_ref"]
        b[self.node_inlet] = pressao_inlet / p_ref
        return b

    def calcular_vazoes_e_potencia(self, pressoes):
        C = self.sistema_atual["C_fisico"]

        delta_p = np.zeros(self.num_pipes)
        for k, connection in enumerate(self.conec):
            from_node = connection[0]
            to_node = connection[1]
            delta_p[k] = pressoes[from_node] - pressoes[to_node]

        vazoes = C * delta_p
        potencia = float(np.dot(delta_p, vazoes))
        return vazoes, potencia

    def indice_centro_membrana(self):
        return self.ij2n(self.N[1] // 2, self.N[0] // 2)

    def medir_estado(self, w, v, p):
        escalas = self.sistema_atual["escalas"]
        h2 = self.sistema_atual["h2"]
        U = self.sistema_atual["U"]
        A_fisico = self.sistema_atual["A_fisico"]

        deslocamentos = w * escalas["w_ref"]
        velocidades = v * escalas["v_ref"]
        pressoes = p * escalas["p_ref"]

        volume_adim = h2 * (U @ w)[self.node_outlet]
        volume = volume_adim * escalas["w_ref"] * self.R ** 2

        balanco_nodal = A_fisico @ pressoes
        vazao_outlet = -float(balanco_nodal[self.node_outlet])

        _, potencia = self.calcular_vazoes_e_potencia(pressoes)

        return {
            "deslocamento_centro": float(deslocamentos[self.indice_centro_membrana()]),
            "pressao_outlet": float(pressoes[self.node_outlet]),
            "vazao_outlet": vazao_outlet,
            "volume_reservatorio": float(volume),
            "potencia": potencia,
            "deslocamentos": deslocamentos,
            "velocidades": velocidades,
            "pressoes": pressoes,
        }

    def resolver_caso_base(
        self,
        N=None,
        dt=None,
        tempo_final=None,
        pressao_inlet=None,
        largura_canal=None,
        beta=None,
        salvar_campos=False,
        passo_salvar_campos=None,
        print_info=True,
    ):
        if N is not None:
            self.N = tuple(N)
            self.mechanic.N = self.N
        if dt is None:
            dt = self.dt
        if tempo_final is None:
            tempo_final = self.time_end
        if pressao_inlet is None:
            pressao_inlet = self.inlet_pressure
        if largura_canal is None:
            largura_canal = self.largura_canal
        if beta is not None:
            self.beta = beta

        start = time.time()
        sistema = self.montar_sistema_global(dt, largura_canal)
        solver = factorized(sistema["A_global"].tocsc())

        n_m = self.num_nodes_membrana
        n_steps = int(round(tempo_final / dt))
        idt = 1.0 / dt

        w = np.zeros(n_m)
        v = np.zeros(n_m)
        p = np.zeros(self.num_nodes)

        tempos = [0.0]
        deslocamento_centro = [0.0]
        pressao_outlet = [0.0]
        vazao_outlet = [0.0]
        volume_reservatorio = [0.0]
        potencia = [0.0]
        snapshots = []

        if salvar_campos:
            snapshots.append((0.0, np.zeros(n_m)))

        b_pressao = self.montar_vetor_pressao_inlet(pressao_inlet)

        for step in range(1, n_steps + 1):
            rhs = np.concatenate([
                idt * w,
                idt * (sistema["M"] @ v),
                b_pressao,
            ])

            solucao = solver(rhs)
            w = solucao[:n_m]
            v = solucao[n_m:2 * n_m]
            p = solucao[2 * n_m:]

            estado = self.medir_estado(w, v, p)
            tempo_atual = step * dt

            tempos.append(tempo_atual)
            deslocamento_centro.append(estado["deslocamento_centro"])
            pressao_outlet.append(estado["pressao_outlet"])
            vazao_outlet.append(estado["vazao_outlet"])
            volume_reservatorio.append(estado["volume_reservatorio"])
            potencia.append(estado["potencia"])

            if salvar_campos:
                salvar_agora = passo_salvar_campos is None or step % passo_salvar_campos == 0
                if salvar_agora:
                    snapshots.append((tempo_atual, estado["deslocamentos"].copy()))

        estado_final = self.medir_estado(w, v, p)

        self.results = {
            "time": np.array(tempos),
            "deslocamento_centro": np.array(deslocamento_centro),
            "pressao_outlet": np.array(pressao_outlet),
            "vazao_outlet": np.array(vazao_outlet),
            "volume_reservatorio": np.array(volume_reservatorio),
            "potencia": np.array(potencia),
            "deslocamentos_finais": estado_final["deslocamentos"],
            "velocidades_finais": estado_final["velocidades"],
            "pressoes_finais": estado_final["pressoes"],
            "snapshots_deslocamento": snapshots,
            "configuracao": {
                "N": self.N,
                "dt": dt,
                "tempo_final": tempo_final,
                "pressao_inlet": pressao_inlet,
                "largura_canal": largura_canal,
                "beta": self.beta,
            },
            "tempo_execucao": time.time() - start,
        }

        if print_info:
            self.print_resumo()

        return self.results

    def resolver_todos_cenarios(self, print_info=True):
        resultados = []

        for N in self.config["MULTI_N"]:
            for dt in self.config["DT_VALUES"]:
                for pressao in self.config["INLET_PRESSURES"]:
                    for largura in self.config["CHANNEL_WIDTHS"]:
                        resultado = self.resolver_caso_base(
                            N=N,
                            dt=dt,
                            pressao_inlet=pressao,
                            largura_canal=largura,
                            print_info=print_info,
                            salvar_campos=True
                        )
                        resultados.append(resultado)

        return resultados

    def print_resumo(self):
        cfg = self.results["configuracao"]
        print("\nRESULTADOS - ACOPLAMENTO HIDRAULICO-MECANICO")
        print(f"N: {cfg['N'][0]} x {cfg['N'][1]}")
        print(f"dt: {cfg['dt']}")
        print(f"tempo final: {cfg['tempo_final']}")
        print(f"pressao inlet: {cfg['pressao_inlet']:.4e} Pa")
        print(f"largura canal: {cfg['largura_canal']:.4e} m")
        print(f"pressao outlet final: {self.results['pressao_outlet'][-1]:.4e} Pa")
        print(f"vazao outlet final: {self.results['vazao_outlet'][-1]:.4e} m3/s")
        print(f"deslocamento centro final: {self.results['deslocamento_centro'][-1]:.4e} m")
        print(f"volume final: {self.results['volume_reservatorio'][-1]:.4e} m3")
        print(f"potencia final: {self.results['potencia'][-1]:.4e} W")
        print(f"tempo de execucao: {self.results['tempo_execucao']:.4f} s\n")

    def run(self, print_info=True, plot=False):
        if plot:
            print("A plotagem ficou fora deste arquivo; use os historicos em self.results.")

        return self.resolver_caso_base(print_info=print_info)

def gerar_todos_os_plots(resultados_simulacao):
    """
    Gera gráficos de evolução temporal para todas as grandezas de interesse,
    organizados por pressão de entrada. Cada figura mostra uma legenda compacta
    baseada em facetas (malha × δt), evitando repetições e garantindo legibilidade.
    """
 
    pressões_unicas = sorted(set(r["configuracao"]["pressao_inlet"] for r in resultados_simulacao))
 
    grandezas = [
        ("deslocamento_centro", "Deslocamento Vertical do Centro",        "Deslocamento (m)"),
        ("pressao_outlet",      "Pressão no Nó de Descarga $p_{outlet}$", "Pressão (Pa)"),
        ("vazao_outlet",        "Vazão de Saída $q_{outlet}$",            "Vazão ($m^3/s$)"),
        ("volume_reservatorio", "Volume Acumulado no Reservatório",        "Volume ($m^3$)"),
        ("potencia",            "Potência Consumida pelo Sistema",         "Potência (W)"),
    ]
 
    # ── Paleta e estilos ────────────────────────────────────────────────────────
    # Cores distinguem tamanho de malha; estilos de linha distinguem δt.
    # Usamos dois tons de azul/laranja com contraste suficiente.
    cores_malha  = {21: "#2ca02c", 51: "#1f77b4", 101: "#d62728"}
    estilos_dt   = {0.00625: "-", 0.0125: "--", 0.025: ":", 0.05: "-."}
    lw_dt        = {0.00625: 1.6, 0.0125: 1.4, 0.025: 1.2, 0.05: 1.0}
    alpha_dt     = {0.00625: 0.95, 0.0125: 0.80, 0.025: 0.65, 0.05: 0.50}
 
    n_cols = len(pressões_unicas)
 
    for chave, titulo, ylabel in grandezas:
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(6 * n_cols + 3, 5),
            sharey=True,                    # eixo Y compartilhado → comparação direta
        )
        if n_cols == 1:
            axes = [axes]
 
        fig.suptitle(f"Evolução Temporal: {titulo}", fontsize=13, fontweight="bold")
 
        # Coleta de handles/labels únicos para a legenda compartilhada
        legend_handles = {}
 
        for col, p_inlet in enumerate(pressões_unicas):
            ax = axes[col]
            ax.set_title(f"$P_{{inlet}}$ = {p_inlet:.2g} Pa", fontsize=10)
 
            casos = [r for r in resultados_simulacao
                     if r["configuracao"]["pressao_inlet"] == p_inlet]
 
            # Ordenar para que malhas maiores (e δt menores) fiquem por cima
            casos = sorted(casos,
                           key=lambda r: (r["configuracao"]["N"][0],
                                          r["configuracao"]["dt"]),
                           reverse=False)
 
            for caso in casos:
                N_pts  = caso["configuracao"]["N"][0]
                dt_val = caso["configuracao"]["dt"]
 
                cor    = cores_malha.get(N_pts, "#7f7f7f")
                ls     = estilos_dt.get(dt_val, "-")
                lw     = lw_dt.get(dt_val, 1.2)
                alpha  = alpha_dt.get(dt_val, 0.7)
 
                line, = ax.plot(
                    caso["time"], caso[chave],
                    color=cor, linestyle=ls, linewidth=lw, alpha=alpha,
                )
 
                # Chave única para a legenda: (malha, δt)
                leg_key = (N_pts, dt_val)
                if leg_key not in legend_handles:
                    legend_handles[leg_key] = (
                        line,
                        rf"Malha {N_pts}×{N_pts},  $\delta t={dt_val}$",
                    )
 
            ax.set_xlabel("Tempo Adimensional ($t$)", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
 
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
 
            # Formata eixo Y com notação científica quando os valores forem pequenos
            ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
            )
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
 
        # ── Legenda única, à direita, organizada por malha ──────────────────────
        # Ordena: primeiro por malha, depois por δt
        sorted_keys    = sorted(legend_handles.keys(), key=lambda k: (k[0], k[1]))
        handles_sorted = [legend_handles[k][0] for k in sorted_keys]
        labels_sorted  = [legend_handles[k][1] for k in sorted_keys]
 
        fig.tight_layout()
        # Reserve space on the right for the legend before placing it
        fig.subplots_adjust(right=0.78)
 
        fig.legend(
            handles_sorted, labels_sorted,
            loc="center left",
            bbox_to_anchor=(0.79, 0.5),
            fontsize=8,
            framealpha=0.95,
            edgecolor="#cccccc",
            title="Malha  /  $\\delta t$",
            title_fontsize=8,
            handlelength=2.8,
            borderpad=0.8,
            labelspacing=0.5,
        )
 
        plt.savefig(f"evolucao_{chave}.png", dpi=200, bbox_inches="tight")
        plt.show()
 
    plotar_perfil_membrana_corte(resultados_simulacao, cores_malha, estilos_dt)
 
 
def plotar_perfil_membrana_corte(resultados_simulacao, cores_malha, estilos_dt):
    """
    Plota o perfil transiente de deflexão da membrana ao longo do corte central (Y=0)
    para o caso de maior fidelidade disponível (malha 101×101, menor δt).
    """
 
    pressões = [r["configuracao"]["pressao_inlet"] for r in resultados_simulacao]
    max_p    = max(pressões)
 
    # Tenta encontrar o caso mais refinado; recorre ao primeiro disponível
    caso_fiel = next(
        (r for r in resultados_simulacao
         if r["configuracao"]["N"][0] == 101
         and r["configuracao"]["pressao_inlet"] == max_p
         and r["configuracao"]["dt"] == min(estilos_dt.keys())),
        resultados_simulacao[0],
    )
 
    snapshots = caso_fiel["snapshots_deslocamento"]
    N0, N1    = caso_fiel["configuracao"]["N"]
    x_coords  = np.linspace(-1, 1, N0)
    idx_meio  = N1 // 2
 
    n_frames = min(7, len(snapshots))
    indices  = np.linspace(0, len(snapshots) - 1, n_frames, dtype=int)
 
    # Paleta de cores sequencial (tempo mais escuro = mais tarde)
    cmap   = plt.get_cmap("viridis")
    colors = [cmap(i / (n_frames - 1)) for i in range(n_frames)]
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for color, idx in zip(colors, indices):
        t_atual, deslocamentos = snapshots[idx]
        W_2d    = deslocamentos.reshape((N1, N0))
        perfil  = W_2d[idx_meio, :]
        ax.plot(x_coords, perfil, color=color, linewidth=1.6,
                label=f"$t$ = {t_atual:.2f}")
 
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Perfil Transiente de Deflexão — Corte Central ($y=0$)\n"
        f"$P_{{inlet}}$ = {caso_fiel['configuracao']['pressao_inlet']:.2g} Pa"
        f"  |  Malha {N0}×{N1}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Posição Normalizada ($x/R$)", fontsize=10)
    ax.set_ylabel("Deflexão $w$ (m)", fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
    )
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
 
    # Barra de cores como legenda temporal
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(
                                    vmin=snapshots[indices[0]][0],
                                    vmax=snapshots[indices[-1]][0]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Tempo Adimensional ($t$)", fontsize=9)
 
    fig.tight_layout()
    plt.savefig("evolucao_perfil_membrana.png", dpi=200, bbox_inches="tight")
    plt.show()



class MH_Problema4(MechanicHydraulic):
    def resolver_P4(self, dt=0.0125, tempo_final=12.0):
        print("\n" + "="*50)
        print("INICIANDO PROBLEMA 4: Oscilação livre (1° e 3° Modo)")
        print("="*50)

        for mode in [1, 3]:
        
            #parametros do problema 4
            self.beta = 0.0              
            largura_canal = 2000.0e-6  
            pressao_inlet = 0.0        

            print("Montando sistema global acoplado...")
            sistema = self.montar_sistema_global(dt, largura_canal)

            #Encontrar o 3 modo fundamental da membrana isolada
            print("Calculando modos de vibração da membrana isolada...")
            K = sistema["K"]
            M = sistema["M"]

            v0_fixo = np.ones(K.shape[0])

            eigenvalues, eigenvectors = eigsh(K, k=4, M=M, sigma=0.0, which='LM', v0=v0_fixo)
            
            w_modo = eigenvectors[:, mode-1]

            w_modo = w_modo / np.max(np.abs(w_modo))

            n_m = self.num_nodes_membrana
            n_p = self.num_nodes
            n_steps = int(round(tempo_final / dt))
            idt = 1.0 / dt

            #condição inicial
            w = w_modo.copy()  
            v = np.zeros(n_m)   
            p = np.zeros(n_p)

            solver = factorized(sistema["A_global"].tocsc())
            b_pressao = self.montar_vetor_pressao_inlet(pressao_inlet)

            #no deslocado
            n_x, n_y = self.N
            idx_medicao_x = int(3 * n_x / 4) #3/4 do caminho em X
            idx_medicao_y = int(n_y / 2)     #Metade do caminho em Y
            no_medicao_w = self.ij2n(idx_medicao_y, idx_medicao_x)

            #no central
            idx_centro_x = int(n_x / 2)
            idx_centro_y = int(n_y / 2)
            no_centro_w = self.ij2n(idx_centro_y, idx_centro_x)

            tempos = [0.0]
            historico_w_medicao = [w[no_medicao_w]]
            historico_w_centro = [w[no_centro_w]]
            historico_p_outlet = [0.0]

            print(f"Iniciando integração no tempo ({n_steps} passos)...")
            start = time.time()

            for step in range(1, n_steps + 1):
                
                rhs = np.concatenate([
                    idt * w,
                    idt * (sistema["M"] @ v),
                    b_pressao,
                ])

                solucao = solver(rhs)
                w = solucao[:n_m]
                v = solucao[n_m:2 * n_m]
                p = solucao[2 * n_m:]

                tempos.append(step * dt)
                
                historico_w_medicao.append(w[no_medicao_w])
                historico_w_centro.append(w[no_centro_w]) # NOVO
                historico_p_outlet.append(p[self.node_outlet])

            print(f"Simulação concluida em {time.time() - start:.2f} s.")
            
            tempos = np.array(tempos)
            sinal_w = np.array(historico_w_medicao)
            sinal_w_centro = np.array(historico_w_centro) # NOVO

            picos, _ = find_peaks(sinal_w)

            if len(picos) > 1:
                periodos_adim = np.diff(tempos[picos])
                periodo_sim_adim = np.mean(periodos_adim)
                freq_sim_rad_adim = (2 * np.pi) / periodo_sim_adim
            else:
                freq_sim_rad_adim = 0.0

            raiz_bessel_11 = 3.83170597
            erro_perc = abs(freq_sim_rad_adim - raiz_bessel_11) / raiz_bessel_11 * 100

            #condicao da membrana
            W_2d = w_modo.reshape((n_y, n_x))
            plt.figure(figsize=(6, 5))
            plt.imshow(W_2d, extent=[-1, 1, -1, 1], origin='lower', cmap='seismic')
            plt.colorbar(label='Amplitude Adimensional')
            plt.title(f"Condição Inicial: {mode}º Modo")
            plt.xlabel("x / R")
            plt.ylabel("y / R")
            plt.show()

            #deslocamento no central vs no lateral
            plt.figure(figsize=(10, 4))
            plt.plot(tempos, sinal_w, label='Nó lateral (Fora do eixo)', color='tab:blue')
            plt.plot(tempos, sinal_w_centro, label='Nó central (Sobre a linha nodal)', color='tab:red', linestyle='--')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.title(f"Deslocamento Transiente: Nó Lateral vs. Nó Central\nErro Frequência: {erro_perc:.2f}% (Malha {n_x}x{n_y}) Modo {mode}")
            plt.xlabel("Tempo Adimensional ($t$)")
            plt.ylabel("Deslocamento Adimensional ($\hat{w}$)")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

            #pressao no outlet
            plt.figure(figsize=(10, 4))
            plt.plot(tempos, historico_p_outlet, color='orange', label='Pressão Outlet ($p_{outlet}$)')
            plt.title(f"Pressão no Nó de Descarga. Modo: {mode}")
            plt.xlabel("Tempo Adimensional ($t$)")
            plt.ylabel("Pressão Adimensional ($\hat{p}$)")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

        return freq_sim_rad_adim, raiz_bessel_11
    

class MH_Problema5(MechanicHydraulic):
    def resolver_P5(self, dt=0.0125, tempo_final=12.0):

        print("\n" + "="*50)
        print("INICIANDO PROBLEMA 5: Oscilação forçada (1° e 3° Modo)")
        print("="*50)

        for mode in [1, 3]:

            #parametros do problema 4 e 5
            self.beta = 0.0              
            largura_canal = 2000.0e-6  
            pressao_inlet = 5000      

            print("Montando sistema global acoplado...")
            sistema = self.montar_sistema_global(dt, largura_canal)

            #Encontrar o 3 modo fundamental da membrana isolada
            print("Calculando modos de vibração da membrana isolada...")
            K = sistema["K"]
            M = sistema["M"]

            v0_fixo = np.ones(K.shape[0])

            eigenvalues, eigenvectors = eigsh(K, k=4, M=M, sigma=0.0, which='LM', v0=v0_fixo)
            
            w_modo = eigenvectors[:, mode-1]

            omega_3 = np.sqrt(eigenvalues[mode-1])

            w_modo = w_modo / np.max(np.abs(w_modo))

            n_m = self.num_nodes_membrana
            n_p = self.num_nodes
            n_steps = int(round(tempo_final / dt))
            idt = 1.0 / dt

            #condição inicial
            w = w_modo.copy()  
            v = np.zeros(n_m)   
            p = np.zeros(n_p)

            solver = factorized(sistema["A_global"].tocsc())
            b_pressao = self.montar_vetor_pressao_inlet(pressao_inlet)

            #no deslocado
            n_x, n_y = self.N
            idx_medicao_x = int(3 * n_x / 4) #3/4 do caminho em X
            idx_medicao_y = int(n_y / 2)     #Metade do caminho em Y
            no_medicao_w = self.ij2n(idx_medicao_y, idx_medicao_x)

            #no central
            idx_centro_x = int(n_x / 2)
            idx_centro_y = int(n_y / 2)
            no_centro_w = self.ij2n(idx_centro_y, idx_centro_x)

            tempos = [0.0]
            historico_w_medicao = [w[no_medicao_w]]
            historico_w_centro = [w[no_centro_w]]
            historico_p_outlet = [0.0]

            print(f"Iniciando integração no tempo ({n_steps} passos)...")
            start = time.time()

            for step in range(1, n_steps + 1):

                b_forced = b_pressao * np.cos(omega_3 * (step-1) * dt)
                
                rhs = np.concatenate([
                    idt * w,
                    idt * (sistema["M"] @ v),
                    b_forced,
                ])

                solucao = solver(rhs)
                w = solucao[:n_m]
                v = solucao[n_m:2 * n_m]
                p = solucao[2 * n_m:]

                tempos.append(step * dt)
                
                historico_w_medicao.append(w[no_medicao_w])
                historico_w_centro.append(w[no_centro_w]) # NOVO
                historico_p_outlet.append(p[self.node_outlet])

            print(f"Simulação concluida em {time.time() - start:.2f} s.")
            
            tempos = np.array(tempos)
            sinal_w = np.array(historico_w_medicao)
            sinal_w_centro = np.array(historico_w_centro) # NOVO

            picos, _ = find_peaks(sinal_w)

            if len(picos) > 1:
                periodos_adim = np.diff(tempos[picos])
                periodo_sim_adim = np.mean(periodos_adim)
                freq_sim_rad_adim = (2 * np.pi) / periodo_sim_adim
            else:
                freq_sim_rad_adim = 0.0

            raiz_bessel_11 = 3.83170597
            erro_perc = abs(freq_sim_rad_adim - raiz_bessel_11) / raiz_bessel_11 * 100

            #condicao da membrana
            W_2d = w_modo.reshape((n_y, n_x))
            plt.figure(figsize=(6, 5))
            plt.imshow(W_2d, extent=[-1, 1, -1, 1], origin='lower', cmap='seismic')
            plt.colorbar(label='Amplitude Adimensional')
            plt.title(f"Condição Inicial: {mode}º Modo")
            plt.xlabel("x / R")
            plt.ylabel("y / R")
            plt.show()

            #deslocamento no central vs no lateral
            plt.figure(figsize=(10, 4))
            plt.plot(tempos, sinal_w, label='Nó lateral (Fora do eixo)', color='tab:blue')
            plt.plot(tempos, sinal_w_centro, label='Nó central (Sobre a linha nodal)', color='tab:red', linestyle='--')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.title(f"Deslocamento Transiente: Nó Lateral vs. Nó Central\nErro Frequência: {erro_perc:.2f}% (Malha {n_x}x{n_y}) Modo: {mode}")
            plt.xlabel("Tempo Adimensional ($t$)")
            plt.ylabel("Deslocamento Adimensional ($\hat{w}$)")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

            #pressao no outlet
            plt.figure(figsize=(10, 4))
            plt.plot(tempos, historico_p_outlet, color='orange', label='Pressão Outlet ($p_{outlet}$)')
            plt.title(f"Pressão no Nó de Descarga. Modo: {mode}")
            plt.xlabel("Tempo Adimensional ($t$)")
            plt.ylabel("Pressão Adimensional ($\hat{p}$)")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()

        return freq_sim_rad_adim, raiz_bessel_11

