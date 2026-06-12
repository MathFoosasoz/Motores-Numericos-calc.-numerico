import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import factorized

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

        pressões_unicas = sorted(list(set(res["configuracao"]["pressao_inlet"] for res in resultados_simulacao)))
        canais_unicos = sorted(list(set(res["configuracao"]["largura_canal"] for res in resultados_simulacao)))
        
        grandezas = [
            ("deslocamento_centro", "Deslocamento Vertical do Ponto Central", "m"),
            ("pressao_outlet", "Pressão no Nó de Descarga $p_{{outlet}}$", "Pa"),
            ("vazao_outlet", "Vazão de Saída $q_{{outlet}}$", "$m^3/s$"),
            ("volume_reservatorio", "Volume Acumulado de Fluido no Reservatório", "$m^3$"),
            ("potencia", "Potência Consumida pelo Sistema", "W")
        ]
        
        estilos_dt = {0.00625: '-', 0.0125: '--', 0.025: ':', 0.05: '-.'}
        cores_malha = {51: 'tab:blue', 101: 'tab:orange', 21: 'tab:green'} 

        for chave, titulo, unidade in grandezas:
            for p_inlet in pressões_unicas:

                fig, eixos = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
  
                eixos_flat = eixos.flatten()

                for idx_canal, largura_w in enumerate(canais_unicos[:4]):
                    ax = eixos_flat[idx_canal]
                    
                    casos_foco = [
                        r for r in resultados_simulacao 
                        if r["configuracao"]["pressao_inlet"] == p_inlet 
                        and r["configuracao"]["largura_canal"] == largura_w
                    ]
                    
                    if not casos_foco:
                        ax.text(0.5, 0.5, 'Sem dados simulados', ha='center', va='center')
                        ax.set_title(rf"Canal = {largura_w*1e6:.0f} $\mu$m")
                        continue

                    for caso in casos_foco:
                        N_points = caso["configuracao"]["N"][0]
                        dt_val = caso["configuracao"]["dt"]
                        
                        label_curva = rf"Malha {N_points}x{N_points}, $\delta t$={dt_val}"
                        ax.plot(caso["time"], caso[chave], label=label_curva, 
                                color=cores_malha.get(N_points, 'black'), 
                                linestyle=estilos_dt.get(dt_val, '-'))

                    ax.set_title(rf"Largura do Canal = {largura_w*1e6:.0f} $\mu$m", fontsize=10, fontweight='bold')
                    ax.set_ylabel(f"{chave.replace('_', ' ').title()} ({unidade})", fontsize=9)
                    ax.grid(True, linestyle=':', alpha=0.5)

                    if idx_canal >= 2:
                        ax.set_xlabel("Tempo Adimensional ($t$)", fontsize=9)

                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

                plt.suptitle(f"{titulo}\nAnálise para $P_{{inlet}}$ = {p_inlet:.1e} Pa", 
                             fontsize=14, fontweight='bold', y=0.98)
                
                plt.tight_layout()

                nome_arquivo = f"{chave}_P_{p_inlet:.1e}.png"
                plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
                plt.show()

        plotar_perfil_membrana_corte(resultados_simulacao, cores_malha, estilos_dt)


def plotar_perfil_membrana_corte(resultados_simulacao, cores_malha, estilos_dt):
        
        pressões_unicas = sorted(list(set(res["configuracao"]["pressao_inlet"] for res in resultados_simulacao)))
        canais_unicos = sorted(list(set(res["configuracao"]["largura_canal"] for res in resultados_simulacao)))

        for p_inlet in pressões_unicas:

            fig, eixos = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
            eixos_flat = eixos.flatten()

            for idx_canal, largura_w in enumerate(canais_unicos[:4]):
                ax = eixos_flat[idx_canal]

                caso_fiel = None
                for r in resultados_simulacao:
                    cfg = r["configuracao"]
                    if (cfg["pressao_inlet"] == p_inlet and 
                        cfg["largura_canal"] == largura_w and 
                        cfg["N"][0] == 101 and 
                        cfg["dt"] == 0.00625):
                        caso_fiel = r
                        break

                if caso_fiel is None:
                    casos_alternativos = [
                        r for r in resultados_simulacao 
                        if r["configuracao"]["pressao_inlet"] == p_inlet 
                        and r["configuracao"]["largura_canal"] == largura_w
                    ]
                    if alternative_cases := casos_alternativos:
                        caso_fiel = alternative_cases[0]
                    else:
                        ax.text(0.5, 0.5, 'Sem dados simulados', ha='center', va='center')
                        ax.set_title(rf"Canal = {largura_w*1e6:.0f} $\mu$m")
                        continue 

                snapshots = caso_fiel["snapshots_deslocamento"]
                N0, N1 = caso_fiel["configuracao"]["N"]
                x_coordenadas = np.linspace(-1, 1, N0)
                idx_linha_central = N1 // 2

                indices_snapshots = np.linspace(0, len(snapshots) - 1, 6, dtype=int)
                for idx in indices_snapshots:
                    t_atual, deslocamentos_globais = snapshots[idx]
                    W_2d = deslocamentos_globais.reshape((N1, N0))
                    ax.plot(x_coordenadas, W_2d[idx_linha_central, :], label=f"Tempo $t$ = {t_atual:.3f}")

                ax.set_title(rf"Largura do Canal = {largura_w*1e6:.0f} $\mu$m", fontsize=10, fontweight='bold')
                ax.set_ylabel("Deflexão Real $w$ (m)", fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.5)
                
                if idx_canal >= 2:
                    ax.set_xlabel("Posição Normalizada na Membrana ($x/R$)", fontsize=9)

                ax.legend(loc='lower center', fontsize=8, ncol=2)

            plt.suptitle(f"Perfil Transiente de Deflexão da Membrana (Corte Central Y=0)\n"
                         f"Análise de Canais para $P_{{inlet}}$ = {p_inlet:.1e} Pa (Malha 101x101, $\delta t$=0.00625)", 
                         fontsize=13, fontweight='bold', y=0.98)
            
            plt.tight_layout()

            nome_arquivo = f"perfil_membrana_P_{p_inlet:.1e}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.show()