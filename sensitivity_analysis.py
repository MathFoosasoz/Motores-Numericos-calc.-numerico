import time
import numpy as np
import matplotlib.pyplot as plt

import env
from mechanic_hydraulic import MechanicHydraulic


class prob_base_3_GD:
    """
    Classe base que agrupa funções utilitárias e a chamada de simulação 
    compartilhadas pelos diferentes problemas.
    """
    def __init__(self, config):
        self.config = config.copy()
        
        self.STYLE = {
            "forward":   {"color": "#e6194b", "ls": "--",  "lw": 1.8, "label": "Forward (1ª ordem)"},
            "centered":  {"color": "#3cb44b", "ls": "-.",  "lw": 2.0, "label": "Centrada (2ª ordem)"},
            "analytical":{"color": "#4363d8", "ls": "-",   "lw": 2.2, "label": "Analítica (Regra da Cadeia)"},
        }

    def viscosity_from_T(self, T_celsius):
        return 0.001791 / (1.0 + 0.03368 * T_celsius + 0.000221 * T_celsius ** 2)

    def simular(self, Tc, H):
        mu = self.viscosity_from_T(Tc)

        config_sim = self.config.copy()
        config_sim["VISCOSITY"] = mu
        config_sim["CHANNEL_WIDTH"] = H

        sim = MechanicHydraulic(config_sim)

        resultado = sim.resolver_caso_base(
            N=(51, 51),
            dt=0.025,
            tempo_final=config_sim["TIME_END"],
            pressao_inlet=config_sim["INLET_PRESSURE"],
            largura_canal=H,
            print_info=False,
        )

        A_fisico = sim.sistema_atual["A_fisico"]
        pressoes_finais = resultado["pressoes_finais"]

        E = float(resultado["potencia"][-1])
        q_inlet = float((A_fisico @ pressoes_finais)[sim.node_inlet])
        V_tf = float(resultado["volume_reservatorio"][-1])

        return E, q_inlet, V_tf


class P1_3_GD(prob_base_3_GD):
    """
    Implementação do Problema 1: Análise Numérica de Sensibilidade
    """
    def __init__(self, config):
        super().__init__(config)
        
    def sensibilidade_analitica_H(self, E, q_inlet, V_tf, H):
        fator = 4.0 / H
        dE_dH  = fator * E
        dq_dH  = fator * q_inlet
        dV_dH  = 0.0
        return np.array([dE_dH, dq_dH, dV_dH])

    def resolver_P1(self, Tc_nom=25.0, H_nom=1000.0e-6, dt_c=1e-3, dh=1e-6, n_pontos=50):
        print("=" * 65)
        print("  PROBLEMA 1: ANÁLISE NUMÉRICA DE SENSIBILIDADE")
        print("=" * 65)
        print(f"  Parâmetros: Tc = {Tc_nom} °C, H = {H_nom*1e6:.0f} µm")
        print(f"  Malha: {n_pontos} (Tc) × {n_pontos} (H) | ΔH = {dh:.0e}")
        print("-" * 65)

        Tc_array = np.linspace(0.0, 250.0, n_pontos)
        H_array  = np.linspace(500.0e-6, 1500.0e-6, n_pontos)

        print("[1/4] Varredura de T_C (H fixo)…")
        E_vs_Tc, q_vs_Tc, V_vs_Tc = np.zeros(n_pontos), np.zeros(n_pontos), np.zeros(n_pontos)
        for i, Tc in enumerate(Tc_array):
            E, q, V = self.simular(Tc, H_nom)
            E_vs_Tc[i], q_vs_Tc[i], V_vs_Tc[i] = E, q, V

        print("[2/4] Varredura de H (Tc fixo)…")
        E_vs_H, q_vs_H, V_vs_H = np.zeros(n_pontos), np.zeros(n_pontos), np.zeros(n_pontos)
        for i, H in enumerate(H_array):
            E, q, V = self.simular(Tc_nom, H)
            E_vs_H[i], q_vs_H[i], V_vs_H[i] = E, q, V

        print("[3/4] Calculando diferenças finitas em H…")
        E_pos, q_pos, V_pos = np.zeros(n_pontos), np.zeros(n_pontos), np.zeros(n_pontos)
        E_neg, q_neg, V_neg = np.zeros(n_pontos), np.zeros(n_pontos), np.zeros(n_pontos)
        for i, H in enumerate(H_array):
            E_pos[i], q_pos[i], V_pos[i] = self.simular(Tc_nom, H + dh)
            E_neg[i], q_neg[i], V_neg[i] = self.simular(Tc_nom, H - dh)

        sens_forward = np.column_stack([
            (E_pos - E_vs_H) / dh,
            (q_pos - q_vs_H) / dh,
            (V_pos - V_vs_H) / dh,
        ])

        sens_centered = np.column_stack([
            (E_pos - E_neg) / (2.0 * dh),
            (q_pos - q_neg) / (2.0 * dh),
            (V_pos - V_neg) / (2.0 * dh),
        ])

        print("[4/4] Calculando sensibilidade analítica…")
        sens_analytical = np.zeros((n_pontos, 3))
        for i, H in enumerate(H_array):
            sens_analytical[i] = self.sensibilidade_analitica_H(E_vs_H[i], q_vs_H[i], V_vs_H[i], H)

        print("\nGerando gráficos do Problema 1…")
        self.plot_comportamento(Tc_array, H_array, H_nom, Tc_nom, E_vs_Tc, q_vs_Tc, V_vs_Tc, E_vs_H, q_vs_H, V_vs_H)
        self.plot_sensibilidade_H(H_array, Tc_nom, dh, sens_forward, sens_centered, sens_analytical)
        self.plot_erro_relativo_H(H_array, Tc_nom, dh, sens_forward, sens_centered, sens_analytical)
        print("✓ Problema 1 finalizado.\n")

    def plot_comportamento(self, Tc_arr, H_arr, H_nom, Tc_nom, E_vs_Tc, q_vs_Tc, V_vs_Tc, E_vs_H, q_vs_H, V_vs_H):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            "Comportamento das Saídas do Gêmeo Digital Microfluídico\n"
            f"(malha 51×51, $\\Delta t=0.025$, $t_f={self.config['TIME_END']}$ s)",
            fontsize=14, fontweight="bold", y=1.01
        )
        datasets_Tc = [(E_vs_Tc, "Potência  $E$ (W)", "#c0392b"), (q_vs_Tc, "Vazão $q_{inlet}$ (m³/s)", "#2980b9"), (V_vs_Tc, "Volume  $V(t_f)$ (m³)", "#27ae60")]
        datasets_H  = [(E_vs_H, "Potência  $E$ (W)", "#c0392b"), (q_vs_H, "Vazão $q_{inlet}$ (m³/s)", "#2980b9"), (V_vs_H, "Volume  $V(t_f)$ (m³)", "#27ae60")]

        for col, (data, ylabel, color) in enumerate(datasets_Tc):
            axes[0, col].plot(Tc_arr, data, color=color, lw=2)
            axes[0, col].set_xlabel("$T_C$ (°C)")
            axes[0, col].set_ylabel(ylabel)
            axes[0, col].set_title(f"$H = {H_nom*1e6:.0f}\\,\\mu$m (fixo)")
            axes[0, col].grid(True, linestyle=":", alpha=0.5)

        for col, (data, ylabel, color) in enumerate(datasets_H):
            axes[1, col].plot(H_arr * 1e6, data, color=color, lw=2)
            axes[1, col].set_xlabel("$H$ ($\\mu$m)")
            axes[1, col].set_ylabel(ylabel)
            axes[1, col].set_title(f"$T_C = {Tc_nom:.0f}$ °C (fixo)")
            axes[1, col].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig("sens_comportamento.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.show(block=False) 

    def plot_sensibilidade_H(self, H_arr, Tc_nom, dh, fwd, cen, ana):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Análise Comparativa de Sensibilidade: $\\partial \\cdot / \\partial H$\n"
            f"($T_C = {Tc_nom:.0f}$ °C fixo,  $\\Delta H = {dh:.0e}$ m)",
            fontsize=13, fontweight="bold"
        )
        labels = [("E", "$\\partial E / \\partial H$  (W/m)"), 
                  ("q_{inlet}", "$\\partial q_{inlet} / \\partial H$  (m³/s / m)"), 
                  ("V(t_f)", "$\\partial V(t_f) / \\partial H$  (m³/m)")]

        for col, (name, ylabel) in enumerate(labels):
            axes[col].plot(H_arr * 1e6, fwd[:, col], **self.STYLE["forward"])
            axes[col].plot(H_arr * 1e6, cen[:, col], **self.STYLE["centered"])
            axes[col].plot(H_arr * 1e6, ana[:, col], **self.STYLE["analytical"])
            axes[col].set_xlabel("$H$ ($\\mu$m)")
            axes[col].set_ylabel(ylabel)
            axes[col].set_title(f"$\\partial {name} / \\partial H$")
            axes[col].legend()
            axes[col].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig("sens_comparativo_H.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.show(block=False)

    def plot_erro_relativo_H(self, H_arr, Tc_nom, dh, fwd, cen, ana):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Erro Relativo das Diferenças Finitas vs. Solução Analítica\n"
            f"($T_C = {Tc_nom:.0f}$ °C fixo,  $\\Delta H = {dh:.0e}$ m)",
            fontsize=13, fontweight="bold"
        )
        names = ["E", "q_{inlet}", "V(t_f)"]

        for col, name in enumerate(names):
            safe_ana = np.where(np.abs(ana[:, col]) > 1e-30, ana[:, col], 1e-30)
            err_fwd = np.abs((fwd[:, col] - ana[:, col]) / safe_ana) * 100.0
            err_cen = np.abs((cen[:, col] - ana[:, col]) / safe_ana) * 100.0

            axes[col].semilogy(H_arr * 1e6, np.clip(err_fwd, 1e-10, None), **self.STYLE["forward"])
            axes[col].semilogy(H_arr * 1e6, np.clip(err_cen, 1e-10, None), **self.STYLE["centered"])
            axes[col].set_xlabel("$H$ ($\\mu$m)")
            axes[col].set_ylabel("Erro relativo (%)")
            axes[col].set_title(f"Erro em $\\partial {name} / \\partial H$")
            axes[col].legend()
            axes[col].grid(True, which="both", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig("sens_erro_relativo_H.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.show(block=False)


class P2_3_GD(prob_base_3_GD):

    def __init__(self, config):
        super().__init__(config)

    def resolver_P2(self, E_target=7.5, H0=1000.0e-6, Tc=25.0, tol=1e-6, max_iter=50):
        print("=" * 65)
        print("  PROBLEMA 2: OTIMIZAÇÃO VIA NEWTON-RAPHSON")
        print("=" * 65)
        
        R = self.config["R"]
        sigma = self.config["TENSION"]
        rho = self.config["DENSITY"]
        e = self.config["THICKNESS"]
        w0_factor = self.config["W0_FACTOR"]

        w_ref = w0_factor * R
        t_ref = R * np.sqrt((rho * e) / sigma)
        v_ref = w_ref / t_ref
       
        E_ref = sigma * w_ref * v_ref
        
        print(f"  [Info] Energia de referência calculada (E_ref): {E_ref:.4e} W")
        
        H_min, H_max = 500.0e-6, 20
        if not (H_min <= H0 <= H_max):
            print(f"  ! Aviso: Chute inicial {H0*1e6:.1f} µm fora do domínio real [500, 1500] µm.")
            print(f"    Forçando início dentro do domínio (1000 µm) para evitar divergência.")
            H0 = 1000.0e-6

        H_k = H0
        historico_H = []
        historico_E = []

        print(f"  Iniciando: Alvo Adimensional E = {E_target}, H inicial = {H0*1e6:.1f} µm, Tc = {Tc} °C")
        
        for i in range(max_iter):
            
            E_k_watts, q_k, V_k = self.simular(Tc, H_k)
            
            E_k_adim = E_k_watts / E_ref
            
            f_k = E_k_adim - E_target
            
            historico_H.append(H_k)
            historico_E.append(E_k_adim)
            
            print(f"    Iter {i:02d}: H = {H_k*1e6:>8.3f} µm | E(H)_adim = {E_k_adim:>8.4f} | Erro f(H) = {f_k:>9.2e}")

            if abs(f_k) < tol:
                print(f"  ✓ Convergiu em {i} iterações. H_ótimo = {H_k*1e6:.3f} µm")
                self.plot_newton_raphson(historico_H, historico_E, E_target)
                return H_k, historico_H, historico_E

            dE_dH = (4.0 / H_k) * E_k_adim
            
            delta_H = f_k / dE_dH
            
            passo_maximo = 0.20 * H_k 
            delta_H = np.clip(delta_H, -passo_maximo, passo_maximo)

            H_k = H_k - delta_H
            H_k = np.clip(H_k, H_min, H_max)
            
            if (H_k >= H_max and f_k < -0.1) or (H_k <= H_min and f_k > 0.1):
                print(f"  ! ERRO: Alvo E={E_target} inatingível no domínio H=[{H_min*1e6}, {H_max*1e6}].")
                print(f"    Energia máxima atingida no limite: {E_k_adim:.6f}")
                break

        print("  ! Aviso: Newton-Raphson não convergiu no número máximo de iterações.")
        self.plot_newton_raphson(historico_H, historico_E, E_target)
        return H_k, historico_H, historico_E

    def plot_newton_raphson(self, historico_H, historico_E, E_target):
        iteracoes = np.arange(len(historico_H))
        H_arr = np.array(historico_H) * 1e6

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Convergência do Método de Newton-Raphson", fontsize=14, fontweight="bold")

        axes[0].plot(iteracoes, H_arr, marker='o', color='#2980b9', lw=2)
        axes[0].set_xlabel("Iteração")
        axes[0].set_ylabel("$H$ ($\\mu$m)")
        axes[0].set_title("Evolução da Largura do Canal ($H$)")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        label_alvo = "Alvo ($\\mathcal{E}$ = " + str(E_target) + ")"

        axes[1].plot(iteracoes, historico_E, marker='s', color='#c0392b', lw=2, label="$\\mathcal{E}(H_k)$ calculada")
        axes[1].axhline(E_target, color='k', linestyle='--', label=label_alvo)
        axes[1].set_xlabel("Iteração")
        axes[1].set_ylabel("Energia Adimensional $\\mathcal{E}$")
        axes[1].set_title("Evolução do Funcional de Energia")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig("newton_raphson_convergencia.png", dpi=150, bbox_inches="tight")
        print(f"  → Gráfico salvo: newton_raphson_convergencia.png\n")
        
        plt.show()