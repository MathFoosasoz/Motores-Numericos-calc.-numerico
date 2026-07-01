import time
import numpy as np
import matplotlib.pyplot as plt

import env
from mechanic_hydraulic import MechanicHydraulic

class prob_base_3_GD:
    """
    Classe base que agrupa funções utilitárias e a chamada de simulação.
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

        tempos = np.array(resultado["time"])
        potencias = np.array(resultado["potencia"])
        E_fisico_joules = float(np.trapezoid(potencias, tempos))

        q_inlet_fisico = float((A_fisico @ pressoes_finais)[sim.node_inlet])
        V_tf = float(resultado["volume_reservatorio"][-1])

        escalas = sim.sistema_atual["escalas"]

        q_ref = escalas["v_ref"] * (self.config["R"] ** 2)
        pot_ref = escalas["p_ref"] * q_ref
        t_ref = escalas["t_ref"]

        t_adim = tempos / t_ref
        pot_adim = potencias / pot_ref
        E_adim_integral = float(np.trapezoid(pot_adim, t_adim))
        print("t_adim[0], t_adim[-1]:")
        print(t_adim[0], t_adim[-1])
        print("np.min(pot_adim), np.max(pot_adim):")
        print(np.min(pot_adim), np.max(pot_adim))
        print("E_adim_integral:")
        print(E_adim_integral)
        
        pot = resultado["potencia"]
        t = resultado["time"]

        i = np.argmax(pot > 1e-6)

        print("Primeira potência diferente de zero:")
        print("tempo =", t[i])
        print("pot =", pot[i])

        q_inlet_adim = q_inlet_fisico / q_ref

        p_inlet_adim = config_sim["INLET_PRESSURE"] / escalas["p_ref"]
        pot_inlet_adim = p_inlet_adim * q_inlet_adim
        
        funcionais = {
            "energia_total_adim": E_adim_integral,
            "vazao_inlet_adim": q_inlet_adim,
            "potencia_inlet_adim": pot_inlet_adim
        }

        return E_fisico_joules, q_inlet_fisico, V_tf, escalas, funcionais


class P2_3_GD(prob_base_3_GD):

    def __init__(self, config):
        super().__init__(config)

    def resolver_P2(self, E_target=7.5, funcional_alvo="energia_total_adim", H0=1000.0e-6, Tc=25.0, tol=1e-5, max_iter=50):
        print("=" * 65)
        print("PROBLEMA 2: OTIMIZAÇÃO VIA NEWTON-RAPHSON NUMÉRICO")
        print("=" * 65)
        
        H_min, H_max = 500.0e-6, 5000.0e-5
        if not (H_min <= H0 <= H_max):
            H0 = 1000.0e-6

        H_k = H0
        historico_H = []
        historico_E = []
        dh_num = 1e-6 

        print(f"  Iniciando: Alvo = {E_target}, H inicial = {H0*1e6:.1f} µm, Tc = {Tc} °C")
        
        for i in range(max_iter):

            E_fisico_joules, q_k, V_k, escalas, funcionais = self.simular(Tc, H_k)
            
            if i == 0:
                print("\n  [DIAGNÓSTICO] Valores Adimensionais na Iteração 0:")
                for chave, valor in funcionais.items():
                    print(f"    -> {chave} = {valor:.4f}")
                print(f"  [Info] Otimizando ativamente para o funcional: '{funcional_alvo}'\n")

            valor_k = funcionais[funcional_alvo]
            f_k = valor_k - E_target
            
            historico_H.append(H_k)
            historico_E.append(valor_k)
            
            print(f"    Iter {i:02d}: H = {H_k*1e6:>8.3f} µm | Valor Atual = {valor_k:>8.4f} | Erro = {f_k:>9.2e}")

            if abs(f_k) < tol:
                print(f"Convergiu em {i} iterações. H_ótimo = {H_k*1e6:.3f} µm")
                self.plot_newton_raphson(historico_H, historico_E, E_target, funcional_alvo)
                return H_k, historico_H, historico_E

            _, _, _, _, func_dh = self.simular(Tc, H_k + dh_num)
            valor_dh = func_dh[funcional_alvo]
            
            dE_dH = (valor_dh - valor_k) / dh_num
            
            if dE_dH == 0:
                print("  ! Erro: Gradiente nulo alcançado.")
                break

            delta_H = f_k / dE_dH
            
            passo_maximo = 0.20 * H_k 
            delta_H = np.clip(delta_H, -passo_maximo, passo_maximo)

            H_k = H_k - delta_H
            H_k = np.clip(H_k, H_min, H_max)
            
            if (H_k >= H_max and f_k < -0.1) or (H_k <= H_min and f_k > 0.1):
                print(f"  ! ERRO: Alvo {E_target} possivelmente inatingível no domínio restrito.")
                break

        print("  ! Aviso: Newton-Raphson finalizou o número de iterações.")
        self.plot_newton_raphson(historico_H, historico_E, E_target, funcional_alvo)
        return H_k, historico_H, historico_E

    def plot_newton_raphson(self, historico_H, historico_E, E_target, funcional_alvo):
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

        label_alvo = f"Alvo ({E_target})"

        axes[1].plot(iteracoes, historico_E, marker='s', color='#c0392b', lw=2, label=f"{funcional_alvo}")
        axes[1].axhline(E_target, color='k', linestyle='--', label=label_alvo)
        axes[1].set_xlabel("Iteração")
        axes[1].set_ylabel("Valor Adimensional")
        axes[1].set_title(f"Evolução: {funcional_alvo}")
        axes[1].legend()
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig("newton_raphson_convergencia.png", dpi=150, bbox_inches="tight")
        print(f"  → Gráfico salvo: newton_raphson_convergencia.png\n")
        plt.show()