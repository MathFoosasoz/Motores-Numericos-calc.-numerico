import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import env
from mechanic_hydraulic import MechanicHydraulic


def viscosity_from_T(T_celsius):
    return 0.001791 / (1.0 + 0.03368 * T_celsius + 0.000221 * T_celsius ** 2)


def dmu_dT(T_celsius):
    num = 0.001791
    denom = 1.0 + 0.03368 * T_celsius + 0.000221 * T_celsius ** 2
    return -num * (0.03368 + 2.0 * 0.000221 * T_celsius) / denom ** 2


def simular(Tc, H):
    mu = viscosity_from_T(Tc)

    config = env.CONFIG_MH.copy()
    config["VISCOSITY"] = mu
    config["CHANNEL_WIDTH"] = H

    sim = MechanicHydraulic(config)

    resultado = sim.resolver_caso_base(
        N=(51, 51),
        dt=0.025,
        tempo_final=config["TIME_END"],
        pressao_inlet=config["INLET_PRESSURE"],
        largura_canal=H,
        print_info=False,
    )

    A_fisico = sim.sistema_atual["A_fisico"]
    pressoes_finais = resultado["pressoes_finais"]

    E = float(resultado["potencia"][-1])
    q_inlet = float((A_fisico @ pressoes_finais)[sim.node_inlet])
    V_tf = float(resultado["volume_reservatorio"][-1])

    return E, q_inlet, V_tf


def sensibilidade_forward_Tc(Tc, H, dTc=1e-3):
    f0 = np.array(simular(Tc,       H))
    f1 = np.array(simular(Tc + dTc, H))
    return (f1 - f0) / dTc


def sensibilidade_forward_H(Tc, H, dH=1e-6):
    f0 = np.array(simular(Tc, H))
    f1 = np.array(simular(Tc, H + dH))
    return (f1 - f0) / dH


def sensibilidade_centrada_H(Tc, H, dH=1e-6):
    f_pos = np.array(simular(Tc, H + dH))
    f_neg = np.array(simular(Tc, H - dH))
    return (f_pos - f_neg) / (2.0 * dH)


def sensibilidade_analitica_H(E, q_inlet, V_tf, H):
    fator = 4.0 / H
    dE_dH  = fator * E
    dq_dH  = fator * q_inlet
    dV_dH  = 0.0
    return np.array([dE_dH, dq_dH, dV_dH])


N_PONTOS = 50
DT_C = 1e-3
DH   = 1e-6

TC_NOM = 25.0
H_NOM  = 1000.0e-6

Tc_array = np.linspace(0.0, 250.0,         N_PONTOS)
H_array  = np.linspace(500.0e-6, 1500.0e-6, N_PONTOS)


STYLE = {
    "forward":   {"color": "#e6194b", "ls": "--",  "lw": 1.8, "label": "Forward (1ª ordem)"},
    "centered":  {"color": "#3cb44b", "ls": "-.",  "lw": 2.0, "label": "Centrada (2ª ordem)"},
    "analytical":{"color": "#4363d8", "ls": "-",   "lw": 2.2, "label": "Analítica (Regra da Cadeia)"},
}

OUTPUT_LABELS = {
    "E":       ("Potência dissipada $E$ (W)",          "E"),
    "q_inlet": ("Vazão de entrada $q_{inlet}$ (m³/s)", "q_{inlet}"),
    "V_tf":    ("Volume final $V(t_f)$ (m³)",          "V(t_f)"),
}


def plot_comportamento(Tc_arr, H_arr,
                       E_vs_Tc, q_vs_Tc, V_vs_Tc,
                       E_vs_H,  q_vs_H,  V_vs_H,
                       filename="sens_comportamento.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Comportamento das Saídas do Gêmeo Digital Microfluídico\n"
        f"(malha 51×51, $\\Delta t=0.025$, $t_f={env.CONFIG_MH['TIME_END']}$ s)",
        fontsize=14, fontweight="bold", y=1.01
    )

    datasets_Tc = [
        (E_vs_Tc, "Potência  $E$ (W)",           "#c0392b"),
        (q_vs_Tc, "Vazão $q_{inlet}$ (m³/s)",    "#2980b9"),
        (V_vs_Tc, "Volume  $V(t_f)$ (m³)",       "#27ae60"),
    ]
    datasets_H = [
        (E_vs_H,  "Potência  $E$ (W)",            "#c0392b"),
        (q_vs_H,  "Vazão $q_{inlet}$ (m³/s)",     "#2980b9"),
        (V_vs_H,  "Volume  $V(t_f)$ (m³)",        "#27ae60"),
    ]

    H_plot = H_arr * 1e6

    for col, (data, ylabel, color) in enumerate(datasets_Tc):
        ax = axes[0, col]
        ax.plot(Tc_arr, data, color=color, lw=2)
        ax.set_xlabel("$T_C$ (°C)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"$H = {H_NOM*1e6:.0f}\\,\\mu$m (fixo)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    for col, (data, ylabel, color) in enumerate(datasets_H):
        ax = axes[1, col]
        ax.plot(H_plot, data, color=color, lw=2)
        ax.set_xlabel("$H$ ($\\mu$m)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"$T_C = {TC_NOM:.0f}$ °C (fixo)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvo: {filename}")


def plot_sensibilidade_H(H_arr,
                         sens_forward,
                         sens_centered,
                         sens_analytical,
                         filename="sens_comparativo_H.png"):
    output_names = ["E", "q_{inlet}", "V(t_f)"]
    output_ylabels = [
        "$\\partial E / \\partial H$  (W/m)",
        "$\\partial q_{inlet} / \\partial H$  (m³/s / m)",
        "$\\partial V(t_f) / \\partial H$  (m³/m)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Análise Comparativa de Sensibilidade: $\\partial \\cdot / \\partial H$\n"
        f"($T_C = {TC_NOM:.0f}$ °C fixo,  $\\Delta H = {DH:.0e}$ m)",
        fontsize=13, fontweight="bold"
    )

    H_plot = H_arr * 1e6

    for col in range(3):
        ax = axes[col]

        ax.plot(H_plot, sens_forward[:, col],   **{k: v for k, v in STYLE["forward"].items()})
        ax.plot(H_plot, sens_centered[:, col],  **{k: v for k, v in STYLE["centered"].items()})
        ax.plot(H_plot, sens_analytical[:, col],**{k: v for k, v in STYLE["analytical"].items()})

        ax.set_xlabel("$H$ ($\\mu$m)", fontsize=11)
        ax.set_ylabel(output_ylabels[col], fontsize=10)
        ax.set_title(f"$\\partial {output_names[col]} / \\partial H$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvo: {filename}")


def plot_erro_relativo_H(H_arr, sens_forward, sens_centered, sens_analytical,
                         filename="sens_erro_relativo_H.png"):
    output_names = ["$E$", "$q_{inlet}$", "$V(t_f)$"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Erro Relativo das Diferenças Finitas vs. Solução Analítica\n"
        f"($T_C = {TC_NOM:.0f}$ °C fixo,  $\\Delta H = {DH:.0e}$ m)",
        fontsize=13, fontweight="bold"
    )

    H_plot = H_arr * 1e6

    for col in range(3):
        ax = axes[col]

        ana = sens_analytical[:, col]
        safe_ana = np.where(np.abs(ana) > 1e-30, ana, 1e-30)

        err_fwd = np.abs((sens_forward[:, col]  - ana) / safe_ana) * 100.0
        err_cen = np.abs((sens_centered[:, col] - ana) / safe_ana) * 100.0

        ax.semilogy(H_plot, np.clip(err_fwd, 1e-10, None),
                    **{k: v for k, v in STYLE["forward"].items()})
        ax.semilogy(H_plot, np.clip(err_cen, 1e-10, None),
                    **{k: v for k, v in STYLE["centered"].items()})

        ax.set_xlabel("$H$ ($\\mu$m)", fontsize=11)
        ax.set_ylabel("Erro relativo (%)", fontsize=10)
        ax.set_title(f"Erro em $\\partial {output_names[col][1:-1]} / \\partial H$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvo: {filename}")


def main():
    t_inicio_total = time.time()

    print("=" * 65)
    print("  ANÁLISE DE SENSIBILIDADE — GÊMEO DIGITAL MICROFLUÍDICO")
    print("=" * 65)
    print(f"  Parâmetros nominais: Tc = {TC_NOM} °C, H = {H_NOM*1e6:.0f} µm")
    print(f"  Perturbações: ΔTc = {DT_C}, ΔH = {DH}")
    print(f"  Pontos na malha: {N_PONTOS} (Tc) × {N_PONTOS} (H)")
    print(f"  Viscosidade nominal: mu(Tc_nom) = {viscosity_from_T(TC_NOM):.4e} Pa·s")
    print("=" * 65)

    print("\n[1/4] Varredura de T_C (H fixo)…")
    E_vs_Tc = np.zeros(N_PONTOS)
    q_vs_Tc = np.zeros(N_PONTOS)
    V_vs_Tc = np.zeros(N_PONTOS)

    for i, Tc in enumerate(Tc_array):
        E, q, V = simular(Tc, H_NOM)
        E_vs_Tc[i], q_vs_Tc[i], V_vs_Tc[i] = E, q, V
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Tc = {Tc:6.1f} °C → E = {E:.3e} W | q = {q:.3e} m³/s | V = {V:.3e} m³")

    print("\n[2/4] Varredura de H (Tc fixo)…")
    E_vs_H = np.zeros(N_PONTOS)
    q_vs_H = np.zeros(N_PONTOS)
    V_vs_H = np.zeros(N_PONTOS)

    for i, H in enumerate(H_array):
        E, q, V = simular(TC_NOM, H)
        E_vs_H[i], q_vs_H[i], V_vs_H[i] = E, q, V
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    H = {H*1e6:7.1f} µm → E = {E:.3e} W | q = {q:.3e} m³/s | V = {V:.3e} m³")

    print("\n[3/4] Calculando diferenças finitas (Forward + Centrada) em H…")
    E_pos = np.zeros(N_PONTOS)
    q_pos = np.zeros(N_PONTOS)
    V_pos = np.zeros(N_PONTOS)
    E_neg = np.zeros(N_PONTOS)
    q_neg = np.zeros(N_PONTOS)
    V_neg = np.zeros(N_PONTOS)

    for i, H in enumerate(H_array):
        Ep, qp, Vp = simular(TC_NOM, H + DH)
        En, qn, Vn = simular(TC_NOM, H - DH)
        E_pos[i], q_pos[i], V_pos[i] = Ep, qp, Vp
        E_neg[i], q_neg[i], V_neg[i] = En, qn, Vn
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    H = {H*1e6:7.1f} µm  [f(H±ΔH) calculados]")

    sens_forward = np.column_stack([
        (E_pos - E_vs_H) / DH,
        (q_pos - q_vs_H) / DH,
        (V_pos - V_vs_H) / DH,
    ])

    sens_centered = np.column_stack([
        (E_pos - E_neg) / (2.0 * DH),
        (q_pos - q_neg) / (2.0 * DH),
        (V_pos - V_neg) / (2.0 * DH),
    ])

    print("\n[4/4] Calculando sensibilidade analítica (dC_k/dH = 4*C_k/H)…")
    sens_analytical_full = np.zeros((N_PONTOS, 3))
    for i, H in enumerate(H_array):
        sens_analytical_full[i] = sensibilidade_analitica_H(
            E_vs_H[i], q_vs_H[i], V_vs_H[i], H
        )

    idx_nom = np.argmin(np.abs(H_array - H_NOM))
    print("\n" + "=" * 65)
    print(f"  SENSIBILIDADES NO PONTO NOMINAL (H = {H_NOM*1e6:.0f} µm)")
    print("  " + "-" * 62)
    print(f"  {'Saída':<12} {'Forward':>18} {'Centrada':>18} {'Analítica':>18}")
    print("  " + "-" * 62)
    for j, nome in enumerate(["dE/dH", "dq/dH", "dV/dH"]):
        fwd = sens_forward[idx_nom, j]
        cen = sens_centered[idx_nom, j]
        ana = sens_analytical_full[idx_nom, j]
        print(f"  {nome:<12} {fwd:>18.4e} {cen:>18.4e} {ana:>18.4e}")
    print("=" * 65)

    print("\nGerando gráficos…")

    plot_comportamento(
        Tc_array, H_array,
        E_vs_Tc, q_vs_Tc, V_vs_Tc,
        E_vs_H,  q_vs_H,  V_vs_H,
        filename="sens_comportamento.png",
    )

    plot_sensibilidade_H(
        H_array,
        sens_forward,
        sens_centered,
        sens_analytical_full,
        filename="sens_comparativo_H.png",
    )

    plot_erro_relativo_H(
        H_array,
        sens_forward,
        sens_centered,
        sens_analytical_full,
        filename="sens_erro_relativo_H.png",
    )

    tempo_total = time.time() - t_inicio_total
    print(f"\n✓ Análise concluída em {tempo_total:.1f} s.")
    print("  Arquivos gerados:")
    print("    • sens_comportamento.png   — Comportamento das funções E, q, V")
    print("    • sens_comparativo_H.png   — Comparação Forward / Centrada / Analítica")
    print("    • sens_erro_relativo_H.png — Erro relativo (escala logarítmica)")


if __name__ == "__main__":
    main()
