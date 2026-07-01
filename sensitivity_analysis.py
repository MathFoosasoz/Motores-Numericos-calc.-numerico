import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    E       = float(resultado["potencia"][-1])
    q_inlet = float((A_fisico @ pressoes_finais)[sim.node_inlet])
    V_tf    = float(resultado["volume_reservatorio"][-1])

    return E, q_inlet, V_tf


N_PONTOS = 50
DT_C     = 1e-3
DH       = 1e-6

TC_NOM = 25.0
H_NOM  = 1000.0e-6

Tc_array = np.linspace(0.0,      250.0,       N_PONTOS)
H_array  = np.linspace(500.0e-6, 1500.0e-6,   N_PONTOS)

STYLE = {
    "forward":    {"color": "#e6194b", "ls": "--",  "lw": 1.8, "label": "Forward (1ª ordem)"},
    "centered":   {"color": "#3cb44b", "ls": "-.",  "lw": 2.0, "label": "Centrada (2ª ordem)"},
    "analytical": {"color": "#4363d8", "ls": "-",   "lw": 2.2, "label": "Analítica"},
}

OUT_NAMES  = ["E", "q_{inlet}", "V(t_f)"]
OUT_LABELS = [
    "Potência $E$ (W)",
    "Vazão $q_{inlet}$ (m³/s)",
    "Volume $V(t_f)$ (m³)",
]


def sensibilidade_analitica_H(E, q_inlet, V_tf, H):
    """
    C_k ∝ H^4  →  dC_k/dH = 4·C_k/H
    Com pressão de entrada fixa, todo funcional F que escala com C escala com H^4:
      dF/dH = 4·F/H
    Isso vale para E, q_inlet e V(tf).
    """
    return np.array([4.0 * E / H, 4.0 * q_inlet / H, 4.0 * V_tf / H])


def sensibilidade_analitica_Tc(E, q_inlet, V_tf, Tc):
    """
    C_k ∝ 1/μ(Tc)  →  dC_k/dTc = -C_k · (dμ/dTc) / μ
    Como todo funcional F ∝ C (pressão fixa):
      dF/dTc = -F · (dμ/dTc) / μ(Tc)
    """
    mu  = viscosity_from_T(Tc)
    dmu = dmu_dT(Tc)
    fator = -dmu / mu
    return np.array([fator * E, fator * q_inlet, fator * V_tf])


def plot_comportamento(Tc_arr, H_arr,
                       E_Tc, q_Tc, V_Tc,
                       E_H,  q_H,  V_H,
                       filename="sens_comportamento.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Comportamento das Saídas do Gêmeo Digital Microfluídico\n"
        f"(malha 51×51, $\\Delta t=0.025$, $t_f={env.CONFIG_MH['TIME_END']}$ s)",
        fontsize=14, fontweight="bold"
    )

    row0 = [(E_Tc, OUT_LABELS[0], "#c0392b"),
            (q_Tc, OUT_LABELS[1], "#2980b9"),
            (V_Tc, OUT_LABELS[2], "#27ae60")]
    row1 = [(E_H,  OUT_LABELS[0], "#c0392b"),
            (q_H,  OUT_LABELS[1], "#2980b9"),
            (V_H,  OUT_LABELS[2], "#27ae60")]

    for col, (data, ylabel, color) in enumerate(row0):
        ax = axes[0, col]
        ax.plot(Tc_arr, data, color=color, lw=2)
        ax.set_xlabel("$T_C$ (°C)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"$H = {H_NOM*1e6:.0f}\\,\\mu$m (fixo)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    H_plot = H_arr * 1e6
    for col, (data, ylabel, color) in enumerate(row1):
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


def plot_derivadas(x_arr, sens_forward, sens_centered, sens_analytical,
                   x_label, deriv_symbol, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Análise Comparativa de Sensibilidade: $\\partial \\cdot / \\partial {deriv_symbol}$\n"
        f"Forward vs. Centrada vs. Analítica",
        fontsize=13, fontweight="bold"
    )

    for col in range(3):
        ax = axes[col]
        ax.plot(x_arr, sens_forward[:, col],    **STYLE["forward"])
        ax.plot(x_arr, sens_centered[:, col],   **STYLE["centered"])
        if sens_analytical is not None:
            ax.plot(x_arr, sens_analytical[:, col], **STYLE["analytical"])
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f"$\\partial {OUT_NAMES[col]} / \\partial {deriv_symbol}$", fontsize=10)
        ax.set_title(f"$\\partial {OUT_NAMES[col]} / \\partial {deriv_symbol}$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvo: {filename}")


def plot_erro_relativo(x_arr, sens_forward, sens_centered, sens_analytical,
                       x_label, deriv_symbol, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Erro Relativo das Diferenças Finitas vs. Solução Analítica\n"
        f"$\\partial \\cdot / \\partial {deriv_symbol}$",
        fontsize=13, fontweight="bold"
    )

    for col in range(3):
        ax = axes[col]
        ana = sens_analytical[:, col]
        safe = np.where(np.abs(ana) > 1e-30, ana, 1e-30)

        err_fwd = np.abs((sens_forward[:, col]  - ana) / safe) * 100.0
        err_cen = np.abs((sens_centered[:, col] - ana) / safe) * 100.0

        ax.semilogy(x_arr, np.clip(err_fwd, 1e-10, None), **STYLE["forward"])
        ax.semilogy(x_arr, np.clip(err_cen, 1e-10, None), **STYLE["centered"])
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Erro relativo (%)", fontsize=10)
        ax.set_title(f"Erro em $\\partial {OUT_NAMES[col]} / \\partial {deriv_symbol}$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvo: {filename}")


def prob3():
    t0 = time.time()

    print("=" * 65)
    print("  ANÁLISE DE SENSIBILIDADE — GÊMEO DIGITAL MICROFLUÍDICO")
    print("=" * 65)
    print(f"  Parâmetros nominais : Tc = {TC_NOM} °C,  H = {H_NOM*1e6:.0f} µm")
    print(f"  Perturbações        : ΔTc = {DT_C},  ΔH = {DH}")
    print(f"  Pontos na varredura : {N_PONTOS}  (cada eixo)")
    print(f"  Viscosidade nominal : μ(Tc_nom) = {viscosity_from_T(TC_NOM):.4e} Pa·s")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    #  [1/5]  Varredura de comportamento: F(Tc) e F(H)
    # ------------------------------------------------------------------ #
    print("\n[1/5] Varredura de comportamento F(Tc) com H fixo…")
    E_Tc = np.zeros(N_PONTOS)
    q_Tc = np.zeros(N_PONTOS)
    V_Tc = np.zeros(N_PONTOS)

    for i, Tc in enumerate(Tc_array):
        E_Tc[i], q_Tc[i], V_Tc[i] = simular(Tc, H_NOM)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Tc = {Tc:6.1f} °C → E = {E_Tc[i]:.3e} W | "
                  f"q = {q_Tc[i]:.3e} m³/s | V = {V_Tc[i]:.3e} m³")

    print("\n[2/5] Varredura de comportamento F(H) com Tc fixo…")
    E_H = np.zeros(N_PONTOS)
    q_H = np.zeros(N_PONTOS)
    V_H = np.zeros(N_PONTOS)

    for i, H in enumerate(H_array):
        E_H[i], q_H[i], V_H[i] = simular(TC_NOM, H)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    H = {H*1e6:7.1f} µm → E = {E_H[i]:.3e} W | "
                  f"q = {q_H[i]:.3e} m³/s | V = {V_H[i]:.3e} m³")

    plot_comportamento(Tc_array, H_array,
                       E_Tc, q_Tc, V_Tc,
                       E_H,  q_H,  V_H,
                       filename="sens_comportamento.png")

    # ------------------------------------------------------------------ #
    #  [3/5]  Sensibilidades em relação a H (forward, centrada, analítica)
    # ------------------------------------------------------------------ #
    print("\n[3/5] Calculando ∂F/∂H (Forward + Centrada + Analítica)…")

    E_Hp = np.zeros(N_PONTOS); q_Hp = np.zeros(N_PONTOS); V_Hp = np.zeros(N_PONTOS)
    E_Hm = np.zeros(N_PONTOS); q_Hm = np.zeros(N_PONTOS); V_Hm = np.zeros(N_PONTOS)

    for i, H in enumerate(H_array):
        E_Hp[i], q_Hp[i], V_Hp[i] = simular(TC_NOM, H + DH)
        E_Hm[i], q_Hm[i], V_Hm[i] = simular(TC_NOM, H - DH)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    H = {H*1e6:7.1f} µm  [f(H±ΔH) calculados]")

    sens_fwd_H = np.column_stack([
        (E_Hp  - E_H)  / DH,
        (q_Hp  - q_H)  / DH,
        (V_Hp  - V_H)  / DH,
    ])
    sens_cen_H = np.column_stack([
        (E_Hp  - E_Hm) / (2.0 * DH),
        (q_Hp  - q_Hm) / (2.0 * DH),
        (V_Hp  - V_Hm) / (2.0 * DH),
    ])
    sens_ana_H = np.array([
        sensibilidade_analitica_H(E_H[i], q_H[i], V_H[i], H_array[i])
        for i in range(N_PONTOS)
    ])

    # ------------------------------------------------------------------ #
    #  [4/5]  Sensibilidades em relação a Tc (forward, centrada, analítica)
    # ------------------------------------------------------------------ #
    print("\n[4/5] Calculando ∂F/∂Tc (Forward + Centrada + Analítica)…")

    E_Tcp = np.zeros(N_PONTOS); q_Tcp = np.zeros(N_PONTOS); V_Tcp = np.zeros(N_PONTOS)
    E_Tcm = np.zeros(N_PONTOS); q_Tcm = np.zeros(N_PONTOS); V_Tcm = np.zeros(N_PONTOS)

    for i, Tc in enumerate(Tc_array):
        E_Tcp[i], q_Tcp[i], V_Tcp[i] = simular(Tc + DT_C, H_NOM)
        E_Tcm[i], q_Tcm[i], V_Tcm[i] = simular(Tc - DT_C, H_NOM)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Tc = {Tc:6.1f} °C  [f(Tc±ΔTc) calculados]")

    sens_fwd_Tc = np.column_stack([
        (E_Tcp  - E_Tc) / DT_C,
        (q_Tcp  - q_Tc) / DT_C,
        (V_Tcp  - V_Tc) / DT_C,
    ])
    sens_cen_Tc = np.column_stack([
        (E_Tcp  - E_Tcm) / (2.0 * DT_C),
        (q_Tcp  - q_Tcm) / (2.0 * DT_C),
        (V_Tcp  - V_Tcm) / (2.0 * DT_C),
    ])
    sens_ana_Tc = np.array([
        sensibilidade_analitica_Tc(E_Tc[i], q_Tc[i], V_Tc[i], Tc_array[i])
        for i in range(N_PONTOS)
    ])

    # ------------------------------------------------------------------ #
    #  [5/5]  Tabela resumo no ponto nominal + gráficos
    # ------------------------------------------------------------------ #
    idx_H  = np.argmin(np.abs(H_array  - H_NOM))
    idx_Tc = np.argmin(np.abs(Tc_array - TC_NOM))

    print("\n" + "=" * 72)
    print(f"  SENSIBILIDADES NO PONTO NOMINAL  (H = {H_NOM*1e6:.0f} µm, Tc = {TC_NOM:.0f} °C)")
    print("  " + "-" * 69)
    print(f"  {'Derivada':<14} {'Forward':>18} {'Centrada':>18} {'Analítica':>18}")
    print("  " + "-" * 69)
    for j, nome in enumerate(["dE/dH", "dq/dH", "dV/dH"]):
        print(f"  {nome:<14} {sens_fwd_H[idx_H, j]:>18.4e} "
              f"{sens_cen_H[idx_H, j]:>18.4e} {sens_ana_H[idx_H, j]:>18.4e}")
    print("  " + "-" * 69)
    for j, nome in enumerate(["dE/dTc", "dq/dTc", "dV/dTc"]):
        print(f"  {nome:<14} {sens_fwd_Tc[idx_Tc, j]:>18.4e} "
              f"{sens_cen_Tc[idx_Tc, j]:>18.4e} {sens_ana_Tc[idx_Tc, j]:>18.4e}")
    print("=" * 72)

    print("\n[5/5] Gerando gráficos…")

    plot_derivadas(H_array * 1e6, sens_fwd_H, sens_cen_H, sens_ana_H,
                   "$H$ ($\\mu$m)", "H",
                   filename="sens_derivadas_H.png")

    plot_erro_relativo(H_array * 1e6, sens_fwd_H, sens_cen_H, sens_ana_H,
                       "$H$ ($\\mu$m)", "H",
                       filename="sens_erro_H.png")

    plot_derivadas(Tc_array, sens_fwd_Tc, sens_cen_Tc, sens_ana_Tc,
                   "$T_C$ (°C)", "T_C",
                   filename="sens_derivadas_Tc.png")

    plot_erro_relativo(Tc_array, sens_fwd_Tc, sens_cen_Tc, sens_ana_Tc,
                       "$T_C$ (°C)", "T_C",
                       filename="sens_erro_Tc.png")

    tempo_total = time.time() - t0
    print(f"\n✓ Análise concluída em {tempo_total:.1f} s.")
    print("  Arquivos gerados:")
    print("    • sens_comportamento.png  — F(Tc) e F(H) para E, q, V")
    print("    • sens_derivadas_H.png    — ∂F/∂H: Forward / Centrada / Analítica")
    print("    • sens_erro_H.png         — Erro relativo ∂F/∂H vs. analítica")
    print("    • sens_derivadas_Tc.png   — ∂F/∂Tc: Forward / Centrada / Analítica")
    print("    • sens_erro_Tc.png        — Erro relativo ∂F/∂Tc vs. analítica")

