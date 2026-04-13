import numpy as np
import matplotlib.pyplot as plt
from thermal import Thermal_P2
from env import CONFIG_T

#cria uma copia do CONFIG_T do env e modifica so o que precisa variar (TC)
def make_config(T_C_value):
    config = CONFIG_T.copy()
    config["MULTI_N"] = [CONFIG_T["N"]]  
    config["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"] = {
        **CONFIG_T["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"],
        "T": T_C_value,
    }
    return config
#tambem testei mudando os dados do config

#para definir a reta linear usa TC=0 e TC=1
sim0 = Thermal_P2(make_config(0.0), method="sparse")
sim0.N = CONFIG_T["N"]
T0 = sim0.solve_system_sparse()

sim1 = Thermal_P2(make_config(1.0), method="sparse")
sim1.N = CONFIG_T["N"]
T1 = sim1.solve_system_sparse()

dT = T1 - T0  #sensibilidade do campo a TC -- T(TC) = T0 + TC * dT

#200 valores de TC entre 0 e 60 graus para varrer
#empty_like cria vetores do mesmo tamanho que TC_values para guardar os resultados
TC_values = np.linspace(0, 60, 200)
T_max  = np.empty_like(TC_values)
T_mean = np.empty_like(TC_values)


for idx, TC in enumerate(TC_values):
    T_field = T0 + TC * dT
    T_max[idx]  = np.max(T_field)   # T maxima
    T_mean[idx] = np.mean(T_field)  # T media


#plota Tmax e Tmean como funcoes de TC
#alem disso esse grafico da pra ver com o mouse cada ponto (x,y)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(TC_values, T_max,  label=r'$T_{\max}$',   color='firebrick', linewidth=2)
ax.plot(TC_values, T_mean, label=r'$T_{\rm mean}$', color='steelblue', linewidth=2)
ax.set_xlabel(r'$T_C$ [°C]', fontsize=13)
ax.set_ylabel('Temperatura [°C]', fontsize=13)
ax.set_title(f'Temperatura máxima e média vs $T_C$', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('p4_Tmax_Tmean_vs_TC.png', dpi=150)
plt.show()