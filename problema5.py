import numpy as np
import matplotlib.pyplot as plt
from thermal import Thermal_P2
from env import CONFIG_T

k_node = 233 #no sugerido pelo problema

#monta o config a partir do env
#TR_val varia entre os 3 pares, topo e base vem do env (dependem de x)
def make_config(TR_val, TC_val):
    config = CONFIG_T.copy()
    config["MULTI_N"] = [CONFIG_T["N"]]
    config["BORDER_TEMPS"] = [
        TR_val,                          # direita — variável
        CONFIG_T["BORDER_TEMPS"][1],     # topo — lambda do env
        CONFIG_T["BORDER_TEMPS"][2],     # esquerda
        CONFIG_T["BORDER_TEMPS"][3],     # base — lambda do env
    ]
    config["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"] = {
        **CONFIG_T["CIRCULAR_SOURCE_KNOWN_TEMP_DICT"],
        "T": TC_val,
    }
    return config

#resolve o sistema para um par e retorna o campo de temperaturas
def solve(TR_val, TC_val):
    sim = Thermal_P2(make_config(TR_val, TC_val), method="sparse")
    sim.N = CONFIG_T["N"]
    return sim.solve_system_sparse()


#3 pares (TR, TC) para montar o sistema, escolhidos para isolar bem cada coeficiente
pairs = [
    (30.0,  0.0),
    (30.0, 30.0),
    ( 0.0,  0.0),
]

#resolve o sistema para cada par e guarda T_k
print("Resolvendo 3 sistemas para determinar a, b, c:")
T_k_vals = []
for TR_val, TC_val in pairs:
    print(f"  TR={TR_val:.1f}°C  TC={TC_val:.1f}°C ->", end=" ", flush=True)
    T_field = solve(TR_val, TC_val)
    T_k_vals.append(T_field[k_node])
    print(f"T_k = {T_field[k_node]:.6f}°C")

#monta o sistema
M = np.array([
    [pairs[0][0], pairs[0][1], 1.0],
    [pairs[1][0], pairs[1][1], 1.0],
    [pairs[2][0], pairs[2][1], 1.0],
])

#resolve o sistema
a, b, c = np.linalg.solve(M, np.array(T_k_vals))

print(f"\nCoeficientes encontrados (nó k={k_node})")
print(f"  a = {a:.8f}   (sensibilidade a T_R)")
print(f"  b = {b:.8f}   (sensibilidade a T_C)")
print(f"  c = {c:.8f}   (termo independente)")
print(f"\nEquacao linear\n  T_{k_node} = {a:.6f}·T_R  +  {b:.6f}·T_C  +  {c:.6f}")