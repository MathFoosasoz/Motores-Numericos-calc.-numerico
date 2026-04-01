"""
CONFIG:

    N_INLET = indice do nó de entrada de vazão/pressão (se houver apenas uma entrada de vazão/pressão)
    INLET_FLOW = vazão de entrada
    INLET_PRESSURE = pressão de entrada
    INLET_FLOW_DICT = dicionário onde as chaves são indices dos nós de entrada e os valores são as vazões de entrada
    INLET_PRESSURE_DICT = dicionário onde as chaves são indices dos nós de entrada e os valores são as pressões de entrada
    INLET_FLOW_SIN_DICT = dicionário que contém as constantes da função f(t) = A*sen(omega*t + theta) + B
    INLET_FLOW_COS_DICT = dicionário que contém as constantes da função g(t) = A*cos(omega*t + theta) + B
    N_OUTLET = indice do nó de saida de vazão/pressão (se houver apenas uma saida de vazão/pressão)
    OUTLET = pressão/vazão de saida

    PIPE_AREA = área da seção tranversal dos canos
    VISCOSITY = viscosidade do fluido
    TIME_ANALYSIS = lista do tempo inicial, final e repartições (para funcionalidades que dependem de análise temporal)

"""

CONFIG_H = {
    "N_INLET": 0,          
    "INLET_FLOW": 1.0e-7,    
    "INLET_PRESSURE": 100,
    "INLET_FLOW_DICT": {"0": 1.0e-7, "175": 1.0e-6},
    "INLET_PRESSURE_DICT": {"0": 1.0e-7, "175": 1.0e-6},
    "INLET_FLOW_SIN_DICT": {"N_INLET": 0, "A": 0.1, "omega": 3, "theta": 0, "B": 1},
    "INLET_FLOW_COS_DICT": {"N_INLET": 175, "A": 0.01, "omega": 4, "theta": 0, "B": 0.1},

    "N_OUTLET": 5,
    "OUTLET": 0,

    "PIPE_AREA": 0.00000025,
    "VISCOSITY": 0.001,
    "TIME_ANALYSIS": [0, 10, 1000]

}

CONFIG_T = {

    "CONDUCTIVITY": 1,
    "BORDER_TEMPS": [1, 2, 3, 4],
    "h": 1

}