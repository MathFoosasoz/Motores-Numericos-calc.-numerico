"""

CONFIG_H:

    N_INLET = indice do nó de entrada de vazão/pressão (se houver apenas uma entrada de vazão/pressão)
    INLET_FLOW = vazão de entrada
    INLET_PRESSURE = pressão de entrada
    INLET_FLOW_DICT = (dict) onde as chaves são indices dos nós de entrada e os valores são as vazões de entrada
    INLET_PRESSURE_DICT = (dict) onde as chaves são indices dos nós de entrada e os valores são as pressões de entrada
    INLET_FLOW_SIN_DICT = (dict) que contém as constantes da função f(t) = A*sen(omega*t + theta) + B
    INLET_FLOW_COS_DICT = (dict) que contém as constantes da função g(t) = A*cos(omega*t + theta) + B
    N_OUTLET = indice do nó de saida de vazão/pressão (se houver apenas uma saida de vazão/pressão)
    OUTLET = pressão/vazão de saida

    PIPE_AREA = área da seção tranversal dos canos
    VISCOSITY = viscosidade do fluido
    TIME_ANALYSIS = (list) tempo inicial, final e repartições (para funcionalidades que dependem de análise temporal)

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

"""
CONFIG_T:

    CONDUCTIVITY: Condutividade térmica fixa para a placa 
    SOURCE: Valor fixo para a fonte térmica
    N: (tuple) Discreções fixas para Nx e Ny (DEFINIDO FORA COMO N_CONFIG_T !!!)
    L: (tuple) Dimensões da placa Lx e Ly
    BORDER_TEMPS: (list) Temperaturas nas bordas, seguindo o padrão do ciclo trigonométrico (direita, cima, esquerda, baixo)

    CIRCULAR_SOURCE_KNOWN_TEMP_DICT: (dict) Raio, coordenadas relativas a Lx e Ly, e temperatura da fonte circular (problema 2)
    CIRCULAR_SOURCE_KNOWN_N_DICT: (dict) Raio, coordenadas relativas a Lx e Ly e tupla das discretizações da fonte circular (problema 4)
    MULTI_N: (list) Tuplas contendo as várias discreções Nx e Ny para análise (problema 1)

"""
N_CONFIG_T = (101, 51)
CONFIG_T = {

    "CONDUCTIVITY": 0.2,
    "SOURCE": 1000000,
    "N": N_CONFIG_T,
    "L": (0.02, 0.01),
    "BORDER_TEMPS": [30, lambda x: 10 + 20*(x/(N_CONFIG_T[0] - 1)), 10, lambda x: 10 + 20*(x/(N_CONFIG_T[0] - 1))],

    "CIRCULAR_SOURCE_KNOWN_TEMP_DICT": { "R": 0.002, "coords": (0.75, 0.5), "T": 30},
    "CIRCULAR_SOURCE_KNOWN_N_DICT": { "R": 0.002, "coords": (0.75, 0.5), "N":(100,50)},
    "MULTI_N": [(21, 11), (41, 21), (81, 41), (161, 81), (321, 161),]

}