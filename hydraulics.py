import numpy as np
from ploting import PlotaMaxPressao, PlotaRede
import matplotlib.pyplot as plt

class Hydraulics():
    def __init__(self, conec, Xno, config):
        self.conec = conec
        self.Xno = Xno

        self.num_nodes = np.max(conec) + 1          # O número de nós pode ser recuperado a partir do maior nó da conec
        self.num_pipes = np.shape(conec)[0]         # O número de canos pode ser recuperado a partir do número de linhas da matriz C

        self.node_outlet = config["N_OUTLET"]       # Indice do nó que está aberto para atmosfera (pressão nesse nó = OUTLET)
        self.node_inlet = config["N_INLET"]         # Indice do nó que está ligado à bomba de fluido (vazão nesse nó = INLET)
        self.inlet = config["INLET_FLOW"]           # Vazão de entrada na rede
        self.outlet = config["OUTLET"]              # Pressão de saída da rede
        self.pipe_area = config["PIPE_AREA"]        # Área da seção transversal do cano
        self.viscosity = config["VISCOSITY"]        # Viscosidade do fluido

        # P = pressões, Q = Vazões nos canos, W = Potência dissipada
        self.results = {'P': None, 'Q': None, 'W': None} 
        
    def calculate_conductancy(self):

            hydraulic_diameter = (4*self.pipe_area/np.pi)**0.5 
            const_K = np.pi*(hydraulic_diameter**4)/(128*self.viscosity)

            C = np.zeros(shape = self.conec.shape[0])

            for index, connection in enumerate(self.conec):
                node_start, node_end = connection

                x_start, y_start = self.Xno[node_start]
                x_end, y_end = self.Xno[node_end]

                Lk = ((x_start-x_end)**2 + (y_start-y_end)**2)**0.5

                C[index]= const_K/Lk

            self.C = C
            return C

    def Assembly(self):
        self.calculate_conductancy() # Gera a matriz C de condutâncias

        A = np.zeros(shape=(self.num_nodes,self.num_nodes)) # matriz quadrada de dimensão igual ao número de nós, preenchida totalmente com zeros

        for index, conectivity in enumerate(self.C):
            from_node = self.conec[index,0]     # nó de saida
            to_node = self.conec[index,1]       # nó de chegada

            A[from_node, from_node] += conectivity #quando i == j, soma-se a conectividade na posição A[i,i]
            A[to_node, to_node] += conectivity     #quando i == j, soma-se a conectividade na posição A[j,j]

            A[to_node, from_node] -= conectivity   #quando i != j, subtrai-se a conectividade na posição A[i, j]
            A[from_node, to_node] -= conectivity   #quando i != j, subtrai-se a conectividade na posição A[j, i]

            #se não há conexão, a posição continua 0

        return A

    def solveNetwork(self):
        A_tilde = self.Assembly()                       # Gera a matriz A

        A_tilde[self.node_outlet, :] = 0                # A linha i == node_atm deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1 # menos na posição i == j == node_atm. Nessa posição deve ser colocado o valor 1    

        num_nodes = A_tilde.shape[0]                    # O número de nós pode ser recuperado a partir do número de linhas da matriz A_tilde

        b_vector = np.zeros(shape=(num_nodes))          # O vetor b é uma linha da dimensão do número de nós, formado inteiramente de zeros menos...
        b_vector[self.node_inlet] = self.inlet          # no indice onde há vazão ...
        b_vector[self.node_outlet] = self.outlet        # e no indice onde é aberto pra pressão externa (n_atm)

        pressures = np.linalg.solve(A_tilde, b_vector)  # Resolução do sistema A_tilde * pressures = b_vector

        self.results['P'] = pressures                   # Coloca o resultado das pressões no dicionário de resultados

        return pressures

    def calculate_flow_rate_and_potency(self):

        pressures = self.solveNetwork()

        # A matriz_K é uma matriz diagonal, cujos valores matriz[i,i] são as conectividades do vetor C[i], e o resto é 0.
        # A matriz_D é uma matriz de dimensão (num_pipes X num_nodes) que relaciona de onde está indo...
        # e vindo o fluido (1 se esta vindo, -1 se está indo, 0 se não há conexão) entre os nós ( ??? eu acho)
        matriz_K = np.zeros(shape=(self.num_pipes, self.num_pipes))   
        matriz_D = np.zeros(shape=(self.num_pipes, self.num_nodes))

        for k in range(self.num_pipes):
            matriz_K[k,k] = self.C[k]     

            from_node = self.conec[k, 0]    # nó de saida
            to_node = self.conec[k, 1]      # nó de chegada

            for j in range(self.num_nodes):
                if (j == from_node): 
                    matriz_D[k, j] = 1

                if (j == to_node):
                    matriz_D[k, j] = -1

        # Multiplicação de matrizes como está escrito na apostila     
        Q = matriz_K @ matriz_D @ pressures 
        W =  pressures.T @ matriz_D.T @ Q

        # Atualiza os resultados da classe 
        self.results['Q'] = Q
        self.results['W'] = W

        return (Q,W)
    
    def run(self, print_info, plot):

        self.calculate_flow_rate_and_potency()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Solução das pressões em cada nó: {self.results['P']}")
            print(f"Solução das vazões em cada cano: {self.results['Q']}")
            print(f"Solução da potência dissipada pelo sistema: {self.results['W']}\n\n")
            

        if plot:
            PlotaRede(self.conec, 1000*self.Xno, self.results['P'], self.results['Q'])
            plt.show()

        
# Usando herança de classe, podemos modificar facilmente as funções que se relacionam aos problemas extras
# e reutilizar da classe pai aquilo que é mantido   

class Hydraulics_p3(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = config["INLET_PRESSURE"]    # Pressão de entrada na rede

    def calculate_conductancy(self):
        return super().calculate_conductancy()
    
    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        # Definindo as equações de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1

        # Vamos usar essa linha da matriz A_tilde pra resolver qual a vazaão de entrada no final
        line_to_find_inlet_flow = np.array(A_tilde[self.node_inlet, :])

        A_tilde[self.node_inlet, :] = 0                       # A linha i == node_inlet deve ser completamente zerada...
        A_tilde[self.node_inlet, self.node_inlet] = 1         # menos na posição i == j == node_inlet. Nessa posição deve ser colocado o valor 1
    
        b_vector = np.zeros(shape = (self.num_nodes))
        b_vector[self.node_inlet] = self.inlet
        b_vector[self.node_outlet] = self.outlet 
        
        pressures = np.linalg.solve(A_tilde, b_vector)        # Solução do sistema A_tilde * pressures = b_vector
        self.results['P'] = pressures                    

        # Resolução da vazão de entrada
        inlet_flow = np.dot(line_to_find_inlet_flow, pressures)
        self.results["Q_inlet"] = inlet_flow

        return pressures
    
    def calculate_flow_rate_and_potency(self):
        return super().calculate_flow_rate_and_potency()
    
    def run(self, print_info, plot):
        
        self.calculate_flow_rate_and_potency()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Solução das pressões em cada nó: {self.results['P']}")
            print(f"Solução das vazões em cada cano: {self.results['Q']}")
            print(f"Solução da potência dissipada pelo sistema: {self.results['W']}")
            print(f"Vazão no ponto de inlet: {self.results["Q_inlet"]}\n\n")
            

        if plot:
            PlotaRede(self.conec, 1000*self.Xno, self.results['P'], self.results['Q'])
            plt.show()
    

class Hydraulics_p4(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = config["INLET_FLOW_SIN_DICT"] 
        self.time = config["TIME_ANALYSIS"]

    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1

        node_entry = self.inlet["N_INLET"]

        b_vector = np.zeros(shape = (self.num_nodes))
                 
        # Primeiro vamos resolver apenas para o valor 1, e depois...
        # na função de achar as máximas pressões vamos multiplicar os resultados por ...
        # f(t) = A*sen(t*omega + theta) + B para cada tempo da análise.
        # Esse procedimento pode ser realisado por causa da linearidade
        b_vector[node_entry] = 1
        pressures = np.linalg.solve(A_tilde, b_vector)
                
        return pressures
    
    def find_max_pressures_over_time(self):
        # Primeiro pegamos os resultados sem o seno
        pressures_without_sin = self.solveNetwork()

        mL_to_m3 = 0.000001

        time_start = self.time[0]
        time_end = self.time[1]
        increments = self.time[2]

        time = np.linspace(time_start, time_end, increments)
        max_pressures = []

        # Para cada tempo, nós multiplicamos o (A*sen(t*omega + theta) + B) pela solução da solve_network para encontrar as pressões reais
        for t in time:
            pressures_in_t = pressures_without_sin * self.sin_of_t(t) * mL_to_m3
            max_pressures.append(pressures_in_t.max())

        return np.array(max_pressures)
    
    def sin_of_t(self, t):
        A = self.inlet["A"]
        B = self.inlet["B"]
        theta = np.radians(self.inlet["theta"])
        omega = self.inlet['omega']

        return (A*np.sin(t*omega + theta) + B)

    def calculate_conductancy(self):
        return super().calculate_conductancy()
    
    def run(self, print_info, plot):
        
        max_pressures = self.find_max_pressures_over_time()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Pressões ao longo do tempo: {max_pressures}\n\n")

        if plot:
            PlotaMaxPressao(max_pressures, self.time)
            plt.show()


class Hydraulics_p5(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = [config["INLET_FLOW_SIN_DICT"], config["INLET_FLOW_COS_DICT"]] 

        self.time = config["TIME_ANALYSIS"]
        
    def calculate_conductancy(self):
        return super().calculate_conductancy()
    
    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        # Definindo as equações de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1
        
        mL_to_m3 = 0.000001

        node_entry_sin = self.inlet[0]["N_INLET"]
        node_entry_cos = self.inlet[1]["N_INLET"]

        b_vector_sin = np.zeros(shape = (self.num_nodes))
        b_vector_cos = np.zeros(shape = (self.num_nodes))
        b_vector_sin[node_entry_sin] = 1
        b_vector_cos[node_entry_cos] = 1
        
        pressures_sin = np.linalg.solve(A_tilde, b_vector_sin)
        pressures_cos = np.linalg.solve(A_tilde, b_vector_cos)
                 
        return pressures_sin, pressures_cos
    
    def find_max_pressures_over_time(self):
        pressures_without_sin, pressures_without_cos = self.solveNetwork()

        time_start = self.time[0]
        time_end = self.time[1]
        increments = self.time[2]

        time = np.linspace(time_start, time_end, num = increments)
        max_pressures = []

        mL_to_m3 = 0.000001

        for t in time:
            pressures_in_t = pressures_without_sin * self.sin_of_t(t) + pressures_without_cos * self.cos_of_t(t)
            pressures_in_t *= mL_to_m3
            max_pressures.append(pressures_in_t.max())

        return np.array(max_pressures)

    def sin_of_t(self, t):
        A = self.inlet[0]["A"]
        B = self.inlet[0]["B"]
        theta = np.radians(self.inlet[0]["theta"])
        omega = self.inlet[0]['omega']

        return (A*np.sin(t*omega + theta) + B)
    

    def cos_of_t(self, t):
        A = self.inlet[1]["A"]
        B = self.inlet[1]["B"]
        theta = np.radians(self.inlet[1]["theta"])
        omega = self.inlet[1]['omega']

        return (A*np.cos(t*omega + theta) + B)


    def run(self, print_info, plot):
        
        max_pressures = self.find_max_pressures_over_time()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"Pressões ao longo do tempo: {max_pressures}\n\n")

        if plot:
            PlotaMaxPressao(max_pressures, self.time)
            plt.show()
    