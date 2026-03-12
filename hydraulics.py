import numpy as np
from json import loads

class Hydraulics():
    def __init__(self, conec, Xno, config):
        self.conec = conec
        self.Xno = Xno

        self.num_nodes = np.max(conec)                      # O número de nós pode ser recuperado a partir do maior nó da conec
        self.num_pipes = np.shape(conec)[0]                 # O número de canos pode ser recuperado a partir do número de linhas da matriz C

        self.node_outlet = int(config["N_OUTLET"]) - 1      # Indice do nó que está aberto para atmosfera (pressão nesse nó = OUTLET)
        self.node_inlet = int(config["N_INLET"]) - 1        # Indice do nó que está ligado à bomba de fluido (vazão nesse nó = INLET)
        self.inlet = float(config["INLET_FLOW"])            # Vazão de entrada na rede
        self.outlet = float(config["OUTLET"])               # Pressão de saída da rede
        self.pipe_area = float(config["PIPE_AREA"])         # Área da seção transversal do cano
        self.viscosity = float(config["VISCOSITY"])         # Viscosidade do fluido

        # P = pressões, Q = Vazões nos canos, W = Potência dissipada
        self.results = {'P': None, 'Q': None, 'W': None} 
        

    def calculate_conductancy(self):

            hydraulic_diameter = (4*self.pipe_area/np.pi)**0.5 
            const_K = np.pi*(hydraulic_diameter**4)/(128*self.viscosity)

            C = np.zeros(shape = self.conec.shape[0])

            for index, connection in enumerate(self.conec):
                node_start, node_end = connection

                x_start, y_start = self.Xno[node_start-1]
                x_end, y_end = self.Xno[node_end-1]

                Lk = ((x_start-x_end)**2 + (y_start-y_end)**2)**0.5

                C[index]= const_K/Lk

            self.C = C
            return C


    def Assembly(self):
        self.calculate_conductancy() # Gera a matriz C de condutâncias

        A = np.zeros(shape=(self.num_nodes,self.num_nodes)) # matriz quadrada de dimensão igual ao número de nós, preenchida totalmente com zeros

        for index, conectivity in enumerate(self.C):
            from_node = self.conec[index,0] -1  # nó de saida
            to_node = self.conec[index,1] -1    # nó de chegada

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

            from_node = self.conec[k, 0] -1 # nó de saida
            to_node = self.conec[k, 1] -1 # nó de chegada

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

# Usando herança de classe, podemos modificar facilmente as funções que se relacionam aos problemas extras
# e reutilizar da classe pai aquilo que é mantido

class Hydraulics_p3(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = float(config["INLET_PRESSURE"])    # Pressão de entrada na rede

    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        # Definindo as equações de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1

        A_tilde[self.node_inlet, :] = 0                       # A linha i == node_inlet deve ser completamente zerada...
        A_tilde[self.node_inlet, self.node_inlet] = 1         # menos na posição i == j == node_inlet. Nessa posição deve ser colocado o valor 1
    
        b_vector = np.zeros(shape = (self.num_nodes))
        b_vector[self.node_inlet] = self.inlet
        b_vector[self.node_outlet] = self.outlet 
        
        pressures = np.linalg.solve(A_tilde, b_vector)        # Solução do sistema A_tilde * pressures = b_vector
        self.results['P'] = pressures                    

        return pressures
    
    def calculate_conductancy(self):
        return super().calculate_conductancy()
    
    def calculate_flow_rate_and_potency(self):
        return super().calculate_flow_rate_and_potency()
    

class Hydraulics_p4(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = loads(config["INLET_FLOW_SIN_DICT"])   
        time_dict = loads(config["TIME_ANALYSIS"])    
        self.time = time_dict["t"] 

    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1

        b_vector = np.zeros(shape = (self.num_nodes))
        mL_to_m3 = 0.000001

        # Primeiro vamos resolver apenas para as constante A que multiplica o seno, e depois...
        # vamos multiplicar os resultados pela sen(t*omega + theta) para cada tempo da análise.
        # Esse procedimento pode ser realisado por causa da linearidade
        b_vector[int(self.inlet["N_INLET"])-1] = float(self.inlet["A"]) * mL_to_m3 
        pressures_without_sin = np.linalg.solve(A_tilde, b_vector)
                
        return pressures_without_sin
    
    def find_max_pressures_over_time(self):
        # Primeiro pegamos os resultados sem o seno
        pressures_without_sin = self.solveNetwork()

        theta = np.radians(float(self.inlet["theta"]))
        omega = float(self.inlet['omega'])

        time_start = float(self.time[0])
        time_end = float(self.time[1])
        increments = int(self.time[2])

        time = np.linspace(time_start, time_end, increments)
        max_pressures = []

        # Para cada tempo, nós multiplicamos o sen(t*omega + theta) pela solução da solve_network para encontrar as pressões reais
        for t in time:
            pressures_in_t = pressures_without_sin * np.sin(t*omega + theta)
            max_pressures.append(pressures_in_t.max())

        return np.array(max_pressures)


    def calculate_conductancy(self):
        return super().calculate_conductancy()


class Hydraulics_p5(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = [loads(config["INLET_FLOW_SIN_DICT"]), loads(config["INLET_FLOW_COS_DICT"])]
        time_dict = loads(config["TIME_ANALYSIS"])    

        self.time = time_dict["t"]
        
    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        # Definindo as equações de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posição i == j == node_outlet. Nessa posição deve ser colocado o valor 1
        
        mL_to_m3 = 0.000001

        b_vector_sin = np.zeros(shape = (self.num_nodes))
        b_vector_cos = np.zeros(shape = (self.num_nodes))

        b_vector_sin[int(self.inlet[0]["N_INLET"])-1] = float(self.inlet[0]["A"]) * mL_to_m3
        b_vector_cos[int(self.inlet[1]["N_INLET"])-1] = float(self.inlet[1]["A"]) * mL_to_m3
        
        pressures_without_sin = np.linalg.solve(A_tilde, b_vector_sin)
        pressures_without_cos = np.linalg.solve(A_tilde, b_vector_cos)
                 
        return pressures_without_sin, pressures_without_cos
    
    def find_max_pressures_over_time(self):
        pressures_without_sin, pressures_without_cos = self.solveNetwork()

        theta_sin = np.radians(float(self.inlet[0]["theta"]))
        theta_cos = np.radians(float(self.inlet[1]["theta"]))

        omega_sin = float(self.inlet[0]['omega'])
        omega_cos = float(self.inlet[1]['omega'])

        time_start = float(self.time[0])
        time_end = float(self.time[1])
        increments = int(self.time[2])

        time = np.linspace(time_start, time_end, num = increments)
        max_pressures = []

        for t in time:
            pressures_in_t = pressures_without_sin * np.sin(t*omega_sin + theta_sin) + pressures_without_cos * np.cos(t*omega_cos + theta_cos)
            max_pressures.append(pressures_in_t.max())

        return np.array(max_pressures)


    def calculate_conductancy(self):
        return super().calculate_conductancy()
    