import numpy as np
from ploting import PlotaMaxPressao, PlotaRede
import matplotlib.pyplot as plt

# =============================================================================
# EXERC�CIO 1  M�ltiplos pontos de inje��o de vaz�o
# =============================================================================
#
# Motiva��o:
#   Na classe base `Hydraulics`, o vetor b � populado com uma �nica vaz�o num
#   �nico n� de entrada (N_INLET / INLET_FLOW).  Para o Exerc�cio 1 queremos
#   generalizar isso: a entrada passa a ser um DICION�RIO cujas chaves s�o os
#   �ndices dos n�s e cujos valores s�o as respectivas vaz�es impostas.
#
#   Exemplo de dicion�rio no CONFIG:
#       INLET_FLOW_DICT = {"0": 1.0e-7, "175": 1.0e-6}
#
#   Isso permite simular redes com bombeamento em v�rios pontos
#   simultaneamente, sem precisar alterar nada na montagem da matriz A.
#
# Estrat�gia matem�tica:
#   O sistema linear continua sendo  A�P = b.
#   A �nica mudan�a � a forma como b � constru�do:
#
#       Para cada (n�_k, Q_k) em INLET_FLOW_DICT:
#           b[n�_k] = Q_k          � vaz�o imposta naquele n�
#
#   Depois, a condi��o de press�o no outlet � aplicada normalmente:
#       A[node_outlet, :]         = 0   � zera a equa��o de conserva��o do n�
#       A[node_outlet, node_outlet] = 1 � substitui por  P[node_outlet] = OUTLET
#       b[node_outlet]            = OUTLET
#
#   Por fim:  P = np.linalg.solve(A, b)
# =============================================================================

class Hydraulics():
    def __init__(self, conec, Xno, config):
        self.conec = conec
        self.Xno = Xno

        self.num_nodes = np.max(conec) + 1          # O número de nós pode ser recuperado a partir do maior nó da conec
        self.num_pipes = np.shape(conec)[0]         # O número de canos pode ser recuperado a partir do número de linhas da matriz C

        self.node_outlet = config["N_OUTLET"]       # Indice do n� que est� aberto para atmosfera (press�o nesse n� = OUTLET)
        self.node_inlet = config["N_INLET"]         # Indice do n� que est� ligado � bomba de fluido (vaz�o nesse n� = INLET)
        self.inlet = config["INLET_FLOW"]           # Vaz�o de entrada na rede
        self.outlet = config["OUTLET"]              # Press�o de sa�da da rede
        self.pipe_area = config["PIPE_AREA"]        # �rea da se��o transversal do cano
        self.viscosity = config["VISCOSITY"]        # Viscosidade do fluido

        # P = press�es, Q = Vaz�es nos canos, W = Pot�ncia dissipada
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
        self.calculate_conductancy() # Gera a matriz C de condut�ncias

        A = np.zeros(shape=(self.num_nodes,self.num_nodes)) # matriz quadrada de dimens�o igual ao n�mero de n�s, preenchida totalmente com zeros

        for index, conectivity in enumerate(self.C):
            from_node = self.conec[index,0]     # n� de saida
            to_node = self.conec[index,1]       # n� de chegada

            A[from_node, from_node] += conectivity #quando i == j, soma-se a conectividade na posi��o A[i,i]
            A[to_node, to_node] += conectivity     #quando i == j, soma-se a conectividade na posi��o A[j,j]

            A[to_node, from_node] -= conectivity   #quando i != j, subtrai-se a conectividade na posi��o A[i, j]
            A[from_node, to_node] -= conectivity   #quando i != j, subtrai-se a conectividade na posi��o A[j, i]

            #se n�o h� conex�o, a posi��o continua 0

        return A

    def solveNetwork(self):
        A_tilde = self.Assembly()                       # Gera a matriz A

        A_tilde[self.node_outlet, :] = 0                # A linha i == node_atm deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1 # menos na posi��o i == j == node_atm. Nessa posi��o deve ser colocado o valor 1    

        num_nodes = A_tilde.shape[0]                    # O n�mero de n�s pode ser recuperado a partir do n�mero de linhas da matriz A_tilde

        b_vector = np.zeros(shape=(num_nodes))          # O vetor b � uma linha da dimens�o do n�mero de n�s, formado inteiramente de zeros menos...
        b_vector[self.node_inlet] = self.inlet          # no indice onde h� vaz�o ...
        b_vector[self.node_outlet] = self.outlet        # e no indice onde � aberto pra press�o externa (n_atm)

        pressures = np.linalg.solve(A_tilde, b_vector)  # Resolu��o do sistema A_tilde * pressures = b_vector

        self.results['P'] = pressures                   # Coloca o resultado das press�es no dicion�rio de resultados

        return pressures

    def calculate_flow_rate_and_potency(self):

        pressures = self.solveNetwork()

        # A matriz_K � uma matriz diagonal, cujos valores matriz[i,i] s�o as conectividades do vetor C[i], e o resto � 0.
        # A matriz_D � uma matriz de dimens�o (num_pipes X num_nodes) que relaciona de onde est� indo...
        # e vindo o fluido (1 se esta vindo, -1 se est� indo, 0 se n�o h� conex�o) entre os n�s ( ??? eu acho)
        matriz_K = np.zeros(shape=(self.num_pipes, self.num_pipes))   
        matriz_D = np.zeros(shape=(self.num_pipes, self.num_nodes))

        for k in range(self.num_pipes):
            matriz_K[k,k] = self.C[k]     

            from_node = self.conec[k, 0]    # n� de saida
            to_node = self.conec[k, 1]      # n� de chegada

            for j in range(self.num_nodes):
                if (j == from_node): 
                    matriz_D[k, j] = 1

                if (j == to_node):
                    matriz_D[k, j] = -1

        # Multiplica��o de matrizes como est� escrito na apostila     
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
            print(f"Solu��o das press�es em cada n�: {self.results['P']}")
            print(f"Solu��o das vaz�es em cada cano: {self.results['Q']}")
            print(f"Solu��o da pot�ncia dissipada pelo sistema: {self.results['W']}\n\n")
            

        if plot:
            PlotaRede(self.conec, 1000*self.Xno, self.results['P'], self.results['Q'])
            plt.show()

        
# Usando herança de classe, podemos modificar facilmente as funções que se relacionam aos problemas extras
# e reutilizar da classe pai aquilo que é mantido   
class Hydraulics_p2(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)
        self.inlet_pressure = config["INLET_PRESSURE_DICT"] 

    def solveNetwork(self):
        A_tilde = self.Assembly()
        num_nodes = A_tilde.shape[0]
        b_vector = np.zeros(shape=(num_nodes))

        b_vector[self.node_inlet] = self.inlet

        for node, value in self.inlet_pressure.items():
            node = int(node)
            A_tilde[node, :] = 0            
            A_tilde[node, node] = 1          
            b_vector[node] = value
        
        pressures = np.linalg.solve(A_tilde, b_vector)
        self.results['P'] = pressures

        return pressures

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


class Hydraulics_p3(Hydraulics):
    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        self.inlet = config["INLET_PRESSURE"]    # Press�o de entrada na rede

    def calculate_conductancy(self):
        return super().calculate_conductancy()
    
    def Assembly(self):
        return super().Assembly()

    def solveNetwork(self):
        A_tilde = self.Assembly()

        # Definindo as equa��es de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posi��o i == j == node_outlet. Nessa posi��o deve ser colocado o valor 1

        # Vamos usar essa linha da matriz A_tilde pra resolver qual a vaza�o de entrada no final
        line_to_find_inlet_flow = np.array(A_tilde[self.node_inlet, :])

        A_tilde[self.node_inlet, :] = 0                       # A linha i == node_inlet deve ser completamente zerada...
        A_tilde[self.node_inlet, self.node_inlet] = 1         # menos na posi��o i == j == node_inlet. Nessa posi��o deve ser colocado o valor 1
    
        b_vector = np.zeros(shape = (self.num_nodes))
        b_vector[self.node_inlet] = self.inlet
        b_vector[self.node_outlet] = self.outlet 
        
        pressures = np.linalg.solve(A_tilde, b_vector)        # Solu��o do sistema A_tilde * pressures = b_vector
        self.results['P'] = pressures                    

        # Resolu��o da vaz�o de entrada
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
            print(f"Vazão no ponto de inlet: {self.results['Q_inlet']}\n\n")
            

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
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posi��o i == j == node_outlet. Nessa posi��o deve ser colocado o valor 1

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
            print(f"Press�es ao longo do tempo: {max_pressures}\n\n")

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

        # Definindo as equa��es de controle
        A_tilde[self.node_outlet, :] = 0                      # A linha i == node_outlet deve ser completamente zerada...
        A_tilde[self.node_outlet, self.node_outlet] = 1       # menos na posi��o i == j == node_outlet. Nessa posi��o deve ser colocado o valor 1
        
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
            print(f"Press�es ao longo do tempo: {max_pressures}\n\n")

        if plot:
            PlotaMaxPressao(max_pressures, self.time)
            plt.show()


# =============================================================================
# Exerc�cio 1  Classe com m�ltiplos pontos de inje��o de vaz�o
# =============================================================================

class Hydraulics_ex1(Hydraulics):
    """
    Extens�o da classe Hydraulics para o Exerc�cio 1.

    Generaliza a condi��o de entrada: em vez de um �nico n� de inje��o
    (N_INLET / INLET_FLOW), recebe um dicion�rio com v�rios n�s e suas
    respectivas vaz�es impostas.

    Par�metro adicional no CONFIG:
        INLET_FLOW_DICT : dict
            Chaves   � �ndice do n� (str ou int)
            Valores  � vaz�o imposta naquele n� [m�/s] (float)

            Exemplo:
                {"0": 1.0e-7, "175": 1.0e-6}

    Tudo o mais (c�lculo de condut�ncias, montagem de A, c�lculo de Q e W)
    � herdado diretamente de Hydraulics sem nenhuma altera��o.
    """

    def __init__(self, conec, Xno, config):
        super().__init__(conec, Xno, config)

        # Substitui a entrada �nica pelo dicion�rio de m�ltiplas entradas.
        # As chaves do dicion�rio podem vir como strings (JSON); convertemos
        # para int para usar como �ndices do vetor b.
        raw_dict = config["INLET_FLOW_DICT"]
        self.inlet_flow_dict = {int(node): float(flow)
                                for node, flow in raw_dict.items()}

    # -------------------------------------------------------------------------
    # solveNetwork  �nica fun��o modificada em rela��o � classe pai
    # -------------------------------------------------------------------------
    def solveNetwork(self):
        """
        Resolve o sistema linear  A�P = b  com m�ltiplos pontos de inje��o.

        Diferen�a em rela��o a Hydraulics.solveNetwork():
            " O vetor b n�o recebe apenas b[N_INLET] = INLET_FLOW.
            " Em vez disso, percorremos self.inlet_flow_dict e fazemos:
                  for node, flow in self.inlet_flow_dict.items():
                      b[node] = flow
            " O restante (condi��o de outlet, resolu��o do sistema) � id�ntico.
        """

        # --- Passo 1: monta a matriz de condut�ncias A ---
        A_tilde = self.Assembly()

        # --- Passo 2: imp�e condi��o de press�o no outlet ---
        # Zera toda a linha do n� de sa�da e coloca 1 na diagonal.
        # Isso substitui a equa��o de conserva��o desse n� pela equa��o:
        #     P[node_outlet] = OUTLET
        A_tilde[self.node_outlet, :]                    = 0
        A_tilde[self.node_outlet, self.node_outlet]     = 1

        # --- Passo 3: inicializa o vetor b com zeros ---
        # Dimens�o: um elemento por n� da rede.
        b_vector = np.zeros(shape=(self.num_nodes,))

        # --- Passo 4: popula as vaz�es a partir do dicion�rio ---
        # Para cada par (n�, vaz�o) no dicion�rio de entradas,
        # atribu�mos a vaz�o � posi��o correspondente no vetor b.
        # N�s com vaz�o n�o especificada continuam com b[n�] = 0
        # (conserva��o de massa sem fonte/sumidouro).
        for node, flow in self.inlet_flow_dict.items():
            b_vector[node] = flow

        # --- Passo 5: imp�e a press�o no outlet ---
        # Sobrescreve a posi��o do outlet com o valor de press�o imposto.
        # (Feito ap�s o loop para garantir que o outlet n�o seja sobrescrito
        # acidentalmente caso ele tamb�m apare�a em inlet_flow_dict.)
        b_vector[self.node_outlet] = self.outlet

        # --- Passo 6: resolve o sistema linear A�P = b ---
        pressures = np.linalg.solve(A_tilde, b_vector)

        # Armazena e retorna
        self.results['P'] = pressures
        return pressures

    def run(self, print_info, plot):
        self.calculate_flow_rate_and_potency()

        if print_info:
            print(f"Resultados para classe: {self.__class__.__name__}")
            print(f"N�s de inje��o (INLET_FLOW_DICT): {self.inlet_flow_dict}")
            print(f"Solu��o das press�es em cada n�:  {self.results['P']}")
            print(f"Solu��o das vaz�es em cada cano:  {self.results['Q']}")
            print(f"Pot�ncia dissipada pelo sistema:  {self.results['W']}\n\n")

        if plot:
            PlotaRede(self.conec, 1000 * self.Xno,
                      self.results['P'], self.results['Q'])
            plt.show()
