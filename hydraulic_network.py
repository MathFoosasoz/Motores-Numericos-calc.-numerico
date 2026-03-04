import numpy as np

class Hydraulics():
    def __init__(self, conec, C, natm, nB, QB):
        self.conec = conec
        self.C = C

        self.num_nodes = np.max(conec)   # O número de nós pode ser recuperado a partir do maior nó da conec
        self.num_pipes = np.shape(C)[0]  # O número de canos pode ser recuperado a partir do número de linhas da matriz C

        self.node_atm = natm -1          # Indice do nó que está aberto para atmosfera (pressão nesse nó = 0)
        self.node_b = nB -1              # Indice do nó que está ligado à bomba de fluido (vazão nesse nó = QB)
        self.QB = QB                     # Vazão de entrada na rede

        # P = pressões, Q = Vazões nos canos, W = Potência dissipada
        self.results = {'P': None, 'Q': None, 'W': None} 
        self.calculate_flow_rate_and_potency()

       
    def Assembly(self):
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

        A_tilde[self.node_atm, :] = 0                   # A linha i == node_atm deve ser completamente zerada...
        A_tilde[self.node_atm, self.node_atm] = 1       # menos na posição i == j == node_atm. Nessa posição deve ser colocado o valor 1

        num_nodes = A_tilde.shape[0]                    # O número de nós pode ser recuperado a partir do número de linhas da matriz A_tilde

        b_vector = np.zeros(shape=(num_nodes))          # O vetor b é uma linha da dimensão do número de nós, formado inteiramente de zeros menos...
        b_vector[self.node_b] = self.QB                 # no indice onde há vazão

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