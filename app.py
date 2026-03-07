import numpy as np
import matplotlib.pyplot as plt
from hydraulic_network import Hydraulics
from plots import PlotaRede

def main():
    # Matriz conec, da apostila.
    conec = np.array([[1,2],
                    [2,3],
                    [3,4],
                    [4,5],
                    [5,2],
                    [5,3],
                    [5,1]]) 

    # OBS: Como os valores dos nós na apostila são indexados a partir de 1, e o Python começa indexação a partir de 0, sempre que usar 
    # os valores da matriz conec abaixo pra simbolizar as origens e destinos dos fluxo é necessário uma correção de -1

    C  = np.array([2, 2, 1, 2, 1, 2, 2]) # Vetor de conectividades (com valores de exemplo da apostila)
    #              c1 c2 c3 c4 c5 c6 c7 

    n_atm = 3
    nB = 1
    QB = 3

    test = Hydraulics(conec, C, n_atm, nB, QB)

# Coordenadas dos nós (com valores de exemplo da apostila)
    Xno = np.array([[0,0],
                    [1,0],
                    [2,0],
                    [2,1],
                    [1,1]])
    
    print(f"Solução das pressões em cada nó: {test.results['P']}")
    print(f"Solução das vazões em cada cano: {test.results['Q']}")
    print(f"Solução da potência dissipada pelo sistema: {test.results['W']}")

    PlotaRede(conec, Xno, test.results['P'], test.results['Q'])
    plt.show()

if __name__ == "__main__":
    main()
