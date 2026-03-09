import numpy as np
import matplotlib.pyplot as plt
from hydraulics import Hydraulics
from ploting import PlotaRede
from data_structures import GeraGrafo

def main():
    Xno, conec = GeraGrafo(levels=1)
    mm_to_m = 0.001
    Xno = Xno * mm_to_m

    conec = conec + 1

    n_atm = 3
    nB = 1
    QB = 3
  
    test = Hydraulics(conec, Xno, n_atm, nB, QB)
    
    print(f"Solução das pressões em cada nó: {test.results['P']}")
    print(f"Solução das vazões em cada cano: {test.results['Q']}")
    print(f"Solução da potência dissipada pelo sistema: {test.results['W']}")


    PlotaRede(conec, Xno, test.results['P'], test.results['Q'])
    plt.show()
    

if __name__ == "__main__":
    main()
