import matplotlib.pyplot as plt
import os
from dotenv import dotenv_values

from hydraulics import Hydraulics, Hydraulics_p3
from ploting import PlotaRede
from data_structures import GeraGrafo

def main():

    config = dotenv_values(".env")

    Xno, conec = GeraGrafo(levels=1)
    mm_to_m = 0.001
    Xno = Xno * mm_to_m

    conec = conec + 1
  
    test = Hydraulics_p3(conec, Xno, config)
    test.calculate_flow_rate_and_potency()
    
    print(f"Solução das pressões em cada nó: {test.results['P']}")
    print(f"Solução das vazões em cada cano: {test.results['Q']}")
    print(f"Solução da potência dissipada pelo sistema: {test.results['W']}")

    #PlotaRede(conec, Xno, test.results['P'], test.results['Q'])
    #plt.show()
    

if __name__ == "__main__":
    main()
