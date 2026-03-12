import matplotlib.pyplot as plt
import os
from dotenv import dotenv_values

from hydraulics import Hydraulics, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5
from ploting import PlotaRede, PlotaMaxPressao
from data_structures import GeraGrafo

def main():
    config = dotenv_values(".env")

    Xno, conec = GeraGrafo(levels=3)
    mm_to_m = 0.001
    Xno = Xno * mm_to_m

    conec = conec + 1

    test = Hydraulics(conec, Xno, config)
    test.calculate_flow_rate_and_potency()
    # print(f"Solução das pressões em cada nó: {test.results['P']}")
    # print(f"Solução das vazões em cada cano: {test.results['Q']}")
    # print(f"Solução da potência dissipada pelo sistema: {test.results['W']}")

    # PlotaRede(conec, 1000*Xno, test.results['P'], test.results['Q'])
    # plt.show()
  
    test_p3 = Hydraulics_p3(conec, Xno, config)
    test_p3.calculate_flow_rate_and_potency()
    # print(f"P3: Solução das pressões em cada nó: {test_p3.results['P']}")
    # print(f"P3: Solução das vazões em cada cano: {test_p3.results['Q']}")
    # print(f"P3: Solução da potência dissipada pelo sistema: {test_p3.results['W']}")

    # PlotaRede(conec, 1000*Xno, test_p3.results['P'], test_p3.results['Q'])
    # plt.show()

    test_p4 = Hydraulics_p4(conec, Xno, config)
    max_pressures_p4 = test_p4.find_max_pressures_over_time()

    PlotaMaxPressao(max_pressures_p4, config)
    plt.show()

    test_p5 = Hydraulics_p5(conec, Xno, config)
    max_pressures_p5 = test_p5.find_max_pressures_over_time()

    PlotaMaxPressao(max_pressures_p5, config)
    plt.show()
    

if __name__ == "__main__":
    main()
