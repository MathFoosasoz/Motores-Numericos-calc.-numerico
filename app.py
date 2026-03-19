from env import CONFIG
from hydraulics import Hydraulics, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5
from data_structures import GeraGrafo

def main():
    config = CONFIG

    Xno, conec = GeraGrafo(levels=4)
    mm_to_m = 0.001
    Xno = Xno * mm_to_m

    test = Hydraulics(conec, Xno, config)
    test.run(print_info = True, plot = True)

    test_p2 = Hydraulics_p2(conec, Xno, config)
    test_p2.run(print_info = True, plot = True)
  
    #test_p3 = Hydraulics_p3(filtered_conec, Xno, config)
    #test_p3.run(print_info = True, plot = False)

    #test_p4 = Hydraulics_p4(filtered_conec, Xno, config)
    #test_p4.run(print_info = True, plot = True)

    test_p5 = Hydraulics_p5(conec, Xno, config)
    test_p5.run(print_info = True, plot = True)

    # -----------------------------------------------------------------------
    # Exerc�cio 1  m�ltiplos pontos de inje��o via INLET_FLOW_DICT
    # O CONFIG j� possui a chave INLET_FLOW_DICT com dois n�s de entrada.
    # Basta instanciar Hydraulics_ex1 passando o mesmo config.
    # -----------------------------------------------------------------------
    test_ex1 = Hydraulics_ex1(conec, Xno, config)
    test_ex1.run(print_info = True, plot = True)


if __name__ == "__main__":
    main()
