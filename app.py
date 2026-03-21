from env import CONFIG
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from data_structures import GeraGrafo

def main():
    config = CONFIG

    Xno, conec = GeraGrafo(levels=4)
    mm_to_m = 0.001
    Xno = Xno * mm_to_m

    # test = Hydraulics(conec, Xno, config)
    # test.run(print_info = True, plot = True)
    # complexity_analysis(test, print_info = True, plot = True)

    # test_p1 = Hydraulics_p1(conec, Xno, config)
    # test_p1.run(print_info = True, plot = True)

    # test_p2 = Hydraulics_p2(conec, Xno, config)
    # test_p2.run(print_info = True, plot = True)
  
    # test_p3 = Hydraulics_p3(conec, Xno, config)
    # test_p3.run(print_info = True, plot = False)

    # test_p4 = Hydraulics_p4(conec, Xno, config)
    # test_p4.run(print_info = True, plot = True)

    # test_p5 = Hydraulics_p5(conec, Xno, config)
    # test_p5.run(print_info = True, plot = True)

    test_p6 = Hydraulics_p6(conec, Xno, config)
    test_p6.run(print_info = True, plot = True)


if __name__ == "__main__":
    main()
