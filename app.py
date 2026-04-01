import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():
    config_h = env.CONFIG_H
    config_t = env.CONFIG_T

    #Xno, conec = GeraGrafo(levels=3)
    #mm_to_m = 0.001
    #Xno = Xno * mm_to_m

    #test = Hydraulics(conec, Xno, config_h)
    #test.run(print_info = True, plot = True)
    #complexity_analysis()

    #test_p1 = Hydraulics_p1(conec, Xno, config_h)
    #test_p1.run(print_info = True, plot = True)
    #complexity_analysis()

    #test_p2 = Hydraulics_p2(conec, Xno, config_h)
    #test_p2.run(print_info = True, plot = True)
    #complexity_analysis()
  
    #test_p3 = Hydraulics_p3(conec, Xno, config_h)
    #test_p3.run(print_info = True, plot = False)
    #complexity_analysis()

    #test_p4 = Hydraulics_p4(conec, Xno, config_h)
    #test_p4.run(print_info = True, plot = True)

    #test_p5 = Hydraulics_p5(conec, Xno, config_h)
    #test_p5.run(print_info = True, plot = True)

    #test_p6 = Hydraulics_p6(conec, Xno, config_h)
    #test_p6.run(print_info = True, plot = True)
    #complexity_analysis()

    test_t = Thermal(config_t, 4)
    t = test_t.SolveSystem()
    test_t.print_temp(t)


if __name__ == "__main__":
    main()
