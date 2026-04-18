import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3
from thermal import Thermal, Thermal_P1_extra
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_t = env.CONFIG_T

    #test_sparse = Thermal(config_t, method = "sparse")
    #test_cholesky = Thermal(config_t, method = "cholesky")
    #test_sparse.run(print_info = True, plot = True)
    #test_cholesky.run(print_info = True, plot= True)

    #test_P1 = Thermal_P1(config_t, method="sparse")
    #test_P1.complexity_analysis(print_info=True, plot=True)
    #test_P1.run(print_info = True, plot = True)
    
    #test_P2 = Thermal_P2(config_t, method="sparse")
    #test_P2.run(print_info = False, plot = True)

    #test_P3 = Thermal_P3(config_t, method="sparse")
    #test_P3.run(print_info = False, plot = True)
    
    test_jacobi = Thermal_P1_extra(config_t, method = "jacobi")
    test_gauss_seidel = Thermal_P1_extra(config_t, method = "gauss_seidel")
    test_jacobi.run(print_info = True, plot = True)
    test_gauss_seidel.run(print_info = True, plot = True)

    return


if __name__ == "__main__":
    main()
