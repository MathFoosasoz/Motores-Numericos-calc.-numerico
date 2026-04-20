import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal, Thermal_P1, Thermal_P2, Thermal_P3, Thermal_P4, Thermal_P5, Thermal_P1_extra, Thermal_P2_Extra, Thermal_P3_Extra
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_t = env.CONFIG_T

    #FUNCIONA!!!
    #test_sparse = Thermal(config_t, method = "sparse")
    #test_sparse.run(print_info = True, plot = True)

    #FUNCIONA!!!
    #test_cholesky = Thermal(config_t, method = "cholesky")
    #test_cholesky.run(print_info = True, plot= True)

    #FUNCIONA!!! 
    #test_P1 = Thermal_P1(config_t, method="sparse")
    #test_P1.complexity_analysis(print_info=True, plot=True)
    #test_P1.run(print_info = True, plot = True)
    
    #FUNCIONA!!!
    #test_P2 = Thermal_P2(config_t, method="sparse")
    #test_P2.run(print_info = False, plot = True)

    #FUNCIONA!!!
    #test_P3 = Thermal_P3(config_t, method="sparse")
    #test_P3.run(print_info = False, plot = True)

    #FUNCIONA!!!
    #test_P4 = Thermal_P4(config_t)
    #test_P4.run(plot=True)

    #FUNCIONA!!!
    #test_P5 = Thermal_P5(config_t, k_node=233)
    #test_P5.run(print_info=True)
    
    #FUNCIONA !!!
    #test_jacobi = Thermal_P1_extra(config_t, method = "jacobi")
    #test_gauss_seidel = Thermal_P1_extra(config_t, method = "gauss_seidel")
    #test_jacobi.run(print_info = True, plot = True)
    #test_gauss_seidel.run(print_info = True, plot = True)
    
    #test_gauss_seidel.run(print_info = True, plot = True, analyze_subdivisions=True, analyze_tolerance=True)

    #FUNCIONA !!!
    #simulacao = Thermal_P2_Extra(config_t)
    #res_j, res_gs = simulacao.run_comparison_history(frame_step=20)
    #simulacao.animate_comparison(interval=50)

    #FUNCIONA !!!
    #test_P3_extra = Thermal_P3_Extra(config_t)
    #test_P3_extra.run(print_info = True, plot=False)
    
    return


if __name__ == "__main__":
    main()
