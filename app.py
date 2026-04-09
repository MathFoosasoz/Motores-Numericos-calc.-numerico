import env
from hydraulics import Hydraulics, Hydraulics_p1, Hydraulics_p2, Hydraulics_p3, Hydraulics_p4, Hydraulics_p5, Hydraulics_p6, complexity_analysis
from thermal import Thermal
from data_structures import GeraGrafo
from analysis import complexity_analysis

def main():

    config_t = env.CONFIG_T

    test_sparse = Thermal(config_t, method = "sparse")
    test_cholesky = Thermal(config_t, method = "cholesky")
    test_sparse.run(print_info = True, plot = True)
    test_cholesky.run(print_info = True, plot= True)

    return


if __name__ == "__main__":
    main()
