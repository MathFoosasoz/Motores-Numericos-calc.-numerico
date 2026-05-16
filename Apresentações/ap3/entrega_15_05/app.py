import env
from mechanic import Mechanic, Mechanic_P2, Mechanic_P4

def main():

    config_m = env.CONFIG_M

    membrana_elastica = Mechanic(config_m)
    membrana_elastica.run(print_info=True, plot=False)

    
    analise_discretizacao = Mechanic_P2(config_m)
    analise_discretizacao.plot_convergence_table_image(config_m["MULTI_N"])
    analise_discretizacao.plot_modes(config_m["MULTI_N"])

    analise_oscilacao_forcada = Mechanic_P4(config_m)
    analise_oscilacao_forcada.run(print_info = True, plot = False)
    analise_oscilacao_forcada.P5_plot_average_elastic_energy()
        

    return


if __name__ == "__main__":
    main()
