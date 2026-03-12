import tkinter
import matplotlib
matplotlib.use("tkagg")

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
    test.run(print_info = True, plot = True)
  
    test_p3 = Hydraulics_p3(conec, Xno, config)
    test_p3.run(print_info = True, plot = True)

    test_p4 = Hydraulics_p4(conec, Xno, config)
    test_p4.run(print_info = True, plot = True)

    test_p5 = Hydraulics_p5(conec, Xno, config)
    test_p5.run(print_info = True, plot = True)
    

if __name__ == "__main__":
    main()
