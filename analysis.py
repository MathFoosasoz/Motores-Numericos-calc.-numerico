import time    #pra medir tempo
import numpy as np
from hydraulics import Hydraulics
from data_structures import GeraGrafo
from env import CONFIG
import matplotlib.pyplot as plt

def main():
    levels_list = [1, 2, 3, 4]   #os levels que foram pedidos no arquivo do professor

    results = []

    for level in levels_list:
        tempos_assembly = []    #tempo de montagem da matriz
        tempos_solve = []       #tempo de resolucao  
        
        for _ in range(10):  # repete 10 vezes - pra diminuir o erro e tirar uma media depois
            # gera a rede
            Xno, conec = GeraGrafo(levels=level)
            Xno = Xno * 0.001  # mesma conversão que usamos no projeto
            
            h = Hydraulics(conec, Xno, CONFIG)
            
            #medindo tempo de montagem da matriz
            start = time.time()
            h.Assembly()
            tempos_assembly.append(time.time() - start)
            
            #medindo tempo de resolucao do sistema linear
            start = time.time()
            h.solveNetwork()
            tempos_solve.append(time.time() - start)
        
        #pra mostrar o resultado
        results.append({
            "level": level,                              #nivel
            "nodes": h.num_nodes,                        #nro de nos    
            "assembly_time": np.mean(tempos_assembly),   #tempo medio de montagem da matriz
            "solve_time": np.mean(tempos_solve)          #tempo medio de resolucao do sistema
        })
        #np.mean e usado pra calcular a media entre as 10 repeticoes

    print("\nRESULTADOS:\n")
    for r in results:
        print(f"Level: {r['level']}")
        print(f"Nós: {r['nodes']}")
        print(f"Tempo montagem: {r['assembly_time']:.6f} s")
        print(f"Tempo resolução: {r['solve_time']:.6f} s")
        print("-" * 30)


     # GERAR TABELA
    colunas = ["Level", "Nós", "Montagem (s)", "Resolução (s)"]

    dados = []
    for r in results:
        dados.append([
            r["level"],
            r["nodes"],
            f"{r['assembly_time']:.6f}",
            f"{r['solve_time']:.6f}"
        ])

    fig, ax = plt.subplots()
    ax.axis('off')

    tabela = ax.table(
        cellText = dados,
        colLabels = colunas,
        loc = 'center',
        cellLoc = 'center',
        colWidths=[0.15, 0.15, 0.25, 0.25] 
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.8)

    plt.show()

if __name__ == "__main__":
    main()