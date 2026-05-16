import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# import tkinter as tk
# import matplotlib
# matplotlib.use('TkAgg')

def PlotaRede(conec, Xno, p, q):

    edges = conec
    coord = Xno
    nv = np.max(np.max(conec))+1
    nc = conec.shape[0]

    # Internal: get edge and midpoint coordinates
    segs = []
    mids = []
    for (i, j) in edges:
      x1, y1 = coord[i,0], coord[i,1]
      x2, y2 = coord[j,0], coord[j,1]
      segs.append(((x1, y1), (x2, y2)))
      mids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    segs = np.array(segs)
    mids = np.array(mids)

    fig_size=(10, 10)
    cmap_name="coolwarm"
    node_size=500
    show_flux_labels=True
    arrow_scale=0.05
    text_scale=1.1
    save_path=None

    fig, ax = plt.subplots(figsize=fig_size)

    # ---- Pressure colormap ----
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = float(p.min()), float(p.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    xs, ys = [], []
    for i in range(nv):
      xs.append(coord[i,0])
      ys.append(coord[i,1])
    colors = [cmap(norm(pi)) for pi in p]
    ax.scatter(xs, ys, s=node_size, c=colors, zorder=3, edgecolors="black")

    # ---- Draw black edges and arrows ----
    #segs, mids = edge_coords()
    for idx, ((x1, y1), (x2, y2)) in enumerate(segs):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=3.0, zorder=1)

        xm, ym = mids[idx]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
          continue
        dxn, dyn = dx / L, dy / L
        nx, ny = -dyn, dxn  # normal vector

        # --- Flux arrow (black) ---
        p1, p2 = p[edges[idx,0]-1], p[edges[idx,1]-1]
        q_dir = 1 if p1 > p2 else -1

        ax.annotate(
              "",
              xy=(xm + q_dir * 1.5 * arrow_scale * dxn, ym + q_dir * 1.5 * arrow_scale * dyn),
              xytext=(xm - q_dir * 1.5 * arrow_scale * dxn, ym - q_dir * 1.5 * arrow_scale * dyn),
              arrowprops=dict(
              arrowstyle="-|>",
              color="black",
              lw=1.5,
              mutation_scale=20 * text_scale,  # scales arrowhead size
              ),
              zorder=5,
              )

        # --- Flux label ---
        if show_flux_labels:
          label_offset = 0.0725
          ax.text(
              xm + nx * label_offset,
              ym + ny * label_offset,
              f"q={q[idx]:.1e}",
              ha="center",
              va="center",
              fontsize=12 * text_scale,
              zorder=6,
              bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0)
              )

    # ---- Node labels ----
    for node, (x, y) in enumerate(coord):
        ax.text(x, y, str(node),
                ha="center", va="center", fontsize=11 * text_scale, zorder=4)
        ax.text(x - 0.075, y - 0.075, f"p={p[node]:.1e}",
                ha="right", va="bottom", fontsize=12 * text_scale,
                color="black", zorder=5)

    # ---- Final adjustments ----
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Set limits with small margin ----
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range, y_range = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # Optionally adapt figure size to graph geometry
    aspect_ratio = x_range / y_range if y_range != 0 else 1.0
    base_size = 8  # base figure size
    fig.set_size_inches(base_size * aspect_ratio, base_size)

    # ---- Colorbar ----
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, label="Pressure, p", fraction=0.0225, pad=0.025)
    cbar.ax.tick_params(labelsize=10 * text_scale)
    cbar.set_label("Pressure, p", fontsize=12 * text_scale)

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches="tight")
      plt.show()

    return fig, ax


def PlotaMaxPressao(pressures, time_constants):

  time = np.linspace(time_constants[0], time_constants[1], time_constants[2])

  plt.figure(figsize=(8, 4)) 
  plt.plot(time, pressures, color='red')
  plt.xlabel("Tempo (s)")
  plt.ylabel("Pressão (Pa)")
  plt.title("Pressão máxima na rede de acordo com o tempo")
  plt.grid(True)


def PlotaPlaca(Nx, Ny, Lx, Ly, T, flag_type='contour', filename=None, Tmax=None):
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)

    X, Y = np.meshgrid(x, y)

    Z = np.copy(T).reshape(Ny, Nx)

    title = f'Contours of temperature, N=({Nx}, {Ny})'

    if Tmax is not None:
       T_max = np.max(T)
       title += f', Tmax = {T_max:.2f}ºC'

    if(flag_type == 'contour'):
      fig, ax = plt.subplots(figsize=(6,6))
      ax.set_aspect('equal')
      ax.set(xlabel='x', ylabel='y', title=f'Contours of temperature, N=({Nx}, {Ny})')
      im = ax.contourf(X, Y, Z, 20, cmap='jet')
      im2 = ax.contour(X, Y, Z, 20, linewidths=0.25, colors='k')
      fig.colorbar(im, ax=ax, orientation='horizontal')

    elif(flag_type == 'surface'):
      fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
      ax.set_aspect('equal')
      surf = ax.plot_surface(X, Y, Z, cmap='jet')
      fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) 
    
    plt.xticks([0, Lx/2, Lx])
    plt.yticks([0, Ly/2, Ly])

    if(filename is not None):
      plt.savefig(filename)

    plt.show()

    return


def PlotaEixoTemps(N, L_eixo, T, filename=None):
  x = np.linspace(0, L_eixo, N[0])
  y = T

  plt.plot(x, y)
  plt.title(f"Temperatura no eixo central, N = ({N[0]}, {N[1]})")
  plt.xlabel("Eixo placa (m)")
  plt.ylabel("Temperatura (ºC)")

  max_temp = -1
  max_temp_pos = -2

  for index, temp in enumerate(y):
    if temp> max_temp:
       max_temp = temp
       max_temp_pos = index

  plt.scatter(x[max_temp_pos], max_temp) 

  plt.annotate(f'Max: ({x[max_temp_pos]:.4f}, {max_temp:.2f})', 
             xy=(x[max_temp_pos], max_temp),  
             xytext=(5, 5),  
             textcoords='offset points') 
  
  plt.xticks([0, L_eixo/2, L_eixo])
  max_box = int(f"{(max_temp +10):.0f}")
  max_box = max_box + 10 - (max_box%10)
  y_space = np.linspace(0, max_box, max_box//10 +1)
  plt.yticks(y_space)

  if(filename is not None):
      plt.savefig(filename)

  plt.show() 


def plot_problem4(TC, Tmax, Tmean, filename=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.plot(TC, Tmax, label="Tmax")
    plt.plot(TC, Tmean, label="Tmean")

    plt.xlabel("Tc (°C)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Temperatura máxima e média vs Tc")

    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename)

    plt.show()
    
    
def plot_p1_extra_subdivisions(nodes_list, times_j, times_gs):
    plt.figure(figsize=(8, 5))
    plt.plot(nodes_list, times_j, marker='o', color='blue', label='Jacobi')
    plt.plot(nodes_list, times_gs, marker='s', color='orange', label='Gauss-Seidel')
    
    plt.title("Tempo de Execução vs Subdivisões")
    plt.xlabel("Subdivisões (Total de Nós)")
    plt.ylabel("Tempo (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_p1_extra_tolerance(tol_list, times_j, times_gs):
    plt.figure(figsize=(8, 5))
    plt.plot(tol_list, times_j, marker='o', color='blue', label='Jacobi')
    plt.plot(tol_list, times_gs, marker='s', color='orange', label='Gauss-Seidel')
    
    plt.title("Tempo de Execução vs Tolerância")
    plt.xlabel("Tolerância")
    plt.ylabel("Tempo (s)")
    plt.xscale('log') 
    plt.gca().invert_xaxis() 
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_p1_complex_analysis(dados, colunas):
  fig, ax = plt.subplots(figsize=(8, 4))
  ax.axis('off')

  tabela = ax.table(
      cellText = dados,
      colLabels = colunas,
      loc = 'center',
      cellLoc = 'center',
      colWidths=[0.2, 0.2, 0.25, 0.25] 
  )

  tabela.auto_set_font_size(False)
  tabela.set_fontsize(10)
  tabela.scale(1.2, 1.8)

  plt.title(f'Resultados de Complexidade: Thermal_P1', pad=20)
  plt.tight_layout()
  plt.show()


def plot_membrane_modes(N, n_modes, modes, freq):
  N0, N1 = N
  plt.figure(figsize=(15, 7))

  plt.suptitle(f"Modos de Vibração - Grade {N0}x{N1}", fontsize=14)
  
  for i in range(min(10,n_modes)):
      plt.subplot(2, 5, i + 1)
      mode_reshaped = modes[:, i].reshape((N1, N0))
      
      plt.contourf(mode_reshaped, cmap='RdYlBu', levels=20)
      plt.title(f"Modo {i+1}\n{freq[i]:.1f} Hz")
      plt.axis('off')
  
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  
def gera_grafico_energia_M_P5(omega_n, c, titulo="Ressonância e Energia Elástica Média"):
  
  peak_points = omega_n[(omega_n >= 0.5) & (omega_n <= 150)]
  w_log_spaced = np.logspace(np.log10(0.5), np.log10(150), 2000)
  all_w_stars = np.unique(np.sort(np.concatenate([w_log_spaced, peak_points])))
  
  betas = [0.01, 0.1, 1.0]
  colors = ['red', 'magenta', 'green']
    
  plt.figure(figsize=(10, 6))
    
  for beta, color in zip(betas, colors):
      energy_avg = np.zeros_like(all_w_stars)
        
      for k, w_star in enumerate(all_w_stars):
            
          denom = (omega_n**2 - w_star**2)**2 + (beta * w_star)**2
            
          Q2 = (c**2) / denom
          energy_avg[k] = 0.25 * np.sum((omega_n**2) * Q2)
            
      plt.loglog(all_w_stars, energy_avg, label=f'$\\beta$ = {beta}', color=color, linewidth=1.2)
        
  plt.xlabel('Frequência de Excitação $\\hat{\\omega}^*$')
  plt.ylabel('Energia Elástica Média $\\langle E \\rangle$')
  plt.title(titulo)
  plt.xlim(0.5, 150)
  plt.ylim(bottom=1e-5)
  plt.legend()
  plt.grid(True, which="both", ls="--", alpha=0.5)
  plt.tight_layout()
  plt.show()