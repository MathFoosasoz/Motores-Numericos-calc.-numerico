import numpy as np
import matplotlib.pyplot as plt

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