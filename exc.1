import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from hydraulics import Hydraulics


class HydraulicsIFE(Hydraulics):

    def calculate_conductancy(self):
        self.C = np.ones(self.num_pipes)
        return self.C

    def montar_A_invertivel(self):
        A = self.assembly()

        A[0, :] = 0.0
        A[0, 0] = 1.0

        print(A)
        print(np.linalg.det(A))
        return A


def montar_malha_membrana(Lx=0.01, Ly=0.01, Nx=26, Ny=26, raio=0.004):
    h = Lx / (Nx - 1)
    nm = Nx * Ny

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    mask = ((X - 0.5 * Lx)**2 + (Y - 0.5 * Ly)**2 > raio**2).astype(int)

    uns = np.ones(nm)
    uns[mask.flatten() == 1] = 0.0

    print(nm)
    print(h)
    print(int(uns.sum()))
    print(nm - int(uns.sum()))

    return h, nm, uns, X.flatten(), Y.flatten()


def montar_matriz_U(np_nodes, nm, uns, nout=5):
    U = np.zeros((np_nodes, nm))
    U[nout, :] = uns
    
    print(int((U != 0).sum()))
    return U


def calcular_R_esparso(h, U, A):
    A_inv = np.linalg.inv(A)

    AinvU = A_inv @ U

    R_dense = (h**2) * (U.T @ AinvU)

    R_sparse = sp.csr_matrix(R_dense)
    R_sparse.eliminate_zeros()

    nnz = R_sparse.nnz
    nm  = R_dense.shape[0]
    
    print(nnz)
    print(100 * nnz / nm**2)

    return R_sparse


def visualizar_esparsidade(R_sparse, uns, Nx=26, Ny=26, raio=0.004, Lx=0.01):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.spy(R_sparse, markersize=0.4, marker='.', color='steelblue')

    ax2 = axes[1]
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Lx, Ny)
    X, Y = np.meshgrid(x, y)
    livre = uns.reshape(Ny, Nx)

    ax2.contourf(X * 1000, Y * 1000, livre, levels=[-0.5, 0.5, 1.5],
                 colors=['#d9534f', '#5cb85c'], alpha=0.7)
    theta = np.linspace(0, 2 * np.pi, 300)
    cx, cy = 0.5 * Lx * 1000, 0.5 * Lx * 1000
    ax2.plot(cx + raio * 1000 * np.cos(theta),
             cy + raio * 1000 * np.sin(theta),
             'k-', lw=2)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("fsi_esparsidade_R.png", dpi=150, bbox_inches='tight')
    plt.show()


def verificar_positividade(R_sparse, uns):
    idx_livres = np.where(uns == 1.0)[0]
    R_dense = R_sparse.toarray()
    R_livre = R_dense[np.ix_(idx_livres, idx_livres)]

    autovalores = np.linalg.eigvalsh(R_livre)
    
    print(autovalores.min())
    print(autovalores.max())
    print(np.linalg.matrix_rank(R_livre))


def criar_rede_hidraulica_ife(np_nos=6):
    conec = np.array([[i, i + 1] for i in range(np_nos - 1)])
    Xno   = np.array([[float(i), 0.0] for i in range(np_nos)])
    return conec, Xno


def run(plot=True, print_info=True):

    Lx    = Ly   = 0.01
    Nx    = Ny   = 26
    raio  = 0.004
    np_nos = 6
    nout  = 5

    h, nm, uns, x_flat, y_flat = montar_malha_membrana(Lx, Ly, Nx, Ny, raio)

    U = montar_matriz_U(np_nos, nm, uns, nout=nout)

    conec, Xno = criar_rede_hidraulica_ife(np_nos)

    config_ife = {
        "N_OUTLET": nout,
        "N_INLET": 0,
        "INLET_FLOW": 1.0,
        "OUTLET": 0.0,
        "PIPE_AREA": 0.00000025,
        "VISCOSITY": 0.001,
    }

    rede = HydraulicsIFE(conec, Xno, config_ife)
    A = rede.montar_A_invertivel()

    R_sparse = calcular_R_esparso(h, U, A)

    if print_info:
        verificar_positividade(R_sparse, uns)

    if plot:
        visualizar_esparsidade(R_sparse, uns, Nx, Ny, raio, Lx)
