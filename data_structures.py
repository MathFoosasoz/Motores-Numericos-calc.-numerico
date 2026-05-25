import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree

def GeraGrafo(levels=3):
    nodes_data = []
    edges_raw = []
    node_id = 0

    spine_length = 6

    # 1. Geração da Estrutura Base
    spine_nodes = []
    for i in range(spine_length):
        nodes_data.append({'x': i * 4.0, 'y': 0.0})
        spine_nodes.append(node_id)
        if i > 0: edges_raw.append((node_id - 1, node_id))
        node_id += 1

    def add_fractal_branches(parent_id, px, py, angle, length, depth):
        nonlocal node_id
        if depth == 0: return
        angles = [angle + np.pi/6, angle - np.pi/6]
        branch_len = length * 0.75
        for a in angles:
            nx = px + branch_len * np.cos(a)
            ny = py + branch_len * np.sin(a)
            curr_id = node_id
            nodes_data.append({'x': nx, 'y': ny})
            edges_raw.append((parent_id, curr_id))
            node_id += 1
            add_fractal_branches(curr_id, nx, ny, a, branch_len, depth - 1)

    for s_id in spine_nodes[1:-1]:
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], np.pi/2, 3.0, levels)
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], -np.pi/2, 3.0, levels)

    # 2. Adição das Coletoras (Manifolds)
    df_temp = pd.DataFrame(nodes_data)
    y_max, y_min = df_temp['y'].max() + 1.0, df_temp['y'].min() - 1.0

    all_indices = [e[0] for e in edges_raw] + [e[1] for e in edges_raw]
    counts = pd.Series(all_indices).value_counts()
    leaf_ids = counts[counts == 1].index.tolist()
    leaf_ids = [idx for idx in leaf_ids if idx not in [spine_nodes[0], spine_nodes[-1]]]

    for l_id in leaf_ids:
        target_y = y_max if nodes_data[l_id]['y'] > 0 else y_min
        new_id = node_id
        nodes_data.append({'x': nodes_data[l_id]['x'], 'y': target_y})
        edges_raw.append((l_id, new_id))
        node_id += 1

    # 3. Processamento Geométrico de Interseções
    lines = [LineString([(nodes_data[e[0]]['x'], nodes_data[e[0]]['y']),
                         (nodes_data[e[1]]['x'], nodes_data[e[1]]['y'])]) for e in edges_raw]

    # Adicionar linhas das coletoras para o merge
    df_nodes_final = pd.DataFrame(nodes_data)
    for y_lim in [y_max, y_min]:
        pts = df_nodes_final[df_nodes_final['y'] == y_lim].sort_values('x')
        if len(pts) > 1:
            lines.append(LineString(pts[['x', 'y']].values))

    # Quebra todas as linhas nas interseções
    merged_graph = unary_union(lines)

    # 4. Conversão para Arrays NumPy (Mapeamento de IDs)
    final_nodes_map = {}
    final_nodes_list = []
    final_edges_list = []

    def get_node_id(pt):
        # Arredondamento para evitar erros de precisão de ponto flutuante
        coords = (round(pt[0], 6), round(pt[1], 6))
        if coords not in final_nodes_map:
            final_nodes_map[coords] = len(final_nodes_list)
            final_nodes_list.append([pt[0], pt[1]])
        return final_nodes_map[coords]

    # Itera sobre cada segmento gerado pelo unary_union
    segments = merged_graph.geoms if hasattr(merged_graph, 'geoms') else [merged_graph]
    for seg in segments:
        id_start = get_node_id(seg.coords[0])
        id_end = get_node_id(seg.coords[-1])
        final_edges_list.append([id_start, id_end])

    nodes_np = np.array(final_nodes_list)
    edges_np = np.array(final_edges_list)
    mask = edges_np[:, 0] != edges_np[:, 1]
    edges_np = edges_np[mask]

    return nodes_np, edges_np

def conec_filter(conec):
    clean_conec = []

    for index, connection in enumerate(conec):
        if connection[0] != connection[1]:
            clean_conec.append([connection[0], connection[1]])
            
    clean_conec_np = np.array(clean_conec)

    for index, connection in enumerate(clean_conec_np):
        if connection[0] == connection[1]:
            print(f"NA MATRIZ LIMPA: index: {index}! Nó de saída é igual ao nó de chegada!")

    return clean_conec_np

def show_conec_issues(conec):

    print(f"[{conec[0]}")
    print(f" {conec[1]}")
    print(f" {conec[2]}")
    print(f" ...  \n")

    for index, connection in enumerate(conec):
        if connection[0] == connection[1]:
            print(f" ...  ")
            print(f"{conec[index - 2]}")
            print(f"{conec[index - 1]}")
            print(f"{conec[index]}")
            print(f"{conec[index + 1]}")
            print(f"{conec[index + 2]}")
            print(f" ...  \n")

    print(f" ...  ")
    print(f"[{conec[-3]}")
    print(f" {conec[-2]}")
    print(f" {conec[-1]}]")
            

def calcular_distancia_ponto_segmento(p, a, b):
    """Calcula a menor distância entre um ponto 'p' e o segmento 'ab'."""
    ab = b - a
    ap = p - a
    ab_2 = np.dot(ab, ab)
    if ab_2 == 0:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_2
    t = np.clip(t, 0.0, 1.0)
    ponto_proximo = a + t * ab
    return np.linalg.norm(p - ponto_proximo)

def CreateMapDistance(Lx, Ly, Nx, Ny, nos_rede, conexoes_rede, d_max):
    """
    Mapeia as arestas próximas a cada ponto da grade utilizando cKDTree.
    """
    # 1. Gerar os pontos da grade bidimensional da placa térmica
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    pontos_grade = np.column_stack((X.ravel(), Y.ravel()))
    
    # 2. CONSTRUÇÃO DA ÁRVORE KD (Onde o KDTree é usado!)
    # Construímos a árvore espacial indexando todos os pontos da grade da placa
    arvore_grade = cKDTree(pontos_grade)
    
    # Inicializa a estrutura de retorno
    mapa_proximidade = [[] for _ in range(Nx * Ny)]
    
    # 3. BUSCA ESPACIAL EFICIENTE
    for id_aresta, (idx_i, idx_j) in enumerate(conexoes_rede):
        p_i = nos_rede[idx_i]
        p_j = nos_rede[idx_j]
        
        # Encontramos o ponto médio da aresta e o seu raio de abrangência máximo
        ponto_medio = (p_i + p_j) / 2.0
        comprimento_aresta = np.linalg.norm(p_j - p_i)
        
        # O raio de busca na árvore engloba toda a aresta mais a distância d_max
        raio_busca = (comprimento_aresta / 2.0) + d_max
        
        # AQUI usamos o KDTree para filtrar rapidamente apenas os pontos da grade 
        # que estão dentro da esfera que envolve a aresta expandida
        indices_candidatos = arvore_grade.query_ball_point(ponto_medio, raio_busca)
        
        # 4. FILTRAGEM REFINADA (Apenas para os pontos candidatos retornados pela árvore)
        for idx_ponto in indices_candidatos:
            ponto = pontos_grade[idx_ponto]
            dist = calcular_distancia_ponto_segmento(ponto, p_i, p_j)
            
            if dist <= d_max:
                mapa_proximidade[idx_ponto].append((id_aresta, dist))
                
    return mapa_proximidade
