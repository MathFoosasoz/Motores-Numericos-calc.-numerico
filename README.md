# Cálculo Numérico Aplicado a Redes Hidráulicas

**Disciplina:** SME0602 — Motores Numéricos para Simulação em Engenharia  
**Instituto:** ICMC-USP  
**Professor:** Roberto F. Ausas

---

## Visão Geral

Este projeto implementa um **simulador de redes hidráulicas** em Python, modelando a distribuição de pressão e vazão em redes de microcanais interconectados. A abordagem é baseada em **álgebra linear computacional**: a rede é representada como um grafo, e o comportamento físico do fluido é descrito por um sistema de equações lineares resolvido numericamente.

O código está organizado em três arquivos principais:

| Arquivo | Papel |
|---|---|
| `hydraulics.py` | Classes que modelam e resolvem a rede |
| `ploting.py` | Funções de visualização gráfica |
| `app.py` | Ponto de entrada — configura e executa a simulação |

---

## Estrutura dos Arquivos

```
├── app.py                   # Executa a simulação
├── hydraulics.py            # Física e álgebra linear da rede
├── ploting.py               # Visualização dos resultados
└── env.py                   # Parâmetros de configuração (CONFIG)
```

---

## Conceitos Físicos e Matemáticos

### O que é uma rede hidráulica?

Uma rede hidráulica é um conjunto de **nós** (pontos de conexão) ligados por **canos** (microcanais). Um fluido entra na rede por um ou mais pontos de entrada (*inlet*) e sai por um ponto de saída (*outlet*). O objetivo é calcular:

- **Pressão em cada nó** — P [Pa]
- **Vazão em cada cano** — Q [m³/s]
- **Potência dissipada** — W [W]

### Lei de Hagen-Poiseuille

A relação entre pressão e vazão em cada cano segue a Lei de Hagen-Poiseuille para escoamento laminar em dutos:

```
Q_k = C_k * (P_i - P_j)
```

onde `C_k` é a **condutância hidráulica** do cano k, dada por:

```
C_k = π * D⁴ / (128 * μ * L_k)
```

- `D` — diâmetro hidráulico do cano (calculado a partir da área da seção transversal)
- `μ` — viscosidade dinâmica do fluido
- `L_k` — comprimento do cano k (distância euclidiana entre os nós)

### Formulação matricial

Aplicando conservação de massa em cada nó , obtemos um sistema linear:

```
A * P = b
```

onde:
- `A` — matriz de condutâncias (assemblada a partir da topologia da rede)
- `P` — vetor de pressões incógnitas em cada nó
- `b` — vetor de condições de contorno (vazão imposta no inlet, pressão imposta no outlet)

---

## Arquivo: `env.py` — Configuração

```python
CONFIG = {
    "N_INLET": 0,              # Índice do nó de entrada de vazão
    "INLET_FLOW": 1.0e-7,      # Vazão de entrada [m³/s]
    "INLET_PRESSURE": 100,     # Pressão de entrada [Pa] (para Problema 3)
    "N_OUTLET": 5,             # Índice do nó de saída (pressão = OUTLET)
    "OUTLET": 0,               # Pressão no ponto de saída [Pa]
    "PIPE_AREA": 0.00000025,   # Área da seção transversal dos canos [m²]
    "VISCOSITY": 0.001,        # Viscosidade do fluido [Pa·s]
    ...
}
```

Esse dicionário centraliza todos os parâmetros físicos e é passado para as classes de simulação.

---

## Arquivo: `hydraulics.py` — Classes de Simulação

### Classe `Hydraulics` (caso base)

#### `__init__(self, conec, Xno, config)`

**O que faz:** Inicializa a rede hidráulica com sua geometria e parâmetros.

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `conec` | array (N_canos × 2) | Matriz de conectividade — cada linha `[i, j]` define que o cano k vai do nó i ao nó j |
| `Xno` | array (N_nos × 2) | Coordenadas (x, y) de cada nó no espaço [m] |
| `config` | dict | Dicionário de parâmetros (do `env.py`) |

**O que armazena:**
- `num_nodes` — número total de nós (máximo índice da `conec` + 1)
- `num_pipes` — número de canos (linhas da `conec`)
- `node_outlet` / `node_inlet` — índices dos nós de saída e entrada
- `results` — dicionário que vai guardar P, Q, W após a simulação

---

#### `calculate_conductancy(self)`

**O que faz:** Calcula a condutância hidráulica de cada cano usando a Lei de Hagen-Poiseuille.

**Passo a passo:**

1. **Calcula o diâmetro hidráulico** a partir da área da seção transversal:
   ```python
   D = sqrt(4 * pipe_area / π)
   ```

2. **Calcula a constante geométrica** que depende apenas do diâmetro e viscosidade:
   ```python
   const_K = π * D⁴ / (128 * μ)
   ```

3. **Para cada cano**, calcula o comprimento L entre os nós de origem e destino:
   ```python
   L_k = sqrt((x_start - x_end)² + (y_start - y_end)²)
   ```

4. **Condutância do cano k:**
   ```python
   C[k] = const_K / L_k
   ```
   Canos mais curtos têm maior condutância (deixam o fluido passar com mais facilidade).

**Retorna:** Vetor `C` com a condutância de cada cano.

---

#### `assembly(self)`

**O que faz:** Monta a **matriz de condutâncias global** `A` da rede, que representa as relações de pressão entre todos os nós.

**Raciocínio físico:** Para cada cano k que conecta o nó `i` ao nó `j` com condutância `C_k`, a conservação de massa exige que a soma das vazões em cada nó seja zero. Isso resulta nas regras de preenchimento:

| Posição na matriz | Operação | Motivo |
|---|---|---|
| `A[i, i] += C_k` | Soma na diagonal do nó de origem | Contribuição direta |
| `A[j, j] += C_k` | Soma na diagonal do nó de destino | Contribuição direta |
| `A[i, j] -= C_k` | Subtrai na posição cruzada | Acoplamento entre nós |
| `A[j, i] -= C_k` | Subtrai na posição cruzada | Acoplamento entre nós |

A matriz resultante é **simétrica**.

**Retorna:** Matriz `A` de dimensão (num_nodes × num_nodes).

---

#### `solve_network(self)`

**O que faz:** Impõe as condições de contorno e resolve o sistema linear para encontrar as pressões em cada nó.

**Passo a passo:**

1. **Chama `assembly()`** para obter a matriz `A`.

2. **Condição de pressão no outlet** — o nó de saída tem pressão conhecida (normalmente zero). Para isso, zera-se a linha correspondente da matriz e coloca-se 1 na diagonal:
   ```
   A[node_outlet, :] = 0
   A[node_outlet, node_outlet] = 1
   ```
   Isso substitui a equação de conservação desse nó pela equação: `P[node_outlet] = OUTLET`

3. **Monta o vetor b** — zeros em todos os nós, exceto:
   - `b[node_inlet] = INLET_FLOW` — vazão imposta no inlet
   - `b[node_outlet] = OUTLET` — pressão imposta no outlet

4. **Resolve o sistema linear:**
   ```python
   P = np.linalg.solve(A, b)
   ```
   O NumPy usa eliminação gaussiana para resolver o sistema.

**Retorna:** Vetor `pressures` com a pressão em cada nó.

---

#### `calculate_flow_rate_and_potency(self)`

**O que faz:** Com as pressões calculadas, determina a **vazão em cada cano** e a **potência dissipada** pela rede.

**Passo a passo:**

1. **Chama `solve_network()`** para obter as pressões.

2. **Monta a matriz diagonal de condutâncias `K`** (num_pipes × num_pipes):
   ```
   K[k, k] = C[k]
   ```

3. **Monta a matriz de incidência `D`** (num_pipes × num_nodes) — indica a topologia da rede:
   - `D[k, j] = +1` se o nó j é a **origem** do cano k
   - `D[k, j] = -1` se o nó j é o **destino** do cano k
   - `D[k, j] =  0` se o nó j não está conectado ao cano k

4. **Calcula as vazões** em cada cano:
   ```
   Q = K @ D @ P
   ```
   Equivalente a: `Q_k = C_k * (P_origem - P_destino)`

5. **Calcula a potência dissipada** total:
   ```
   W = Pᵀ @ Dᵀ @ Q
   ```
   Representa o trabalho total realizado contra o atrito viscoso em todos os canos.

**Retorna:** Tupla `(Q, W)`.

---

#### `run(self, print_info, plot)`

**O que faz:** Executa toda a simulação e exibe os resultados.

- Se `print_info=True` → imprime pressões, vazões e potência no console
- Se `plot=True` → chama `PlotaRede()` para gerar o gráfico da rede

---
## Arquivo: `ploting.py` — Visualização

### Função `PlotaRede(conec, Xno, p, q)`

**O que faz:** Gera um gráfico da rede hidráulica com:
- Nós coloridos pela pressão (mapa de cores *coolwarm*)
- Setas nos canos indicando o sentido do fluxo
- Rótulos com os valores de pressão em cada nó
- Rótulos com os valores de vazão em cada cano
- Barra de cores lateral indicando a escala de pressão

**Parâmetros:**

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `conec` | array (N_canos × 2) | Conectividade da rede |
| `Xno` | array (N_nos × 2) | Coordenadas dos nós |
| `p` | array (N_nos,) | Pressão em cada nó [Pa] |
| `q` | array (N_canos,) | Vazão em cada cano [m³/s] |

**Passo a passo interno:**

1. **Calcula segmentos e pontos médios** de cada cano.
2. **Plota os nós** com cores proporcionais à pressão usando `scatter()`.
3. **Plota as arestas** (canos) como linhas pretas.
4. **Desenha setas** no ponto médio de cada cano, apontando do nó de maior para o de menor pressão.
5. **Adiciona rótulos** de pressão nos nós e vazão nos canos.
6. **Adiciona barra de cores** lateral.
### Função `PlotaMaxPressao(pressures, time_constants)`

**O que faz:** Plota a evolução temporal da **pressão máxima** na rede.

**Parâmetros:**

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `pressures` | array (N_tempos,) | Pressão máxima em cada instante |
| `time_constants` | lista [t_ini, t_fim, N] | Parâmetros do intervalo de tempo |

Gera um gráfico de linha simples: eixo x = tempo [s], eixo y = pressão [Pa].



## Conceitos-Chave de Álgebra Linear Usados

| Conceito | Onde aparece |
|---|---|
| Montagem de sistema linear | `assembly()` |
| Condições de contorno (Dirichlet) | `solve_network()` |
| Eliminação gaussiana | `np.linalg.solve()` |
| Produto matricial | `Q = K @ D @ P` |
| Princípio da superposição | `Hydraulics_p4`, `Hydraulics_p5` |
| Matriz de incidência de grafo | `matriz_D` em `calculate_flow_rate_and_potency()` |
