import subprocess
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Importar funciones personalizadas
from qseg.graph_utils import image_to_grid_graph, draw, draw_graph_cut_edges
from qseg.utils import decode_binary_string

# Dimensiones de la imagen
height, width = 3, 3

# Imagen de ejemplo
image = np.array([
       [0.82,  0.1, 0.99],
       [0.83,  0.2, 0.95],
       [0.1,  0.05, 0.98]
])

# Crear grafo de la imagen
normalized_nx_elist = image_to_grid_graph(image)
G = nx.grid_2d_graph(image.shape[0], image.shape[1])
G.add_weighted_edges_from(normalized_nx_elist)
draw(G, image)

# Convertir la imagen en formato QUBO
def create_qubo(image):
    h, w = image.shape
    Q = np.zeros((h*w, h*w))

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if i < h - 1:
                Q[idx, idx + w] = -1  # Penalización si los píxeles son distintos
            if j < w - 1:
                Q[idx, idx + 1] = -1
    return Q

qubo_matrix = create_qubo(image)

# Guardar el QUBO en un archivo CSV para Julia
np.savetxt("qubo_matrix.csv", qubo_matrix, delimiter=",")

# Ejecutar Julia para resolver el QUBO con QuantumAnnealing.jl
subprocess.run(["julia", "solve_qubo.jl"])

# Cargar la solución desde Julia
solution = np.loadtxt("solution.csv", dtype=int)

# Asegúrate de que la solución tiene la forma correcta
segmentation_mask = solution.reshape(height, width)

# Mostrar la segmentación resultante
plt.imshow(segmentation_mask, cmap=plt.cm.gray)

# Dibujar los bordes de corte
cut_edges = [(u, v) for (u, v, d) in G.edges(data=True) if segmentation_mask[u] != segmentation_mask[v]]
draw_graph_cut_edges(G, image, cut_edges)


