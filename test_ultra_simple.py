import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from garnier_nn import GarnierLayer
from typing import Tuple, Dict
import logging
import time

# Test ultra-simple para identificar cuello de botella
print("🔥 TEST ULTRA-SIMPLE RESMA-GARNIER")
print("="*50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

# Test 1: Solo crear GarnierLayer sin topología
print("\n📊 Test 1: Crear GarnierLayer simple")
start_time = time.time()
layer = GarnierLayer(784, 64, device)
creation_time = time.time() - start_time
print(f"⏱️  Tiempo creación GarnierLayer: {creation_time:.3f}s")

# Test 2: SVD isolated
print("\n📊 Test 2: SVD isolated")
start_time = time.time()
U, S, Vh = torch.linalg.svd(layer.weight, full_matrices=False)
svd_time = time.time() - start_time
print(f"⏱️  Tiempo SVD: {svd_time:.3f}s")

# Test 3: Entropía isolated
print("\n📊 Test 3: Entropía isolated")
x_dummy = torch.randn(32, 784, device=device)  # Batch pequeño
x_dummy = x_dummy.view(32, -1)
pre_activation = (U @ Vh) @ x_dummy.T
output = torch.relu(pre_activation.T)

start_time = time.time()
rho = torch.softmax(output @ output.T, dim=-1)
S_vn = -torch.sum(rho * torch.log(rho + 1e-12))
entropia_time = time.time() - start_time
print(f"⏱️  Tiempo entropía von Neumann: {entropia_time:.3f}s")

# Test 4: Topología BA+WS
print("\n📊 Test 4: Topología BA+WS")
scale = 100  # Muy pequeña para test
start_time = time.time()
G_ba = nx.barabasi_albert_graph(scale, m=3)
G_ws = nx.watts_strogatz_graph(scale, k=3, p=0.1)
G = nx.compose(G_ba, G_ws)
density = nx.density(G)
topo_time = time.time() - start_time
print(f"⏱️  Tiempo construcción topología (scale={scale}): {topo_time:.3f}s")
print(f"🌐 Densidad: {density:.3f}")

print("\n" + "="*50)
print("TEST COMPLETADO - Si algún test falló, ahí está el cuello de botella")