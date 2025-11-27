"""
resma-core/physics.py v5.2.0 (High Flow)
========================================
Ajuste: Aumento de kappa_init para permitir fase Soberana (>90%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Tuple

class PTSymmetricActivation(nn.Module):
    def __init__(self, omega: float = 50e12, chi: float = 0.6, kappa_init: float = 8.0e10): # <--- CAMBIO AQUÍ
        super().__init__()
        self.omega = omega
        self.chi = chi
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        
        # Constantes calibradas
        self.ZEEMAN_SCALE = 1.2e-10
        self.GAIN_FACTOR = 1000.0
        self.MAX_AMPLITUDE = 20.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_safe = torch.clamp(x, min=-self.MAX_AMPLITUDE, max=self.MAX_AMPLITUDE)
        threshold = self.chi * self.omega
        coherence_ratio = self.kappa / (threshold + 1e-8)
        zeeman_term = torch.pow(torch.abs(x_safe), 8.0) * self.ZEEMAN_SCALE
        gate_status = torch.sigmoid(self.GAIN_FACTOR * (coherence_ratio - zeeman_term))
        x_out = x * gate_status
        return x_out, gate_status, coherence_ratio, zeeman_term

class E8LatticeLayer(nn.Module):
    # (El resto de la clase es idéntico a la versión anterior)
    def __init__(self, in_features: int, out_features: int, q_order: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_order = q_order // 2 
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('topology_mask', self._generate_ramsey_mask())

    def _generate_ramsey_mask(self) -> torch.Tensor:
        G = nx.barabasi_albert_graph(self.in_features, min(4, self.in_features-1))
        adj = nx.to_numpy_array(G)
        mask = torch.tensor(adj, dtype=torch.float32)
        if self.out_features != self.in_features:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(self.out_features, self.in_features), 
                mode='nearest'
            ).squeeze()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weights = self.weights * self.topology_mask
        linear_out = F.linear(x, masked_weights, self.bias)
        syk_correction = torch.pow(torch.abs(linear_out), self.q_order) * 1e-7
        return linear_out + syk_correction

class RESMABrain(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = E8LatticeLayer(input_dim, hidden_dim)
        self.act1 = PTSymmetricActivation()
        self.layer2 = E8LatticeLayer(hidden_dim, hidden_dim)
        self.act2 = PTSymmetricActivation()
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x, _, _, _ = self.act1(x)
        x = self.layer2(x)
        x, _, _, _ = self.act2(x)
        return self.readout(x)