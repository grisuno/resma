"""
monitor.py (Local Version for RESMA)
Basado en liber-monitor v2.0.0
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dataclasses import dataclass, asdict
from enum import Enum

class Regime(Enum):
    SOBERANO = "soberano"
    EMERGENTE = "emergente"
    ESPURIO = "espurio"

@dataclass
class LayerDiagnostics:
    layer_name: str
    L: float
    entropy_vn: float
    rank_effective: int
    regime: str
    weight_shape: tuple

@dataclass
class EpochSnapshot:
    epoch: int
    L_promedio: float
    regime_promedio: str
    layers: List[LayerDiagnostics]
    entropia_promedio: float
    rango_promedio: float

class SovereigntyMonitor:
    def __init__(self, epsilon_c: float = 0.1, patience: int = 2, 
                 umbral_soberano: float = 1.0, umbral_espurio: float = 0.5,
                 track_layers: bool = True, verbose: bool = False):
        self.epsilon_c = epsilon_c
        self.patience = patience
        self.umbral_soberano = umbral_soberano
        self.umbral_espurio = umbral_espurio
        self.track_layers = track_layers
        self.verbose = verbose
        self.history: List[EpochSnapshot] = []
        self.layer_history: Dict[str, List[float]] = {}
        self._critical_epochs = 0
        self._warning_epochs = 0
    
    def _extract_weights(self, model: torch.nn.Module):
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    weights.append((name, module.weight, tuple(module.weight.shape)))
        return weights
    
    def _calculate_svd_metrics(self, weight_matrix: np.ndarray):
        try:
            if weight_matrix.ndim > 2:
                weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)
            
            # SVD Robusto
            try:
                U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
            except:
                S = np.abs(weight_matrix).flatten()
                S = np.sort(S)[::-1]
            
            if len(S) == 0: return 0.0, 1
            
            threshold = 0.01 * np.max(S)
            rank_effective = max(1, int(np.sum(S > threshold)))
            
            S_sum = np.sum(S) + 1e-10
            S_normalized = S / S_sum
            S_normalized = S_normalized[S_normalized > 1e-15]
            S_vn = -np.sum(S_normalized * np.log(S_normalized))
            
            return float(S_vn), int(rank_effective)
        except:
            return 0.0, 1
    
    def calcular_libertad(self, weights: torch.Tensor):
        W = weights.detach().cpu().numpy()
        S_vn, rank_effective = self._calculate_svd_metrics(W)
        log_rank = np.log(rank_effective + 1)
        denominador = np.abs(S_vn - log_rank) + self.epsilon_c
        L = 1.0 / denominador
        return L, S_vn, rank_effective

    def calculate(self, model: torch.nn.Module) -> float:
        weights = self._extract_weights(model)
        if not weights: return 1.0
        
        layer_diagnostics = []
        for name, w, shape in weights:
            L, S_vn, rank = self.calcular_libertad(w)
            regime = Regime.ESPURIO.value
            if L > self.umbral_soberano: regime = Regime.SOBERANO.value
            elif L > self.umbral_espurio: regime = Regime.EMERGENTE.value
            
            layer_diagnostics.append(LayerDiagnostics(name, L, S_vn, rank, regime, shape))
            
        L_promedio = np.mean([d.L for d in layer_diagnostics])
        S_promedio = np.mean([d.entropy_vn for d in layer_diagnostics])
        R_promedio = np.mean([d.rank_effective for d in layer_diagnostics])
        
        regime_avg = Regime.ESPURIO.value
        if L_promedio > self.umbral_soberano: regime_avg = Regime.SOBERANO.value
        elif L_promedio > self.umbral_espurio: regime_avg = Regime.EMERGENTE.value
        
        self.history.append(EpochSnapshot(
            len(self.history), L_promedio, regime_avg, layer_diagnostics, S_promedio, R_promedio
        ))
        
        return float(L_promedio)