"""
resma-observer/observer.py v1.0.0
=================================
Sistema de Telemetría Unificado (Estructura + Dinámica)
Integra 'liber-monitor' con ganchos físicos de RESMA.

Métricas:
- L (Libertad Estructural): Vía SVD de pesos (liber-monitor)
- C (Coherencia Dinámica): Vía estado de gate PT
- Ξ (Criticalidad): Producto L * C
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json

# Importación de tu módulo existente (asumiendo que está en el path)
try:
    from monitor import SovereigntyMonitor
except ImportError:
    raise ImportError("Falta 'monitor.py'. Asegúrate de tener tu módulo liber-monitor disponible.")

@dataclass
class QuantumState:
    """Snapshot del estado físico-estructural de la red"""
    epoch: int
    L_structural: float      # Libertad (Estructura)
    C_dynamic: float         # Coherencia (Dinámica)
    Zeeman_energy: float     # Caos promedio
    Criticality: float       # Índice de Criticalidad (L * C)
    Phase: str               # Clasificación de fase
    
    def to_dict(self):
        return asdict(self)

class RESMAObserver:
    def __init__(self, model: torch.nn.Module, epsilon_c: float = 0.1):
        self.model = model
        
        # 1. Componente Estructural (Tu código original)
        self.structural_monitor = SovereigntyMonitor(epsilon_c=epsilon_c, verbose=False)
        
        # 2. Componente Dinámico (Hooks de física)
        self.history: List[QuantumState] = []
        self._dynamic_buffer = {"gate": [], "zeeman": []}
        self._register_hooks()
        
    def _register_hooks(self):
        """Inyecta sondas en las capas PT para leer telemetría en tiempo real"""
        def hook_fn(module, input, output):
            # output signature: (x, gate, coherence, zeeman)
            if isinstance(output, tuple) and len(output) == 4:
                _, gate, _, zeeman = output
                self._dynamic_buffer["gate"].append(gate.detach().cpu().mean().item())
                self._dynamic_buffer["zeeman"].append(zeeman.detach().cpu().mean().item())
        
        count = 0
        for name, layer in self.model.named_modules():
            # Detectar clases PT por nombre o tipo
            if "PTSymmetricActivation" in str(type(layer)):
                layer.register_forward_hook(hook_fn)
                count += 1
        print(f"👁️ RESMA Observer: Acoplado a {count} capas cuánticas.")

    def step(self, epoch: int) -> QuantumState:
        """
        Ejecutar al final de cada época de entrenamiento/validación.
        Fusiona métricas y determina la fase.
        """
        # A. Análisis Estructural (SVD)
        L = self.structural_monitor.calculate(self.model)
        
        # B. Análisis Dinámico (Promedio de buffers)
        if self._dynamic_buffer["gate"]:
            avg_gate = np.mean(self._dynamic_buffer["gate"])
            avg_zeeman = np.mean(self._dynamic_buffer["zeeman"])
        else:
            avg_gate, avg_zeeman = 1.0, 0.0 # Valores default
            
        # Limpiar buffers
        self._dynamic_buffer = {"gate": [], "zeeman": []}
        
        # C. Determinación de Fase (Lógica RESMA 5.1)
        phase = "INDEFINIDO"
        xi = L * avg_gate # Índice de Criticalidad
        
        if L < 0.5:
            phase = "💀 COLAPSO ESTRUCTURAL (Espurio)"
        elif avg_gate < 0.05:
            phase = "🔇 SILENCIO ACTIVO (Protección)"
        elif L > 1.0 and avg_gate > 0.9:
            phase = "👑 SOBERANO (E8 Critical)"
        elif avg_gate < 0.9:
            phase = "⚠️ ESTRÉS DINÁMICO"
        else:
            phase = "🔄 EMERGENTE"
            
        state = QuantumState(epoch, L, avg_gate, avg_zeeman, xi, phase)
        self.history.append(state)
        
        return state

    def report(self, state: QuantumState):
        """Imprime reporte formateado a consola"""
        # Emojis de estado
        icon = "❓"
        if "SOBERANO" in state.Phase: icon = "👑"
        elif "SILENCIO" in state.Phase: icon = "🔇"
        elif "COLAPSO" in state.Phase: icon = "💀"
        elif "ESTRÉS" in state.Phase: icon = "⚡"
        
        print(f"Ep {state.epoch:03d} | {icon} {state.Phase:<25} | "
              f"L={state.L_structural:.3f} | C={state.C_dynamic:.1%} | Ξ={state.Criticality:.3f}")

    def plot_phase_space(self, save_path="resma_phase_diagram.png"):
        """Genera el diagrama de fase: Estructura vs Dinámica"""
        if not self.history: return
        
        epochs = [s.epoch for s in self.history]
        Ls = [s.L_structural for s in self.history]
        Cs = [s.C_dynamic for s in self.history]
        
        plt.figure(figsize=(10, 7))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Scatter plot con color por época
        sc = plt.scatter(Ls, Cs, c=epochs, cmap='plasma', s=100, edgecolors='black', alpha=0.8)
        plt.colorbar(sc, label='Época')
        
        # Líneas de umbral
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Límite Estructural (L=0.5)')
        plt.axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='Límite PT (C=0.1)')
        
        # Zona Soberana
        plt.fill_between([1.0, max(Ls+[1.5])*1.1], 0.9, 1.0, color='green', alpha=0.15, label='Zona Soberana')
        
        plt.title("Espacio de Fase RESMA: Estructura vs Dinámica", fontsize=14, fontweight='bold')
        plt.xlabel("Libertad Estructural (L)", fontsize=12)
        plt.ylabel("Coherencia Dinámica (Gate %)", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        
        plt.savefig(save_path, dpi=300)
        print(f"📊 Diagrama de Fase guardado en: {save_path}")