import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Tuple, Dict
import logging

class GarnierLayer(nn.Module):
    """Capa neuronal con temporalidad Garnier T³ (simplificada para demo)"""
    def __init__(self, in_features: int, out_features: int, device: str = 'cpu'):
        super().__init__()
        
        # Pesos lineales (emulación de operador D̂_G)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)
        
        # Fases temporales φ = [φ₀, φ₂, φ₃] (aprendibles)
        self.phi = nn.Parameter(torch.rand(3, device=device) * 2 * np.pi)
        
        # Escalas temporales Garnier (fijas por diseño)
        self.C0, self.C2, self.C3 = 1.0, 2.7, 7.3
        
        # Umbral de silencio-activo (ε_c)
        self.epsilon_c = np.log(2) * (self.C0 / self.C3) ** 2
        
        self.device = device
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Forward simplificado para demostración"""
        batch_size = x.size(0)
        
        # Propagación lineal + no-linealidad Garnier
        output = torch.relu(x @ self.weight.T)
        
        # Calcular acoplamiento temporal ξ(φ)
        xi = torch.abs(torch.cos(self.phi[0]) * torch.sin(self.phi[1]) * torch.cos(self.phi[2])).item()
        epsilon_c = self.epsilon_c * (1 + xi)
        
        # Calcular coherencia de estado (entropía simplificada)
        # Usar una medida más simple de coherencia
        output_mean = torch.mean(output, dim=-1)
        delta_s = torch.var(output_mean).item()  # Variancia como proxy de entropía
        
        # Gate silencio-activo simplificado
        gate_factor = max(0.1, 1.0 - delta_s / (epsilon_c + 1e-6))
        output = output * gate_factor
        
        return output, delta_s

def demo_resma():
    """Demostración rápida de la arquitectura RESMA-Garnier"""
    print("🔥 DEMOSTRACIÓN RESMA-GARNIER")
    print("="*50)
    
    DEVICE = 'cpu'
    SCALE = 100  # Reducción extrema para demo
    
    # Dataset pequeño para demo
    data = torch.randn(10, 784)  # 10 samples de 784 features (MNIST flatten)
    targets = torch.randint(0, 10, (10,))
    
    print(f"📊 Datos de demo: {data.shape}")
    print(f"🎯 Targets: {targets}")
    
    # Construir topología RESMA (BA+WS)
    print("🌐 Construyendo topología RESMA...")
    G_ba = nx.barabasi_albert_graph(SCALE, m=2)
    G_ws = nx.watts_strogatz_graph(SCALE, k=3, p=0.1)
    G = nx.compose(G_ba, G_ws)
    connectivity = nx.density(G)
    
    print(f"🔌 Conectividad de red: ρ = {connectivity:.2%}")
    print(f"📏 Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
    
    # Crear una capa Garnier
    print("\n🧠 Creando capa Garnier...")
    garnier_layer = GarnierLayer(784, 64, DEVICE)
    
    # Forward pass
    print("🚀 Ejecutando forward pass...")
    output, delta_s = garnier_layer(data)
    
    print(f"📐 Forma de salida: {output.shape}")
    print(f"⚖️ Delta S (entropía): {delta_s:.6f}")
    
    # Calcular métricas RESMA
    libertad = 1.0 / (delta_s + 1e-12)  # L = 1/ε
    BF = np.log(libertad + 1e-12)
    
    print(f"🌟 Libertad L: {libertad:.2f}")
    print(f"📈 Factor Bayes BF: {BF:+.2f}")
    
    # Determinar estado
    if libertad > 100:
        estado = "SOBERANO"
        emoji = "🎉"
    elif libertad > 10:
        estado = "EMERGENTE" 
        emoji = "⚠️"
    else:
        estado = "NO SOBERANO"
        emoji = "❌"
    
    print(f"{emoji} Estado: {estado}")
    
    # Mostrar fases temporales
    print(f"\n🕰️ Fases temporales (φ):")
    print(f"  φ₀ (físico): {garnier_layer.phi[0].item():.3f}")
    print(f"  φ₂ (crítico): {garnier_layer.phi[1].item():.3f}")  
    print(f"  φ₃ (teleológico): {garnier_layer.phi[2].item():.3f}")
    
    # Calcular acoplamiento temporal
    xi = torch.abs(torch.cos(garnier_layer.phi[0]) * torch.sin(garnier_layer.phi[1]) * torch.cos(garnier_layer.phi[2])).item()
    print(f"🔗 Acoplamiento temporal ξ: {xi:.3f}")
    
    print("\n" + "="*50)
    print("✅ DEMO COMPLETADA")
    print("="*50)
    
    # Resumen matemático
    print("\n📊 RESUMEN MATEMÁTICO:")
    print(f"  • Operador D̂_G: Aproximado por pesos lineales")
    print(f"  • Tiempo T³: φ = [{garnier_layer.phi[0].item():.2f}, {garnier_layer.phi[1].item():.2f}, {garnier_layer.phi[2].item():.2f}]")
    print(f"  • Escalas: C₀={garnier_layer.C0}, C₂={garnier_layer.C2}, C₃={garnier_layer.C3}")
    print(f"  • Gate silencio-activo: ε_c = {garnier_layer.epsilon_c:.6f}")
    print(f"  • Medida de libertad: L = 1/ΔS = {libertad:.2f}")
    print(f"  • Factor Bayes: BF = ln(L) = {BF:+.2f}")
    
    return {
        'connectivity': connectivity,
        'delta_s': delta_s,
        'libertad': libertad,
        'BF': BF,
        'estado': estado,
        'topology': G
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    results = demo_resma()