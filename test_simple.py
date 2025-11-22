#!/usr/bin/env python3
"""
Test ultra-simple de la implementación RESMA-Garnier
"""
import torch
import numpy as np

def test_basic_math():
    """Test de las matemáticas básicas RESMA"""
    print("🧮 TEST MATEMÁTICAS RESMA-GARNIER")
    print("="*40)
    
    # Escalas temporales Garnier
    C0, C2, C3 = 1.0, 2.7, 7.3
    print(f"Escalas temporales:")
    print(f"  C₀ = {C0} ns⁻¹ (físico)")
    print(f"  C₂ = {C2} ns⁻¹ (crítico)")  
    print(f"  C₃ = {C3} ns⁻¹ (teleológico)")
    
    # Umbral de silencio-activo
    epsilon_c = np.log(2) * (C0 / C3) ** 2
    print(f"\nUmbral silencio-activo:")
    print(f"  ε_c = ln(2) × (C₀/C₃)² = {epsilon_c:.6f}")
    
    # Fases temporales aleatorias
    phi = np.random.rand(3) * 2 * np.pi
    print(f"\nFases temporales aleatorias:")
    print(f"  φ = [{phi[0]:.3f}, {phi[1]:.3f}, {phi[2]:.3f}]")
    
    # Acoplamiento temporal
    xi = abs(np.cos(phi[0]) * np.sin(phi[1]) * np.cos(phi[2]))
    print(f"Acoplamiento temporal:")
    print(f"  ξ(φ) = |cos(φ₀)sin(φ₂)cos(φ₃)| = {xi:.6f}")
    
    # Entropía simulada
    delta_s = np.random.uniform(0.001, 0.1)
    print(f"\nEntropía simulada:")
    print(f"  ΔS_loop = {delta_s:.6f}")
    
    # Libertad
    libertad = 1.0 / (delta_s + 1e-12)
    print(f"Libertad:")
    print(f"  L = 1/ΔS = {libertad:.2f}")
    
    # Factor Bayes
    BF = np.log(libertad + 1e-12)
    print(f"Factor Bayes:")
    print(f"  BF = ln(L) = {BF:+.2f}")
    
    # Estado
    if libertad > 100:
        estado = "SOBERANO 🎉"
    elif libertad > 10:
        estado = "EMERGENTE ⚠️"
    else:
        estado = "NO SOBERANO ❌"
    
    print(f"\nEstado final: {estado}")
    
    # Test de PyTorch
    print(f"\n🔧 Test PyTorch:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
    
    # Tensor simple
    x = torch.randn(5, 10, device=device)
    print(f"Tensor test: {x.shape}")
    print(f"Tensor mean: {x.mean().item():.3f}")
    print(f"Tensor std: {x.std().item():.3f}")
    
    print("\n" + "="*40)
    print("✅ TEST COMPLETADO")
    print("="*40)

if __name__ == '__main__':
    test_basic_math()