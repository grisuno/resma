import torch
import numpy as np
import matplotlib.pyplot as plt
from resma_core import RESMABrain

def find_break_point():
    print("⚡ RESMA 5.2: Buscando el Punto de Ruptura (Criticality Scanner)...")
    
    # Inicializamos modelo fresco
    model = RESMABrain(784, 128, 10)
    model.eval()
    
    # Rango extendido: Vamos de 0 a 25 sigma
    sigmas = np.linspace(0, 25, 50)
    gates = []
    
    print(f"{'RUIDO (σ)':<10} | {'GATE %':<10} | {'ESTADO'}")
    print("-" * 40)
    
    break_point = None
    
    with torch.no_grad():
        for sigma in sigmas:
            # Generar imagen de puro ruido con esa desviación estándar
            noise = torch.randn(1, 784) * sigma
            
            # Pasar por la capa física
            linear_out = model.layer1(noise)
            _, gate, _, _ = model.act1(linear_out)
            
            # Promedio de apertura de compuertas
            avg_gate = gate.mean().item()
            gates.append(avg_gate)
            
            # Determinar estado
            status = "🟢 Estable"
            if avg_gate < 0.90: status = "⚠️ Estrés"
            if avg_gate < 0.50: status = "🔴 Colapso PT"
            if avg_gate < 0.05: status = "💀 Silencio"
            
            # Detectar el cruce exacto de 0.5 (Transición de Fase)
            if break_point is None and avg_gate < 0.5:
                break_point = sigma
            
            # Imprimir solo hitos para no saturar consola
            if int(sigma) == sigma or sigma in [sigmas[0], sigmas[-1]]: 
                 print(f"{sigma:5.1f}      | {avg_gate:.1%}      | {status}")
            elif avg_gate < 0.9 and avg_gate > 0.1: # Imprimir detalle durante la caída
                 print(f"{sigma:5.1f}      | {avg_gate:.1%}      | {status}")

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, gates, color='purple', linewidth=3, label='Permeabilidad Mielina (Gate)')
    plt.axhline(0.935, color='green', linestyle='--', label='Reposo (Soberano)')
    plt.axhline(0.0, color='black', linestyle='-', alpha=0.3)
    
    if break_point:
        plt.axvline(break_point, color='red', linestyle='--', label=f'Tc (Crítico) = {break_point:.2f}')
        plt.text(break_point + 0.5, 0.5, f'Ruputura PT\nσ ≈ {break_point:.1f}', color='red')
        
    plt.title("Diagrama de Fase: Robustez vs Ruido")
    plt.xlabel("Nivel de Ruido de Entrada (σ)")
    plt.ylabel("Estado de Compuerta (0=Cerrado, 1=Abierto)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("resma_break_curve.png")
    
    print("\n" + "="*40)
    if break_point:
        print(f"🔥 PUNTO DE RUPTURA ENCONTRADO: σ ≈ {break_point:.2f}")
        print("   (Hasta aquí aguanta tu red antes de activar el escudo)")
    else:
        print("🛡️  El sistema es indestructible en este rango.")
    print("📈 Gráfico guardado como 'resma_break_curve.png'")

if __name__ == "__main__":
    find_break_point()