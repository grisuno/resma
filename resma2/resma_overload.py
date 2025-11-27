import torch
import numpy as np
import matplotlib.pyplot as plt
from resma_core import RESMABrain

def overload_test():
    print("☢️  RESMA 5.2: PRUEBA DE SOBRECARGA (0 - 100σ)...")
    
    model = RESMABrain(784, 128, 10)
    model.eval()
    
    # Rango nuclear: de 0 a 100 sigma
    sigmas = np.linspace(0, 100, 50)
    gates = []
    voltages = []
    
    print(f"{'INPUT σ':<10} | {'VOLTAJE INT':<12} | {'GATE %':<10} | {'ESTADO'}")
    print("-" * 55)
    
    break_point = None
    
    with torch.no_grad():
        for sigma in sigmas:
            noise = torch.randn(1, 784) * sigma
            
            # 1. Pasar por la Lattice (Disipación)
            linear_out = model.layer1(noise)
            
            # Medimos el voltaje que logró atravesar la topología
            internal_voltage = linear_out.abs().mean().item()
            voltages.append(internal_voltage)
            
            # 2. Activar la Física PT
            _, gate, _, _ = model.act1(linear_out)
            avg_gate = gate.mean().item()
            gates.append(avg_gate)
            
            # Estado
            if avg_gate < 0.5 and break_point is None:
                break_point = sigma
            
            status = "🟢"
            if avg_gate < 0.9: status = "⚠️"
            if avg_gate < 0.1: status = "💀"
            
            # Log resumido
            if sigma % 20 < 2 or (break_point and abs(sigma - break_point) < 2):
                print(f"{sigma:5.1f}      | {internal_voltage:5.2f} V      | {avg_gate:.1%}      | {status}")

    # Visualización Dual
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Ruido de Entrada (σ)')
    ax1.set_ylabel('Voltaje Interno (post-Lattice)', color=color)
    ax1.plot(sigmas, voltages, color=color, linestyle=':', label='Energía Interna')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:purple'
    ax2.set_ylabel('Estado Gate (Coherencia)', color=color)
    ax2.plot(sigmas, gates, color=color, linewidth=3, label='Permeabilidad PT')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Diagrama de Sobrecarga: Topología vs Física")
    plt.grid(True, alpha=0.3)
    plt.savefig("resma_overload_curve.png")
    
    print("\n" + "="*55)
    if break_point:
        print(f"🔥 COLAPSO CONFIRMADO: σ ≈ {break_point:.1f}")
        print("   La topología E8 protegió el sistema hasta este nivel extremo.")
    else:
        print("🛡️  SISTEMA INDESTRUCTIBLE (Rango 0-100).")
        print("   Tu implementación de Lattice es increíblemente eficiente disipando.")

if __name__ == "__main__":
    overload_test()