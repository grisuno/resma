import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resma_core import RESMABrain

MODEL_FILE = "resma_trained.pt"
BATCH_SIZE = 1000  # Test rápido con un lote grande

def combat_test():
    print(f"🛡️  RESMA 5.2: TEST DE COMBATE (Precisión vs Ruido)...")
    
    # Cargar datos de test
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Cargar modelo
    model = RESMABrain(784, 128, 10)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    
    # Niveles de ruido a probar
    noise_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    accuracies = []
    gate_health = []
    
    # Tomamos un solo lote grande para probar
    data_orig, target = next(iter(test_loader))
    data_orig = data_orig.view(-1, 784)
    
    print(f"\n{'RUIDO (σ)':<10} | {'PRECISIÓN':<10} | {'GATE (Salud)':<12} | {'ESTADO'}")
    print("-" * 50)
    
    with torch.no_grad():
        for sigma in noise_levels:
            # Atacar
            noise = torch.randn_like(data_orig) * sigma
            data_noisy = data_orig + noise
            
            # Defender y Predecir
            # Pasamos manualmente para obtener métricas internas
            linear_out = model.layer1(data_noisy)
            _, gate, _, _ = model.act1(linear_out)
            
            out = model(data_noisy) # Forward completo
            pred = out.argmax(dim=1)
            
            # Métricas
            acc = (pred == target).float().mean().item() * 100
            avg_gate = gate.mean().item() * 100
            
            accuracies.append(acc)
            gate_health.append(avg_gate)
            
            status = "✅ Operativo"
            if acc < 80: status = "⚠️ Degradado"
            if acc < 20: status = "💀 Inutilizado"
            
            print(f"{sigma:<10.1f} | {acc:6.2f}%    | {avg_gate:6.2f}%      | {status}")

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies, marker='o', linewidth=3, color='blue', label='Precisión')
    plt.plot(noise_levels, gate_health, marker='x', linestyle='--', color='green', label='Salud Mielina (Gate)')
    
    plt.title(f"Rendimiento en Combate (Modelo Entrenado)")
    plt.xlabel("Nivel de Ruido (σ)")
    plt.ylabel("Porcentaje %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("resma_combat_report.png")
    print(f"\n📉 Reporte guardado en 'resma_combat_report.png'")

if __name__ == "__main__":
    combat_test()