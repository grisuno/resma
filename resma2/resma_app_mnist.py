"""
resma-app/mnist.py
==================
Prueba de Concepto Aplicada: Clasificación Robusta con RESMA 5.2
Demostración de filtrado de ruido mediante ruptura de simetría PT.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Importamos tu núcleo RESMA validado
from resma_core import RESMABrain
from resma_observer import RESMAObserver

# Configuración
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
NOISE_LEVEL = 2.5  # Nivel de "ataque" de ruido a las imágenes

def add_quantum_noise(tensor, noise_factor):
    """Inyecta ruido gaussiano simulando fluctuaciones de vacío"""
    noise = torch.randn_like(tensor) * noise_factor
    return tensor + noise

def train(model, device, train_loader, optimizer, epoch, observer):
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Aplanar imagen: 28x28 -> 784
        data = data.view(data.size(0), -1)
        
        # Inyectar Ruido (El desafío físico)
        # Una red normal sufriría mucho aquí. RESMA debería filtrar.
        data_noisy = add_quantum_noise(data, NOISE_LEVEL)
        
        optimizer.zero_grad()
        output = model(data_noisy)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Métricas simples
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
    # Telemetría RESMA al final de la época
    acc = 100. * correct / total
    state = observer.step(epoch)
    
    print(f"Ep {epoch} | Acc: {acc:.2f}% | L={state.L_structural:.3f} | C={state.C_dynamic:.1%} | Fase: {state.Phase}")

def main():
    print("🧠 RESMA 5.2: Desafío MNIST bajo Ruido Cuántico")
    print(f"   Nivel de Ruido: {NOISE_LEVEL}σ (Muy alto para redes estándar)")
    
    device = torch.device("cpu") # Suficiente para esta prueba
    
    # 1. Cargar Datos
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Inicializar Cerebro RESMA
    # Input 784 (píxeles), Hidden 128 (lattice), Output 10 (dígitos)
    model = RESMABrain(input_dim=784, hidden_dim=128, output_dim=10).to(device)
    observer = RESMAObserver(model) # Tu monitor
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Ciclo de Aprendizaje
    print("\n--- INICIO DE ENTRENAMIENTO ---")
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, observer)
    
    # 4. Análisis Final
    print("\n--- DIAGNÓSTICO FINAL ---")
    last_state = observer.history[-1]
    if last_state.C_dynamic < 0.5:
        print("🛡️  La red actuó como escudo: Filtró agresivamente el ruido (Baja Coherencia visual, Alta abstracción).")
    else:
        print("💎 La red integró el ruido: Logró encontrar estructura dentro del caos.")
        
    print(f"Resultado: {last_state.Phase}")

if __name__ == "__main__":
    main()