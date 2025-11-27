"""
resma-exp/main.py
=================
Script de ejecución para validación de RESMA 5.1
Simula un entorno de entrenamiento con fluctuaciones de energía
para probar la respuesta del Observador y el Cerebro.
"""

import torch
import torch.nn as nn
import numpy as np
import random

# Importaciones locales
from resma_core import RESMABrain
from resma_observer import RESMAObserver

# Configuración
SEED = 42
EPOCHS = 25
BATCH_SIZE = 32
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 10

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_experiment():
    print("🚀 INICIANDO PROTOCOLO EXPERIMENTAL RESMA 5.1")
    print("============================================")
    set_seed(SEED)
    
    # 1. Inicialización
    model = RESMABrain(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    observer = RESMAObserver(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # Datos sintéticos
    base_inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
    targets = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))
    
    print(f"🧠 Modelo inicializado. Parámetros: {sum(p.numel() for p in model.parameters())}")
    print("📉 Iniciando ciclo de entrenamiento con inyección de energía variable...\n")
    
    # 2. Ciclo de Entrenamiento
    for epoch in range(1, EPOCHS + 1):
        # A. Simulación de Ambiente (Energía Variable)
        # Ep 1-8: Normal | Ep 9-15: Tormenta (Alta Energía) | Ep 16-25: Recuperación
        energy_amp = 1.0
        if 8 < epoch <= 15:
            energy_amp = 12.0 + (epoch - 8) * 1.5 # Rampa de energía hasta colapso potencial
        
        # B. Forward Pass
        optimizer.zero_grad()
        
        # Inyectar energía en inputs
        inputs = base_inputs * energy_amp
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # C. Backward & Step
        loss.backward()
        # Gradient Clipping (Esencial en física no lineal)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # D. Observación
        state = observer.step(epoch)
        
        # E. Reporte en vivo
        print(f"[Ambiente E={energy_amp:4.1f}] ", end="")
        observer.report(state)
        
        # F. Lógica de Intervención (Meta-Learning simulado)
        if "SILENCIO" in state.Phase:
            print("      >>> 🛡️  SISTEMA PROTEGIDO: Señal disipada por Mielina PT.")
    
    # 3. Finalización
    print("\n============================================")
    print("✅ Experimento finalizado.")
    observer.plot_phase_space()

if __name__ == "__main__":
    run_experiment()