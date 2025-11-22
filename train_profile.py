import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from garnier_nn import SilencioActivoNetwork
import logging
import time
import os

def main():
    # Configuración
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 1  # Solo 1 época para perfilado
    SCALE = 500  # Reducción de escala
    
    print("🔥 INICIANDO PERFILADO RESMA-GARNIER")
    print("="*50)
    print(f"Dispositivo: {DEVICE}")
    print(f"Épocas: {EPOCHS}")
    print(f"Escala de red: {SCALE}")
    
    # Dataset MNIST simple
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Binarizar
    ])
    
    print("📊 Cargando dataset MNIST...")
    start_time = time.time()
    train_data = datasets.MNIST('/workspace/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Batch más pequeño
    load_time = time.time() - start_time
    print(f"⏱️  Tiempo carga dataset: {load_time:.3f}s")
    
    # Crear red RESMA-Garnier mini CON PERFILADO
    print("\n🧠 Construyendo red RESMA-Garnier...")
    start_time = time.time()
    model = SilencioActivoNetwork(
        layer_sizes=[784, 64],  # Solo 1 capa Garnier + salida directa
        scale=SCALE,
        device=DEVICE
    ).to(DEVICE)
    build_time = time.time() - start_time
    print(f"⏱️  Tiempo construcción red: {build_time:.3f}s")
    
    # PERFILADO DETALLADO DE FORWARD PASS
    print("\n🔍 PERFILADO DETALLADO DE FORWARD PASS:")
    
    # Tomar una muestra pequeña para perfilado
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.view(batch_x.size(0), -1).to(DEVICE)
    
    print(f"Procesando batch de {batch_x.size(0)} muestras...")
    
    # Activar perfilado en el modelo
    model.activar_perfilado()
    
    # Un solo forward pass para medir tiempo
    start_time = time.time()
    output = model(batch_x)
    forward_time = time.time() - start_time
    
    print(f"\n⏱️  Tiempo total forward pass: {forward_time:.3f}s")
    
    # Mostrar estadísticas de perfilado
    model.mostrar_estadisticas_perfilado()
    
    # Entrenamiento rápido con perfilado
    print("\n🚀 Iniciando entrenamiento con perfilado...")
    start_time = time.time()
    final_metrics = model.entrenar_con_perfilado(train_loader, epochs=EPOCHS)
    train_time = time.time() - start_time
    
    print(f"\n⏱️  Tiempo total entrenamiento: {train_time:.3f}s")
    
    print("\n" + "="*50)
    print("PERFILADO COMPLETADO")
    print("="*50)
    
    return final_metrics, build_time, forward_time, train_time

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    final_metrics, build_time, forward_time, train_time = main()