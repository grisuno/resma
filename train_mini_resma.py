import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from garnier_nn import SilencioActivoNetwork
import logging
import os

def main():
    # Configuración
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 3  # Muy reducido para prueba rápida
    SCALE = 500  # Reducción de escala
    
    print("🔥 INICIANDO ENTRENAMIENTO RESMA-GARNIER")
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
    train_data = datasets.MNIST('/workspace/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # Batch más pequeño
    
    # Crear red RESMA-Garnier mini
    print("🧠 Construyendo red RESMA-Garnier...")
    model = SilencioActivoNetwork(
        layer_sizes=[784, 64],  # Solo 1 capa Garnier + salida directa
        scale=SCALE,
        device=DEVICE
    ).to(DEVICE)
    
    # Entrenamiento
    print("\n🚀 Iniciando entrenamiento...")
    final_metrics = model.entrenar(train_loader, epochs=EPOCHS)
    
    # Guardar modelo
    checkpoint_path = '/workspace/mini-resma/mini_resma_final.pth'
    torch.save({
        'model_state': model.state_dict(),
        'topology': model.topology,
        'metrics': final_metrics
    }, checkpoint_path)
    
    print("\n" + "="*50)
    print("FIN DEL ENTRENAMIENTO RESMA-GARNIER")
    print("="*50)
    print(f"📊 Estado Final: {final_metrics['estado']}")
    print(f"🌟 Libertad L: {final_metrics['libertad']:.2f}")
    print(f"📈 Factor Bayes: BF = {final_metrics['BF']:+.2f}")
    print(f"🔌 Conectividad: ρ = {final_metrics['connectivity']:.2%}")
    
    if final_metrics['estado'] == "SOBERANO":
        print("\n🎉 ¡La red alcanzó soberanía conceptual!")
    elif final_metrics['estado'] == "EMERGENTE":
        print("\n⚠️ Estado emergente - ajustar φ₃ hacia π")
        # Sugerencia: model.layers[-1].phi[2] += 0.1  # Empujar fase
    
    print(f"\n💾 Modelo guardado en: {checkpoint_path}")
    
    return final_metrics, checkpoint_path

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    final_metrics, checkpoint_path = main()