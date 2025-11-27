import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from resma_core import RESMABrain

# Configuración
NOISE_LEVEL = 2.5

def add_noise(tensor, factor):
    return tensor + torch.randn_like(tensor) * factor

def visualize_resma_perception():
    print("👁️ Cargando sistema de visión RESMA...")
    
    # 1. Preparar una imagen
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Tomamos un dígito aleatorio
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]
    
    # 2. Inyectar Ruido (El Ataque)
    flat_image = image.view(1, 784)
    noisy_image = add_noise(flat_image, NOISE_LEVEL)
    
    # 3. Inicializar un cerebro fresco 
    model = RESMABrain(784, 128, 10)
    model.eval() # Modo evaluación
    
    # 4. Proceso Físico
    with torch.no_grad():
        # Capa 1: Lattice + PT Activation
        linear_out = model.layer1(noisy_image)
        # Aquí ocurre la magia
        filtered_signal, gate, _, zeeman = model.act1(linear_out)
        
        # Salida final
        prediction = model(noisy_image)
        pred_label = prediction.argmax(dim=1).item()
        
        # Reconstrucción (Proyección inversa para visualizar qué ve la red)
        # Usamos .t() (transpuesta) de los pesos para proyectar de 128 -> 784
        reconstructed = torch.nn.functional.linear(filtered_signal, model.layer1.weights.t())

    # 5. Visualización (Añadido .detach() para corregir el error)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # A. Entrada Ruidosa
    img_noisy = noisy_image.view(28, 28).detach().numpy()
    axes[0].imshow(img_noisy, cmap='gray')
    axes[0].set_title(f"Entrada (Ruido {NOISE_LEVEL}σ)\nEtiqueta Real: {label}")
    axes[0].axis('off')
    
    # B. Lo que ve la Red (Estado de la Gate PT)
    gate_visual = gate.view(1, 128).detach().numpy()
    axes[1].imshow(gate_visual, cmap='plasma', aspect='auto')
    axes[1].set_title(f"Filtro PT-Simétrico (Capa Oculta)\nAmarillo=Pasa, Azul=Bloqueado")
    axes[1].set_xlabel("Neuronas Lattice E8 (1-128)")
    axes[1].set_yticks([])
    
    # C. Reconstrucción Aproximada (Señal Filtrada)
    img_recon = reconstructed.view(28, 28).detach().numpy()
    
    axes[2].imshow(img_recon, cmap='gray')
    axes[2].set_title(f"Señal Filtrada (Proyección)\nPredicción RESMA: {pred_label}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("resma_vision_result.png")
    print(f"📸 Visualización guardada en 'resma_vision_result.png'")
    print(f"   Revisa la imagen para ver cómo la física filtra el ruido.")

if __name__ == "__main__":
    visualize_resma_perception()