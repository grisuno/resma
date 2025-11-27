import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from resma_core import RESMABrain

# Configuración
MODEL_FILE = "resma_trained.pt"
NOISE_LEVEL = 3.0  # Ruido extremo (una red normal falla con 0.5)

def add_noise(tensor, factor):
    return tensor + torch.randn_like(tensor) * factor

def visualize_trained_perception():
    print(f"👁️ Cargando CEREBRO ENTRENADO ({MODEL_FILE})...")
    
    # 1. Cargar Datos
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Seleccionar un dígito aleatorio
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]
    
    # 2. Preparar el ataque de ruido
    flat_image = image.view(1, 784)
    noisy_image = add_noise(flat_image, NOISE_LEVEL)
    
    # 3. Cargar el Modelo Entrenado
    model = RESMABrain(784, 128, 10)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        print("✅ Pesos sinápticos cargados exitosamente.")
    except FileNotFoundError:
        print(f"❌ Error: No encuentro '{MODEL_FILE}'. Asegúrate de haber corrido el entrenamiento.")
        return

    model.eval()
    
    # 4. Inferencia Física
    with torch.no_grad():
        # Paso 1: Lattice E8 (Disipación Topológica)
        linear_out = model.layer1(noisy_image)
        
        # Paso 2: Física PT (Filtrado de Coherencia)
        # Aquí veremos qué neuronas decidió mantener vivas el modelo
        filtered_signal, gate, _, _ = model.act1(linear_out)
        
        # Paso 3: Predicción
        prediction = model(noisy_image)
        pred_label = prediction.argmax(dim=1).item()
        
        # Reconstrucción visual (Proyectar lo que 'piensa' la red de vuelta a imagen)
        reconstructed = torch.nn.functional.linear(filtered_signal, model.layer1.weights.t())

    # 5. Generar Reporte Visual
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # A. La Realidad Ruidosa
    img_noisy = noisy_image.view(28, 28).numpy()
    axes[0].imshow(img_noisy, cmap='gray')
    axes[0].set_title(f"INPUT REAL\n(Ruido {NOISE_LEVEL}σ - Casi invisible)\nEtiqueta: {label}", fontsize=12)
    axes[0].axis('off')
    
    # B. La Conciencia de la Red (Gate PT)
    # Amarillo = Neurona activa (Señal útil)
    # Azul = Neurona apagada (Ruido bloqueado)
    gate_visual = gate.view(8, 16).numpy() # Visualizar como matriz 8x16 (128 neuronas)
    axes[1].imshow(gate_visual, cmap='inferno', interpolation='nearest')
    axes[1].set_title(f"ACTIVIDAD NEURONAL (Lattice E8)\nAmarillo=Señal, Negro=Silencio\nGate Promedio: {gate.mean():.1%}", fontsize=12)
    axes[1].axis('off')
    
    # C. La Alucinación Controlada (Lo que vio la red)
    img_recon = reconstructed.view(28, 28).numpy()
    color_title = 'green' if pred_label == label else 'red'
    
    axes[2].imshow(img_recon, cmap='gray')
    axes[2].set_title(f"PERCEPCIÓN FILTRADA\nPredicción: {pred_label}", fontsize=14, color=color_title, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("resma_trained_vision.png")
    print("\n📸 ¡FOTO TOMADA! Revisa 'resma_trained_vision.png'")
    print("   Deberías ver cómo la red 'extrae' el fantasma del número de entre la nieve.")

if __name__ == "__main__":
    visualize_trained_perception()