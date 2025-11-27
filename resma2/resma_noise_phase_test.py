import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from resma_core import RESMABrain   # ← Tu cerebro RESMA

#########################################################
# CONFIG
#########################################################
HIDDEN = 128
NOISE_RANGE = np.linspace(0.0, 3.0, 13)   # 13 pasos: 0.0 → 3.0
SAMPLES = 300                             # imágenes a testear
DEVICE = "cpu"

#########################################################
# Preparación de dataset
#########################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

#########################################################
# Inicializar cerebro RESMA
#########################################################
model = RESMABrain(784, HIDDEN, 10).to(DEVICE)
model.eval()

#########################################################
# Funciones clave
#########################################################
def add_noise(x, sigma):
    return x + torch.randn_like(x) * sigma

def measure_entropy(gate_tensor):
    # p = prob. activación
    p = torch.clamp(gate_tensor, 1e-8, 1-1e-8)
    H = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
    return H.item()

#########################################################
# Ejecución del experimento
#########################################################
results_noise = []
results_gate = []
results_entropy = []
results_acc = []

print("\n🔬 Iniciando test de robustez RESMA...\n")

for sigma in NOISE_RANGE:
    correct = 0
    gate_on_count = []

    for i in range(SAMPLES):
        img, label = testset[np.random.randint(0,len(testset))]
        img = img.view(1,784).to(DEVICE)

        noisy = add_noise(img, sigma)

        with torch.no_grad():
            lin = model.layer1(noisy)
            filtered, gate, _, _ = model.act1(lin)  # gating RESMA
            
            # Medir activación
            gate_on_count.append(gate.mean().item())  # % neuronas activas

            pred = model(noisy).argmax(dim=1).item()
            if pred == label: correct += 1

    # Guardar estadísticas
    acc = correct / SAMPLES * 100
    gavg = np.mean(gate_on_count)
    H = measure_entropy(torch.tensor(gate_on_count))

    results_noise.append(sigma)
    results_gate.append(gavg)
    results_entropy.append(H)
    results_acc.append(acc)

    print(f"σ={sigma:.2f} | Gate={gavg:.3f} | Entropía={H:.3f} | Acc={acc:.1f}%")

#########################################################
# 📈 Visualización de fases
#########################################################
plt.figure(figsize=(10,6))
plt.plot(results_noise, results_gate, label="Actividad del Gate (PASO)")
plt.plot(results_noise, results_acc, label="Accuracy (%)")
plt.plot(results_noise, results_entropy, label="Entropía del Gating")

plt.title("Transición de Fase Perceptual en RESMA vs Ruido")
plt.xlabel("Ruido σ")
plt.ylabel("Magnitud")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("resma_phase_transition.png")

print("\n📌 Resultado guardado como: resma_phase_transition.png")
print("Si ves un descenso brusco del gate → descubriste una transición de fase real 🔥")
