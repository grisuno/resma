import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from resma_core import RESMABrain
from resma_observer import RESMAObserver
from monitor import SovereigntyMonitor

# ===========================
# CONFIG GLOBAL
# ===========================
EPOCHS = 8
BATCH = 64
LR = 0.0008
NOISE = 2.2           # alto pero evitable si RESMA está funcionando
DEVICE = "cpu"        # puedes subir a cuda si tienes GPU

# ===========================
# FUNCIONES
# ===========================

def inject_noise(x, sigma):
    return x + torch.randn_like(x) * sigma

def train_epoch(model, loader, optim, obs, epoch):
    model.train()
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        data = data.view(data.size(0), -1)

        noisy = inject_noise(data, NOISE)

        optim.zero_grad()
        out = model(noisy)
        loss = nn.CrossEntropyLoss()(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        pred = out.argmax(1)
        correct += pred.eq(target).sum().item()
        total += len(target)

    acc = correct / total * 100
    state = obs.step(epoch)

    print(f"⚡ Ep {epoch} | Acc={acc:.2f}% | L={state.L_structural:.3f}  "
          f"| C={state.C_dynamic:.2f} | Ξ={state.Criticality:.3f} | {state.Phase}")

    return acc

# ===========================
# MAIN
# ===========================

def run():
    print("\n\n🚀 RESMA 5.2 — EJECUCIÓN PRINCIPAL")
    print(f"Ruido activo: σ={NOISE}\n")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

    model = RESMABrain(784, 128, 10).to(DEVICE)
    monitor = SovereigntyMonitor(verbose=False)     # LIBER-MONITOR SVD
    observer = RESMAObserver(model)

    optimzr = optim.Adam(model.parameters(), lr=LR)

    print("\n────── ENTRENANDO ──────\n")
    for ep in range(1, EPOCHS+1):
        train_epoch(model, trainloader, optimzr, observer, ep)

    torch.save(model.state_dict(), "resma_trained.pt")
    observer.report("resma_training_log.json")

    print("\n\n💾 Guardado: resma_trained.pt")
    print("📄 Log: resma_training_log.json")
    print("🧠 FIN DEL ENTRENAMIENTO\n")


if __name__ == "__main__":
    run()
