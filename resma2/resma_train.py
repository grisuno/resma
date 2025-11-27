import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resma_core import RESMABrain

# =============================
# 🔥 CONFIGURACIÓN
# =============================
EPOCHS = 6
LR = 0.0008
BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🧠 Entrenando RESMA en {DEVICE}...\n")

# =============================
# 1) CARGA DE DATA
# =============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH)

# =============================
# 2) MODELO RESMA
# =============================
model = RESMABrain(784, 128, 10).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# =============================
# 🔥 ENTRENAMIENTO
# =============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        data = data.view(-1, 784).to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"📈 Epoch {epoch+1}/{EPOCHS} | Loss = {total_loss/len(train_loader):.4f}")

# =============================
# 🔍 TEST FINAL
# =============================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784).to(DEVICE)
        target = target.to(DEVICE)
        out = model(data)
        pred = out.argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)

acc = correct/total*100
print(f"\n🏁 Precisión final RESMA en MNIST: {acc:.2f}%")

# =============================
# 💾 GUARDAMOS PESOS ENTRENADOS
# =============================
torch.save(model.state_dict(), "resma_trained.pt")
print("\n💾 Modelo guardado → resma_trained.pt\n")
print("Ahora puedes correr resma_noise_phase_test.py y resma_breakpoint.py con un CEREBRO REAL 🔥")
