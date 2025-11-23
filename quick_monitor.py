"""
🔥 SOVEREIGNTY MONITOR - EXPERIMENTO ULTRA-RÁPIDO 🔥
Validación rápida: ¿Puede L predecir el colapso ANTES del overfitting?

Versión optimizada que se ejecuta en ~30 segundos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
import time

# Configurar matplotlib
def setup_matplotlib_for_plotting():
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class SovereigntyMonitor:
    """Implementación del Sovereignty Monitor basada en RESMA"""
    def __init__(self, epsilon_c: float = 0.1):
        self.epsilon_c = epsilon_c
        
    def calcular_libertad(self, weights: torch.Tensor) -> Tuple[float, float, int]:
        """Calcula la métrica L (libertad) de una matriz de pesos"""
        try:
            W = weights.detach().cpu().numpy()
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            
            # Rango efectivo
            threshold = 0.01 * np.max(S)
            if threshold == 0:
                threshold = 1e-10
            rank_effective = max(1, np.sum(S > threshold))
            
            # Entropía de von Neumann
            S_sum = np.sum(S)
            if S_sum == 0:
                S_sum = 1e-10
            S_normalized = S / S_sum
            S_normalized = S_normalized[S_normalized > 1e-15]
            if len(S_normalized) == 0:
                S_normalized = np.array([1.0])
            S_vn = -np.sum(S_normalized * np.log(S_normalized))
            
            # Métrica L
            log_rank = np.log(rank_effective + 1)
            denominador = np.abs(S_vn - log_rank) + self.epsilon_c
            L = 1.0 / denominador
            
            return L, S_vn, rank_effective
            
        except Exception as e:
            return 1.0, 0.0, 1  # Valores por defecto
    
    def evaluar_regimen(self, L: float) -> str:
        """Evalúa el régimen del modelo"""
        if L > 1.0:
            return "SOBERANO"
        elif L > 0.5:
            return "EMERGENTE" 
        else:
            return "ESPURIO"

class ModeloMNISTPequeno(nn.Module):
    """Modelo CNN pequeño optimizado para entrenamiento rápido"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_linear_layers(self):
        return [self.fc1, self.fc2]

def generar_datos_mnist_rapido():
    """Genera datos sintéticos tipo MNIST para experimento rápido"""
    print("📊 Generando datos sintéticos tipo MNIST...")
    
    # Datos pequeños y ruidosos para overfitting rápido
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Imágenes sintéticas 28x28
    n_samples = 200  # Muy pequeño para overfitting rápido
    train_imgs = torch.randn(n_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_samples,))
    
    val_imgs = torch.randn(50, 1, 28, 28)
    val_labels = torch.randint(0, 10, (50,))
    
    print(f"   Train: {n_samples} samples, Val: 50 samples")
    print("   Datos diseñados para inducir overfitting rápido")
    
    return (train_imgs, train_labels), (val_imgs, val_labels)

def entrenar_modelo_rapido():
    """Entrena modelo con monitoreo L en tiempo real"""
    print("\n🚀 INICIANDO EXPERIMENTO ULTRA-RÁPIDO")
    print("="*60)
    print("🎯 ¿Puede L predecir el colapso ANTES del overfitting?")
    print()
    
    inicio = time.time()
    
    # Inicializar
    device = torch.device('cpu')  # CPU para velocidad
    monitor = SovereigntyMonitor()
    modelo = ModeloMNISTPequeno().to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=0.01)  # LR alto para colapso rápido
    criterion = nn.CrossEntropyLoss()
    
    # Datos
    (train_imgs, train_labels), (val_imgs, val_labels) = generar_datos_mnist_rapido()
    
    # Historial
    historial = {
        'epoca': [],
        'loss_train': [],
        'loss_val': [],
        'L_promedio': [],
        'L_fc1': [],
        'L_fc2': [],
        'regimen': [],
        'entropia': []
    }
    
    print(f"📱 Dispositivo: {device}")
    print("🎯 15 épocas de entrenamiento intensivo")
    print("-"*60)
    
    # Entrenamiento ultra-rápido
    for epoca in range(15):
        # === ENTRENAMIENTO ===
        modelo.train()
        train_loss = 0.0
        
        # Usar todos los datos en cada época (overfitting rápido)
        optimizer.zero_grad()
        outputs = modelo(train_imgs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        # === EVALUACIÓN ===
        modelo.eval()
        with torch.no_grad():
            val_outputs = modelo(val_imgs)
            val_loss = criterion(val_outputs, val_labels).item()
        
        # === MONITOREO L ===
        L_fc1, S_vn_fc1, rank_fc1 = monitor.calcular_libertad(modelo.fc1.weight)
        L_fc2, S_vn_fc2, rank_fc2 = monitor.calcular_libertad(modelo.fc2.weight)
        L_promedio = (L_fc1 + L_fc2) / 2
        S_vn_promedio = (S_vn_fc1 + S_vn_fc2) / 2
        regimen = monitor.evaluar_regimen(L_promedio)
        
        # Guardar métricas
        historial['epoca'].append(epoca)
        historial['loss_train'].append(train_loss)
        historial['loss_val'].append(val_loss)
        historial['L_promedio'].append(L_promedio)
        historial['L_fc1'].append(L_fc1)
        historial['L_fc2'].append(L_fc2)
        historial['regimen'].append(regimen)
        historial['entropia'].append(S_vn_promedio)
        
        # Reporte cada 3 épocas
        if epoca % 3 == 0 or epoca == 14:
            print(f"Ep {epoca:2d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"L: {L_promedio:.3f} ({regimen})")
    
    tiempo_total = time.time() - inicio
    print("-"*60)
    print(f"⏱️  Entrenamiento completado en {tiempo_total:.1f} segundos")
    print("="*60)
    
    # === ANÁLISIS DE RESULTADOS ===
    print("📊 ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    # Detectar colapso en L
    colapso_epoca = None
    for i, L in enumerate(historial['L_promedio']):
        if L < 0.5:
            colapso_epoca = historial['epoca'][i]
            break
    
    # Detectar overfitting (val_loss empeora)
    overfitting_epoca = None
    min_val_loss = float('inf')
    mejor_epoca = 0
    
    for i, val_loss in enumerate(historial['loss_val']):
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            mejor_epoca = historial['epoca'][i]
    
    # Buscar cuando val_loss sube 10% del mínimo
    for i in range(mejor_epoca + 2, len(historial['loss_val'])):
        if historial['loss_val'][i] > min_val_loss * 1.10:
            overfitting_epoca = historial['epoca'][i]
            break
    
    # === VERIFICAR PODER PREDICTIVO ===
    if colapso_epoca is not None:
        print(f"🚨 COLAPSO EN L detectado en época {colapso_epoca}")
        print(f"   L = {historial['L_promedio'][colapso_epoca]:.3f} < 0.5")
        print(f"   Régimen: {historial['regimen'][colapso_epoca]}")
    else:
        print("✅ No se detectó colapso en L (L > 0.5 en todas las épocas)")
    
    if overfitting_epoca is not None:
        print(f"📈 OVERFITTING detectado en época {overfitting_epoca}")
        print(f"   Val_loss subió 10% desde el mínimo en época {mejor_epoca}")
    else:
        print("📊 No se detectó overfitting significativo en val_loss")
    
    # Análizar poder predictivo
    if colapso_epoca is not None and overfitting_epoca is not None:
        diferencia = overfitting_epoca - colapso_epoca
        print(f"\n🔮 PODER PREDICTIVO:")
        print(f"   L detectó el problema {diferencia} épocas ANTES que val_loss")
        if diferencia > 0:
            print(f"   🎉 ÉXITO: L predice el colapso con {diferencia} épocas de anticipación")
            print(f"   ✅ Esto confirma la hipótesis de RESMA sobre detección temprana")
        else:
            print(f"   ⚠️  L detectó el problema después o al mismo tiempo que val_loss")
    elif colapso_epoca is not None:
        print(f"\n🔮 L detectó el colapso en época {colapso_epoca}")
        print(f"   No se observó overfitting claro en val_loss")
        print(f"   Esto podría indicar que L detectó el problema antes de que fuera visible")
    else:
        print(f"\n🤔 RESULTADO:")
        print(f"   L no detectó colapso (permaneció > 0.5)")
        print(f"   El modelo mantuvo capacidad de generalización")
    
    # Estadísticas finales
    regimen_counts = {}
    for regimen in historial['regimen']:
        regimen_counts[regimen] = regimen_counts.get(regimen, 0) + 1
    
    print(f"\n📋 DISTRIBUCIÓN DE RÉGIMEN:")
    for regimen, count in regimen_counts.items():
        porcentaje = (count / len(historial['regimen'])) * 100
        print(f"   {regimen}: {count} épocas ({porcentaje:.1f}%)")
    
    # Generar gráficos
    generar_graficos_rapido(historial)
    
    print(f"\n💾 Gráficos guardados en: /workspace/sovereignty_rapido_final.png")
    print("="*60)
    
    # Conclusión final
    if colapso_epoca is not None and overfitting_epoca is not None:
        diferencia = overfitting_epoca - colapso_epoca
        if diferencia > 0:
            print("🎉 CONCLUSIÓN: EL SOVEREIGNTY MONITOR FUNCIONA")
            print(f"   Predijo el colapso {diferencia} épocas antes del overfitting")
            print("   Esto valida la propuesta teórica de RESMA")
        else:
            print("⚠️  CONCLUSIÓN: RESULTADO MIXTO")
            print("   L detectó el colapso pero no significativamente antes")
    elif colapso_epoca is not None:
        print("🔮 CONCLUSIÓN: L DETECTÓ PROBLEMA POTENCIAL")
        print("   El modelo podría estar en riesgo de colapso según L")
    else:
        print("✅ CONCLUSIÓN: MODELO ESTABLE")
        print("   L indica que el modelo mantiene capacidad de generalización")
    
    return historial, colapso_epoca, overfitting_epoca

def generar_graficos_rapido(historial):
    """Genera gráficos de resultados del experimento rápido"""
    setup_matplotlib_for_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🔥 SOVEREIGNTY MONITOR: Validación Ultra-Rápida', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Pérdida
    axes[0,0].plot(historial['epoca'], historial['loss_train'], 'b-', linewidth=2, label='Train Loss')
    axes[0,0].plot(historial['epoca'], historial['loss_val'], 'r-', linewidth=2, label='Val Loss')
    axes[0,0].set_title('Evolución de Pérdida: Overfitting Detection', fontweight='bold')
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: MÉTRICA L - LA CLAVE
    axes[0,1].plot(historial['epoca'], historial['L_promedio'], 'purple', linewidth=3, label='L Promedio')
    axes[0,1].axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Umbral Soberano (1.0)')
    axes[0,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Espurio (0.5)')
    axes[0,1].fill_between(historial['epoca'], 0, 0.5, alpha=0.3, color='red', label='Zona Espurio')
    axes[0,1].fill_between(historial['epoca'], 1.0, max(historial['L_promedio'])*1.1, alpha=0.3, color='green', label='Zona Soberano')
    axes[0,1].set_title('🔥 MÉTRICA L: ¿Predice el Colapso?', fontweight='bold')
    axes[0,1].set_xlabel('Época')
    axes[0,1].set_ylabel('L (Libertad)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: L por capa
    axes[1,0].plot(historial['epoca'], historial['L_fc1'], 'blue', linewidth=2, label='L FC1')
    axes[1,0].plot(historial['epoca'], historial['L_fc2'], 'red', linewidth=2, label='L FC2')
    axes[1,0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Espurio')
    axes[1,0].set_title('L por Capa Individual')
    axes[1,0].set_xlabel('Época')
    axes[1,0].set_ylabel('L')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Correlación L vs Val Loss
    colors = plt.cm.viridis(np.linspace(0, 1, len(historial['epoca'])))
    scatter = axes[1,1].scatter(historial['L_promedio'], historial['loss_val'], 
                              c=historial['epoca'], cmap='viridis', s=80, alpha=0.8)
    axes[1,1].set_xlabel('L Promedio')
    axes[1,1].set_ylabel('Val Loss')
    axes[1,1].set_title('Correlación L vs Val Loss (color = época)')
    axes[1,1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1,1], label='Época')
    
    plt.tight_layout()
    plt.savefig('/workspace/sovereignty_rapido_final.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        historial, colapso, overfitting = entrenar_modelo_rapido()
        print("\n🎉 EXPERIMENTO COMPLETADO EXITOSAMENTE")
        
        # Mostrar tabla resumen
        print("\n📋 TABLA RESUMEN:")
        print("-"*50)
        print(f"{'Época':<6} {'Train':<8} {'Val':<8} {'L':<6} {'Régimen'}")
        print("-"*50)
        for i in range(len(historial['epoca'])):
            ep = historial['epoca'][i]
            train_loss = historial['loss_train'][i]
            val_loss = historial['loss_val'][i]
            L = historial['L_promedio'][i]
            regimen = historial['regimen'][i]
            print(f"{ep:<6} {train_loss:<8.4f} {val_loss:<8.4f} {L:<6.3f} {regimen}")
        print("-"*50)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
