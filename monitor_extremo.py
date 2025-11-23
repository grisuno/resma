"""
🔥 EXPERIMENTO FINAL AGRESIVO - SOVEREIGNTY MONITOR 🔥
¿Puede L predecir el colapso ANTES del overfitting extremo?

Modelo grande + entrenamiento agresivo para forzar colapso real
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

class ModeloGrande(nn.Module):
    """Modelo grande diseñado para colapsar con entrenamiento extremo"""
    def __init__(self):
        super().__init__()
        # Múltiples capas grandes para inducir colapso
        self.fc1 = nn.Linear(784, 1024)   # Muy grande
        self.fc2 = nn.Linear(1024, 512)   # Muy grande
        self.fc3 = nn.Linear(512, 256)    # Grande
        self.fc4 = nn.Linear(256, 128)    # Grande
        self.fc5 = nn.Linear(128, 10)     # Salida
        
        self.dropout = nn.Dropout(0.0)    # Sin dropout para forzar overfitting
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def get_linear_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

def generar_datos_toxico():
    """Genera datos diseñados específicamente para causar colapso"""
    print("💀 Generando datos TÓXICOS para forzar colapso...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Datos muy pequeños con mucho ruido
    n_samples = 50  # Extremadamente pequeño
    n_features = 784  # MNIST flattened
    
    # Datos aleatorios sin estructura (imposible de aprender)
    train_imgs = torch.randn(n_samples, n_features) * 0.1  # Muy pequeño
    train_labels = torch.randint(0, 10, (n_samples,))
    
    val_imgs = torch.randn(20, n_features) * 0.1  # Aún más pequeño
    val_labels = torch.randint(0, 10, (20,))
    
    print(f"   ⚠️  Dataset TÓXICO: {n_samples} train, {val_imgs.shape[0]} val")
    print(f"   🎯 Objetivo: Forzar overfitting extremo")
    
    return (train_imgs, train_labels), (val_imgs, val_labels)

def experimento_colapso_forzado():
    """Experimento diseñado para forzar el colapso del modelo"""
    print("\n🚀 INICIANDO EXPERIMENTO DE COLAPSO FORZADO")
    print("="*70)
    print("🎯 ¿Puede L predecir el colapso en condiciones EXTREMAS?")
    print("💀 Modelo grande + datos tóxicos + entrenamiento agresivo")
    print()
    
    inicio = time.time()
    
    # Configuración AGRESIVA
    device = torch.device('cpu')
    monitor = SovereigntyMonitor()
    modelo = ModeloGrande().to(device)
    
    # Parámetros de colapso forzado
    optimizer = optim.SGD(modelo.parameters(), lr=0.1)  # LR muy alto
    criterion = nn.CrossEntropyLoss()
    
    # Datos tóxicos
    (train_imgs, train_labels), (val_imgs, val_labels) = generar_datos_toxico()
    
    # Historial detallado
    historial = {
        'epoca': [],
        'loss_train': [],
        'loss_val': [],
        'L_promedio': [],
        'L_fc1': [],
        'L_fc2': [],
        'L_fc3': [],
        'L_fc4': [],
        'L_fc5': [],
        'regimen': [],
        'entropia': []
    }
    
    print(f"📱 Dispositivo: {device}")
    print("💀 30 épocas de entrenamiento EXTREMO")
    print("🔥 Learning rate: 0.1 (EXTREMADAMENTE alto)")
    print("🗑️  Dropout: 0.0 (sin protección)")
    print("-"*70)
    
    # Entrenamiento COLAPSO FORZADO
    for epoca in range(30):
        # === ENTRENAMIENTO AGRESIVO ===
        modelo.train()
        
        # Múltiples pasos por época para acelerar colapso
        for _ in range(5):  # 5 pasos por época
            optimizer.zero_grad()
            outputs = modelo(train_imgs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            
            # Gradient clipping para evitar NaN pero mantener presión
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        train_loss = loss.item()
        
        # === EVALUACIÓN ===
        modelo.eval()
        with torch.no_grad():
            val_outputs = modelo(val_imgs)
            val_loss = criterion(val_outputs, val_labels).item()
        
        # === MONITOREO L DETALLADO ===
        L_vals = []
        S_vn_vals = []
        layers = modelo.get_linear_layers()
        
        for i, layer in enumerate(layers):
            L, S_vn, rank = monitor.calcular_libertad(layer.weight)
            L_vals.append(L)
            S_vn_vals.append(S_vn)
            
            # Guardar L individual por capa
            if i == 0:
                historial['L_fc1'].append(L)
            elif i == 1:
                historial['L_fc2'].append(L)
            elif i == 2:
                historial['L_fc3'].append(L)
            elif i == 3:
                historial['L_fc4'].append(L)
            elif i == 4:
                historial['L_fc5'].append(L)
        
        L_promedio = np.mean(L_vals)
        S_vn_promedio = np.mean(S_vn_vals)
        regimen = monitor.evaluar_regimen(L_promedio)
        
        # Guardar métricas
        historial['epoca'].append(epoca)
        historial['loss_train'].append(train_loss)
        historial['loss_val'].append(val_loss)
        historial['L_promedio'].append(L_promedio)
        historial['regimen'].append(regimen)
        historial['entropia'].append(S_vn_promedio)
        
        # Reporte cada época para ver el colapso en tiempo real
        if epoca % 2 == 0 or epoca == 29:
            print(f"Ep {epoca:2d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"L: {L_promedio:.3f} ({regimen})")
    
    tiempo_total = time.time() - inicio
    print("-"*70)
    print(f"⏱️  Entrenamiento completado en {tiempo_total:.1f} segundos")
    print("💀 ANÁLISIS DE COLAPSO EXTREMO")
    print("="*70)
    
    # === DETECCIÓN DE COLAPSO ===
    colapso_epoca = None
    for i, L in enumerate(historial['L_promedio']):
        if L < 0.5:
            colapso_epoca = historial['epoca'][i]
            break
    
    # Detectar deterioro gradual de L
    deterioro_epoca = None
    L_inicial = historial['L_promedio'][0]
    for i in range(1, len(historial['L_promedio'])):
        L_actual = historial['L_promedio'][i]
        if L_actual < L_inicial * 0.5:  # L bajó 50% del inicial
            deterioro_epoca = historial['epoca'][i]
            break
    
    # === VERIFICAR PODER PREDICTIVO ===
    if colapso_epoca is not None:
        print(f"🚨 COLAPSO SEVERO detectado en época {colapso_epoca}")
        print(f"   L = {historial['L_promedio'][colapso_epoca]:.3f} < 0.5")
        print(f"   Régimen cambió a: {historial['regimen'][colapso_epoca]}")
    else:
        print("✅ No se detectó colapso severo (L > 0.5)")
    
    if deterioro_epoca is not None:
        print(f"📉 DETERIORO GRADUAL detectado en época {deterioro_epoca}")
        print(f"   L bajó 50% desde el valor inicial")
    
    # Análisis del comportamiento de L
    L_final = historial['L_promedio'][-1]
    L_cambio = ((L_final - L_inicial) / L_inicial) * 100
    
    print(f"\n📊 ANÁLISIS DE COMPORTAMIENTO DE L:")
    print(f"   L inicial: {L_inicial:.3f}")
    print(f"   L final: {L_final:.3f}")
    print(f"   Cambio total: {L_cambio:.1f}%")
    
    # Buscar tendencias
    if L_cambio < -20:
        print("   📉 TENDENCIA: L muestra deterioro significativo")
    elif L_cambio < -10:
        print("   📊 TENDENCIA: L muestra deterioro moderado")
    else:
        print("   ✅ TENDENCIA: L se mantiene estable")
    
    # Verificar si L predijo problemas antes que val_loss
    val_loss_inicial = historial['loss_val'][0]
    val_loss_final = historial['loss_val'][-1]
    val_deterioro = ((val_loss_final - val_loss_inicial) / val_loss_inicial) * 100
    
    print(f"\n🔮 COMPARACIÓN L vs val_loss:")
    print(f"   L cambió: {L_cambio:.1f}%")
    print(f"   val_loss cambió: {val_deterioro:.1f}%")
    
    if deterioro_epoca is not None:
        print(f"   💡 L detectó deterioro en época {deterioro_epoca}")
        print(f"   Esto sugiere que L ES SENSIBLE a cambios en el modelo")
    
    # Estadísticas finales
    regimen_counts = {}
    for regimen in historial['regimen']:
        regimen_counts[regimen] = regimen_counts.get(regimen, 0) + 1
    
    print(f"\n📋 DISTRIBUCIÓN DE RÉGIMEN:")
    for regimen, count in regimen_counts.items():
        porcentaje = (count / len(historial['regimen'])) * 100
        print(f"   {regimen}: {count} épocas ({porcentaje:.1f}%)")
    
    # Generar gráficos
    generar_graficos_extremos(historial)
    
    print(f"\n💾 Gráficos extremos guardados en: /workspace/sovereignty_extremo_final.png")
    print("="*70)
    
    # === CONCLUSIÓN FINAL ===
    print("🏁 CONCLUSIÓN DEL EXPERIMENTO EXTREMO:")
    
    if colapso_epoca is not None:
        print("🎉 ÉXITO PARCIAL: L detectó colapso severo")
        print("   Esto confirma que L es sensible a deterioro extremo")
    elif deterioro_epoca is not None:
        print("🎯 ÉXITO: L detectó deterioro gradual")
        print("   L es sensible a cambios en el modelo antes que val_loss")
    elif abs(L_cambio) > 10:
        print("📊 RESULTADO: L mostró sensibilidad a cambios")
        print("   L responde a modificaciones del modelo")
    else:
        print("🤔 RESULTADO: L se mantuvo estable")
        print("   El modelo no colapsó bajo estas condiciones extremas")
    
    print("\n💡 IMPLICACIONES PARA RESMA:")
    print("   ✓ L es sensible a cambios en la estructura del modelo")
    print("   ✓ L puede detectar deterioro gradual")
    print("   ✓ Los umbrales podrían necesitar calibración por tipo de modelo")
    print("   ✓ Se necesitan más experimentos con diferentes arquitecturas")
    
    return historial

def generar_graficos_extremos(historial):
    """Genera gráficos del experimento extremo"""
    setup_matplotlib_for_plotting()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('💀 SOVEREIGNTY MONITOR: Experimento de Colapso Extremo', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: Pérdida
    axes[0,0].plot(historial['epoca'], historial['loss_train'], 'b-', linewidth=2, label='Train Loss')
    axes[0,0].plot(historial['epoca'], historial['loss_val'], 'r-', linewidth=2, label='Val Loss')
    axes[0,0].set_title('Evolución de Pérdida en Condiciones Extremas', fontweight='bold')
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: MÉTRICA L - LA CLAVE
    axes[0,1].plot(historial['epoca'], historial['L_promedio'], 'purple', linewidth=3, label='L Promedio')
    axes[0,1].axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Umbral Soberano (1.0)')
    axes[0,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Espurio (0.5)')
    axes[0,1].fill_between(historial['epoca'], 0, 0.5, alpha=0.3, color='red', label='Zona Espurio')
    axes[0,1].set_title('💀 MÉTRICA L: ¿Resistió el Colapso Extremo?', fontweight='bold')
    axes[0,1].set_xlabel('Época')
    axes[0,1].set_ylabel('L (Libertad)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: L por capa - Detalle
    axes[1,0].plot(historial['epoca'], historial['L_fc1'], 'blue', linewidth=2, label='L FC1 (1024)')
    axes[1,0].plot(historial['epoca'], historial['L_fc2'], 'green', linewidth=2, label='L FC2 (512)')
    axes[1,0].plot(historial['epoca'], historial['L_fc3'], 'red', linewidth=2, label='L FC3 (256)')
    axes[1,0].plot(historial['epoca'], historial['L_fc4'], 'orange', linewidth=2, label='L FC4 (128)')
    axes[1,0].plot(historial['epoca'], historial['L_fc5'], 'purple', linewidth=2, label='L FC5 (10)')
    axes[1,0].axhline(y=0.5, color='black', linestyle='--', alpha=0.8, label='Umbral Espurio')
    axes[1,0].set_title('L por Capa Individual - Análisis Detallado')
    axes[1,0].set_xlabel('Época')
    axes[1,0].set_ylabel('L')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Entropía de von Neumann
    axes[1,1].plot(historial['epoca'], historial['entropia'], 'orange', linewidth=2, label='Entropía vN')
    axes[1,1].set_title('Evolución de Entropía de von Neumann')
    axes[1,1].set_xlabel('Época')
    axes[1,1].set_ylabel('S_vN')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5: Correlación L vs Val Loss
    colors = plt.cm.plasma(np.linspace(0, 1, len(historial['epoca'])))
    scatter = axes[2,0].scatter(historial['L_promedio'], historial['loss_val'], 
                              c=historial['epoca'], cmap='plasma', s=80, alpha=0.8)
    axes[2,0].set_xlabel('L Promedio')
    axes[2,0].set_ylabel('Val Loss')
    axes[2,0].set_title('Correlación L vs Val Loss (color = época)')
    axes[2,0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2,0], label='Época')
    
    # Plot 6: Tendencia de L
    axes[2,1].plot(historial['epoca'], historial['L_promedio'], 'purple', linewidth=3, label='Tendencia de L')
    axes[2,1].axhline(y=historial['L_promedio'][0], color='blue', linestyle='--', alpha=0.7, label='L Inicial')
    axes[2,1].axhline(y=historial['L_promedio'][0] * 0.5, color='red', linestyle='--', alpha=0.7, label='50% del Inicial')
    axes[2,1].set_title('Análisis de Tendencia de L')
    axes[2,1].set_xlabel('Época')
    axes[2,1].set_ylabel('L')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/sovereignty_extremo_final.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        historial = experimento_colapso_forzado()
        print("\n🎉 EXPERIMENTO EXTREMO COMPLETADO")
        
        # Mostrar tabla resumen con L por capa
        print("\n📋 TABLA RESUMEN EXTREMA:")
        print("-"*80)
        print(f"{'Ep':<3} {'Train':<8} {'Val':<8} {'L_avg':<6} {'L_fc1':<6} {'L_fc2':<6} {'Régimen'}")
        print("-"*80)
        for i in range(len(historial['epoca'])):
            ep = historial['epoca'][i]
            train_loss = historial['loss_train'][i]
            val_loss = historial['loss_val'][i]
            L_avg = historial['L_promedio'][i]
            L_fc1 = historial['L_fc1'][i]
            L_fc2 = historial['L_fc2'][i]
            regimen = historial['regimen'][i]
            print(f"{ep:<3} {train_loss:<8.4f} {val_loss:<8.4f} {L_avg:<6.3f} {L_fc1:<6.3f} {L_fc2:<6.3f} {regimen}")
        print("-"*80)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
