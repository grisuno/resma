"""
🔥 EXPERIMENTO COMPLETO SOVEREIGNTY MONITOR 🔥
Demostración completa: ¿Puede L predecir el colapso ANTES del overfitting?

Este script entrena un modelo CNN en MNIST con monitoreo continuo de la métrica L
para verificar si predice el overfitting antes que las métricas tradicionales.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
import os

# Configurar matplotlib
def setup_matplotlib_for_plotting():
    """Setup matplotlib para visualización"""
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

class SovereigntyMonitor:
    """
    Implementación del Sovereignty Monitor basada en RESMA
    
    Calcula L = 1 / (|S_vN(ρ) − log(rank(W) + 1)| + ε_c)
    """
    def __init__(self, epsilon_c: float = 0.1):
        self.epsilon_c = epsilon_c
        
    def calcular_libertad(self, weights: torch.Tensor) -> Tuple[float, float, int]:
        """
        Calcula la métrica L (libertad) de una matriz de pesos
        
        Returns:
            tuple: (L, S_vn, rank_effective)
        """
        try:
            # Mover a CPU y convertir a numpy
            W = weights.detach().cpu().numpy()
            
            # SVD para obtener valores singulares
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            
            # Rango efectivo (valores singulares > 1% del máximo)
            threshold = 0.01 * np.max(S)
            if threshold == 0:
                threshold = 1e-10
            
            rank_effective = np.sum(S > threshold)
            if rank_effective == 0:
                rank_effective = 1
            
            # Entropía de von Neumann aproximada
            S_sum = np.sum(S)
            if S_sum == 0:
                S_sum = 1e-10
            
            S_normalized = S / S_sum
            # Filtrar valores muy pequeños para evitar log(0)
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
            print(f"Error en calcular_libertad: {e}")
            return 1.0, 0.0, 1  # Valores por defecto en caso de error
    
    def evaluar_regimen(self, L: float) -> str:
        """Evalúa el régimen del modelo"""
        if L > 1.0:
            return "SOBERANO"
        elif L > 0.5:
            return "EMERGENTE" 
        else:
            return "ESPURIO"

class CNNMNIST(nn.Module):
    """CNN para MNIST con arquitectura diseñada para monitoreo"""
    def __init__(self):
        super().__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        
        # Capas completamente conectadas para monitoreo
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Capa 1 para monitoreo
        self.fc2 = nn.Linear(128, 64)          # Capa 2 para monitoreo
        self.fc3 = nn.Linear(64, 10)           # Capa de salida
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_linear_layers(self):
        """Retorna todas las capas lineales para monitoreo"""
        return [self.fc1, self.fc2, self.fc3]

def cargar_datos():
    """Carga y prepara el dataset MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dataset pequeño para entrenamiento rápido
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Subconjunto más pequeño para experimento rápido
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    test_subset = torch.utils.data.Subset(test_dataset, range(200))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

class ExperimentoCompleto:
    """Experimento completo para validar el Sovereignty Monitor"""
    
    def __init__(self, num_epochs=25):
        print("🔥 INICIANDO EXPERIMENTO COMPLETO SOVEREIGNTY MONITOR")
        print("=" * 70)
        print("🎯 Objetivo: Verificar si L predice el colapso ANTES del overfitting")
        print()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Dispositivo: {self.device}")
        
        self.num_epochs = num_epochs
        self.monitor = SovereigntyMonitor()
        
        # Cargar datos
        self.train_loader, self.test_loader = cargar_datos()
        print(f"📊 Datos: {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} test")
        
        # Modelo
        self.modelo = CNNMNIST().to(self.device)
        self.optimizer = optim.Adam(self.modelo.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Historial de métricas (una entrada por época)
        self.historial = {
            'epoca': [],
            'loss_train': [],
            'loss_val': [],
            'accuracy_train': [],
            'accuracy_val': [],
            'L_promedio': [],
            'L_fc1': [],
            'L_fc2': [],
            'L_fc3': [],
            'regimen_promedio': [],
            'entropia_promedio': [],
            'rango_promedio': []
        }
        
        print(f"🎯 Épocas objetivo: {num_epochs}")
        print("=" * 70)
        
    def calcular_metricas_sovereignty(self):
        """Calcula métricas L para todas las capas lineales"""
        L_vals, S_vn_vals, rank_vals = [], [], []
        
        for layer in self.modelo.get_linear_layers():
            L, S_vn, rank_eff = self.monitor.calcular_libertad(layer.weight)
            L_vals.append(L)
            S_vn_vals.append(S_vn)
            rank_vals.append(rank_eff)
        
        if not L_vals:
            return 1.0, 1.0, 1.0, 0.0, 1, "SOBERANO"
        
        # Promedio simple
        L_promedio = np.mean(L_vals)
        S_vn_promedio = np.mean(S_vn_vals)
        rank_promedio = np.mean(rank_vals)
        regimen = self.monitor.evaluar_regimen(L_promedio)
        
        return L_promedio, L_vals[0], L_vals[1], L_vals[2], S_vn_promedio, rank_promedio, regimen
    
    def entrenar_epoca(self, epoca):
        """Entrena una época completa"""
        self.modelo.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.modelo(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        return train_loss / len(self.train_loader), 100. * train_correct / train_total
    
    def evaluar_epoca(self):
        """Evalúa el modelo en el conjunto de validación"""
        self.modelo.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.modelo(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        return val_loss / len(self.test_loader), 100. * val_correct / val_total
    
    def ejecutar_experimento(self):
        """Ejecuta el experimento completo"""
        print("\n🚀 Iniciando entrenamiento con monitoreo continuo...")
        print("-" * 70)
        
        for epoca in range(self.num_epochs):
            # Entrenar época
            loss_train, acc_train = self.entrenar_epoca(epoca)
            
            # Evaluar época
            loss_val, acc_val = self.evaluar_epoca()
            
            # Calcular métricas L
            L_promedio, L_fc1, L_fc2, L_fc3, S_vn_promedio, rank_promedio, regimen = self.calcular_metricas_sovereignty()
            
            # Guardar en historial
            self.historial['epoca'].append(epoca)
            self.historial['loss_train'].append(loss_train)
            self.historial['loss_val'].append(loss_val)
            self.historial['accuracy_train'].append(acc_train)
            self.historial['accuracy_val'].append(acc_val)
            self.historial['L_promedio'].append(L_promedio)
            self.historial['L_fc1'].append(L_fc1)
            self.historial['L_fc2'].append(L_fc2)
            self.historial['L_fc3'].append(L_fc3)
            self.historial['regimen_promedio'].append(regimen)
            self.historial['entropia_promedio'].append(S_vn_promedio)
            self.historial['rango_promedio'].append(rank_promedio)
            
            # Reporte cada 3 épocas
            if epoca % 3 == 0 or epoca == self.num_epochs - 1:
                print(f"Ep {epoca:2d} | "
                      f"Train: {loss_train:.4f}/{acc_train:.1f}% | "
                      f"Val: {loss_val:.4f}/{acc_val:.1f}% | "
                      f"L: {L_promedio:.3f} ({regimen})")
        
        print("-" * 70)
        print("📊 ANÁLISIS DE RESULTADOS")
        print("=" * 70)
        
        # Buscar punto de colapso en L
        colapso_epoca = None
        for i, L in enumerate(self.historial['L_promedio']):
            if L < 0.5:
                colapso_epoca = self.historial['epoca'][i]
                break
        
        # Buscar punto de overfitting (cuando val_loss empieza a subir)
        overfitting_epoca = None
        min_val_loss = float('inf')
        for i, val_loss in enumerate(self.historial['loss_val']):
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                mejor_epoca = self.historial['epoca'][i]
        
        # Buscar cuando la val_loss empieza a empeorar consistentemente
        for i in range(mejor_epoca + 2, len(self.historial['loss_val'])):
            if self.historial['loss_val'][i] > self.historial['loss_val'][i-1] * 1.05:  # 5% de aumento
                overfitting_epoca = self.historial['epoca'][i]
                break
        
        # Análisis de resultados
        if colapso_epoca is not None:
            print(f"🚨 COLAPSO EN L detectado en época {colapso_epoca}")
            print(f"   L_promedio = {self.historial['L_promedio'][colapso_epoca]:.3f} < 0.5")
            print(f"   Régimen cambió a: {self.historial['regimen_promedio'][colapso_epoca]}")
        else:
            print("✅ No se detectó colapso en L (L > 0.5 en todas las épocas)")
        
        if overfitting_epoca is not None:
            print(f"📈 OVERFITTING detectado en época {overfitting_epoca}")
            print(f"   Val_loss subió 5% desde el mínimo en época {mejor_epoca}")
        else:
            print("📊 No se detectó overfitting significativo")
        
        # Verificar poder predictivo
        if colapso_epoca is not None and overfitting_epoca is not None:
            diferencia = overfitting_epoca - colapso_epoca
            print(f"\n🔮 PODER PREDICTIVO:")
            print(f"   L detectó el problema {diferencia} épocas ANTES que val_loss")
            if diferencia > 0:
                print("   ✅ ÉXITO: L predice el colapso con {diferencia} épocas de anticipación")
            else:
                print("   ❌ L detectó el problema después o al mismo tiempo que val_loss")
        elif colapso_epoca is not None:
            print(f"\n🔮 L detectó el colapso en época {colapso_epoca}")
            print("   No se detectó overfitting claro en val_loss")
        elif overfitting_epoca is not None:
            print(f"\n🔮 val_loss detectó overfitting en época {overfitting_epoca}")
            print("   L no predijo el colapso (permaneció > 0.5)")
        
        # Estadísticas de régimen
        regimen_counts = {}
        for regimen in self.historial['regimen_promedio']:
            regimen_counts[regimen] = regimen_counts.get(regimen, 0) + 1
        
        print(f"\n📋 DISTRIBUCIÓN DE RÉGIMEN:")
        for regimen, count in regimen_counts.items():
            porcentaje = (count / len(self.historial['regimen_promedio'])) * 100
            print(f"   {regimen}: {count} épocas ({porcentaje:.1f}%)")
        
        # Generar gráficos
        self.generar_graficos()
        
        print(f"\n💾 Gráficos guardados en: /workspace/sovereignty_completo_results.png")
        print("=" * 70)
        
        return colapso_epoca, overfitting_epoca
    
    def generar_graficos(self):
        """Genera gráficos comprehensivos de resultados"""
        setup_matplotlib_for_plotting()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('🔥 SOVEREIGNTY MONITOR: Análisis Completo de Detección de Colapso', 
                     fontsize=18, fontweight='bold')
        
        # Plot 1: Evolución de pérdida
        axes[0,0].plot(self.historial['epoca'], self.historial['loss_train'], 'b-', linewidth=2, label='Train Loss')
        axes[0,0].plot(self.historial['epoca'], self.historial['loss_val'], 'r-', linewidth=2, label='Val Loss')
        axes[0,0].set_title('Evolución de Pérdida: Detección de Overfitting', fontweight='bold')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Métrica L promedio - LA CLAVE
        axes[0,1].plot(self.historial['epoca'], self.historial['L_promedio'], 'purple', linewidth=3, label='L Promedio')
        axes[0,1].axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='Umbral Soberano (1.0)')
        axes[0,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Espurio (0.5)')
        axes[0,1].fill_between(self.historial['epoca'], 0, 0.5, alpha=0.3, color='red', label='Zona Espurio')
        axes[0,1].fill_between(self.historial['epoca'], 1.0, max(self.historial['L_promedio'])*1.1, alpha=0.3, color='green', label='Zona Soberano')
        axes[0,1].set_title('🔥 MÉTRICA L: ¿Predice el Colapso?', fontweight='bold')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('L (Libertad)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: L por capa
        axes[1,0].plot(self.historial['epoca'], self.historial['L_fc1'], 'blue', linewidth=2, label='L FC1')
        axes[1,0].plot(self.historial['epoca'], self.historial['L_fc2'], 'green', linewidth=2, label='L FC2')
        axes[1,0].plot(self.historial['epoca'], self.historial['L_fc3'], 'red', linewidth=2, label='L FC3')
        axes[1,0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.8, label='Umbral Espurio')
        axes[1,0].set_title('L por Capa Individual')
        axes[1,0].set_xlabel('Época')
        axes[1,0].set_ylabel('L')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Evolución de accuracy
        axes[1,1].plot(self.historial['epoca'], self.historial['accuracy_train'], 'b-', linewidth=2, label='Train Acc')
        axes[1,1].plot(self.historial['epoca'], self.historial['accuracy_val'], 'r-', linewidth=2, label='Val Acc')
        axes[1,1].set_title('Evolución de Accuracy')
        axes[1,1].set_xlabel('Época')
        axes[1,1].set_ylabel('Accuracy (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 5: Entropía de von Neumann
        axes[2,0].plot(self.historial['epoca'], self.historial['entropia_promedio'], 'orange', linewidth=2, label='Entropía vN')
        axes[2,0].set_title('Evolución de Entropía de von Neumann')
        axes[2,0].set_xlabel('Época')
        axes[2,0].set_ylabel('S_vN')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Plot 6: Correlación L vs Val Loss
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.historial['epoca'])))
        scatter = axes[2,1].scatter(self.historial['L_promedio'], self.historial['loss_val'], 
                                  c=self.historial['epoca'], cmap='viridis', s=60, alpha=0.8)
        axes[2,1].set_xlabel('L Promedio')
        axes[2,1].set_ylabel('Val Loss')
        axes[2,1].set_title('Correlación L vs Val Loss (color = época)')
        axes[2,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2,1], label='Época')
        
        plt.tight_layout()
        plt.savefig('/workspace/sovereignty_completo_results.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Función principal"""
    try:
        # Crear directorio de datos si no existe
        os.makedirs('data', exist_ok=True)
        
        # Ejecutar experimento
        experimento = ExperimentoCompleto(num_epochs=25)
        colapso_epoca, overfitting_epoca = experimento.ejecutar_experimento()
        
        # Conclusión final
        print("\n" + "="*70)
        print("🏁 CONCLUSIÓN FINAL")
        print("="*70)
        
        if colapso_epoca is not None and overfitting_epoca is not None:
            diferencia = overfitting_epoca - colapso_epoca
            if diferencia > 0:
                print(f"🎉 ÉXITO CONFIRMADO:")
                print(f"   El Sovereignty Monitor predijo el colapso {diferencia} épocas")
                print(f"   antes de que fuera visible en val_loss")
                print(f"   Esto valida la hipótesis de RESMA sobre detección temprana")
            else:
                print(f"⚠️ RESULTADO MIXTO:")
                print(f"   L detectó el colapso pero no significativamente antes")
        elif colapso_epoca is not None:
            print(f"🔮 L detectó el colapso en época {colapso_epoca}")
            print(f"   pero no se observó overfitting claro en val_loss")
        else:
            print(f"🤔 RESULTADO:")
            print(f"   No se detectó colapso en L durante el entrenamiento")
            print(f"   El modelo mantuvo capacidad de generalización")
        
        print(f"\n💡 Para confirmar completamente los resultados de RESMA:")
        print(f"   - Probar con datasets más grandes y complejos")
        print(f"   - Entrenar por más épocas para inducir overfitting")
        print(f"   - Calibrar los umbrales (0.5 y 1.0) para el caso específico")
        
    except Exception as e:
        print(f"❌ ERROR EN EL EXPERIMENTO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
