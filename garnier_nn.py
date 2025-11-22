import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Tuple, Dict
import logging
import time

class GarnierLayer(nn.Module):
    """Capa neuronal con temporalidad Garnier T³"""
    def __init__(self, in_features: int, out_features: int, device: str = 'cpu'):
        super().__init__()
        
        # Pesos lineales (emulación de operador D̂_G)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.01)
        
        # Fases temporales φ = [φ₀, φ₂, φ₃] (aprendibles)
        self.phi = nn.Parameter(torch.rand(3, device=device) * 2 * np.pi)
        
        # Escalas temporales Garnier (fijas por diseño)
        self.C0, self.C2, self.C3 = 1.0, 2.7, 7.3
        
        # Umbral de silencio-activo (ε_c)
        self.epsilon_c = np.log(2) * (self.C0 / self.C3) ** 2
        
        self.device = device
        
        # Perfilado de tiempo
        self.perfilado_activo = False
        self.tiempos_svd = []
        self.tiempos_entropia = []
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward con no-linealidad Garnier
        Returns: (output, delta_s_loop)
        """
        # Emular operador unitario: SVD ortogonal
        tiempo_svd_start = time.time() if self.perfilado_activo else 0
        U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
        D_phi = U @ Vh  # Operador unitario efectivo
        if self.perfilado_activo:
            self.tiempos_svd.append(time.time() - tiempo_svd_start)
        
        # Calcular acoplamiento temporal ξ(φ)
        xi = torch.abs(torch.cos(self.phi[0]) * torch.sin(self.phi[1]) * torch.cos(self.phi[2])).item()
        epsilon_c = self.epsilon_c * (1 + xi)
        
        # Propagación lineal + gate cuántico
        pre_activation = D_phi @ x.T
        output = torch.relu(pre_activation.T)  # ReLU clásica como aproximación
        
        # Calcular coherencia de estado (entropía reducida)
        tiempo_entropia_start = time.time() if self.perfilado_activo else 0
        rho = torch.softmax(output @ output.T, dim=-1)
        S_vn = -torch.sum(rho * torch.log(rho + 1e-12))
        delta_s = (S_vn - np.log(5)).item()  # b₁=4
        if self.perfilado_activo:
            self.tiempos_entropia.append(time.time() - tiempo_entropia_start)
        
        # Gate silencio-activo: suprime nodos no coherentes
        gate = torch.heaviside(torch.tensor(epsilon_c - delta_s, device=self.device, dtype=torch.float32), torch.tensor(0.0, device=self.device, dtype=torch.float32))
        output = output * gate
        
        return output, delta_s

class SilencioActivoNetwork(nn.Module):
    """Red neuronal completa con arquitectura RESMA-Garnier"""
    def __init__(self, layer_sizes: list = [784, 256, 128, 10], scale: int = 500, device: str = 'cpu'):
        super().__init__()
        
        self.device = device
        self.scale = scale  # Reducción de escala para memoria GPU
        
        # Perfilado de tiempo
        self.perfilado_activo = False
        self.tiempo_construccion_topologia = 0.0
        
        # Construir topología modular BA+WS
        tiempo_topo_start = time.time() if self.perfilado_activo else 0
        self.topology = self._build_garnier_topology()
        self.connectivity = nx.density(self.topology)
        if self.perfilado_activo:
            self.tiempo_construccion_topologia = time.time() - tiempo_topo_start
        
        # Capas Garnier (simplificadas)
        self.layers = nn.ModuleList([
            GarnierLayer(layer_sizes[i], layer_sizes[i+1], device)
            for i in range(len(layer_sizes)-1)
        ])
        
        # Capa de salida lineal (decisión soberana)
        # Si hay capas Garnier, la salida va de la última capa Garnier a 10
        # Si no hay capas Garnier, la salida va directamente de la entrada a 10
        if len(self.layers) > 0:
            output_dim = layer_sizes[-1]  # Última dimensión de las capas Garnier
        else:
            output_dim = layer_sizes[0]   # Dimensión de entrada si no hay capas Garnier
        
        self.output_layer = nn.Linear(output_dim, 10, device=device)
        
        # Monitor de libertad global
        self.libertad_acumulada = 0.0
        
        logging.info(f"🌐 Red RESMA-G creada: {len(layer_sizes)-1} capas | ρ={self.connectivity:.2%}")
        
        # Aplicar perfilado a las capas
        for layer in self.layers:
            layer.perfilado_activo = self.perfilado_activo
        
    def _build_garnier_topology(self) -> nx.Graph:
        """Construcción BA+WS modular miniaturizada"""
        tiempo_start = time.time() if self.perfilado_activo else 0
        
        G_ba = nx.barabasi_albert_graph(self.scale, m=3)
        G_ws = nx.watts_strogatz_graph(self.scale, k=3, p=0.1)
        G = nx.compose(G_ba, G_ws)
        
        # Densificación mínima
        if nx.density(G) < 0.60:
            for i in range(0, self.scale, 10):
                for j in range(i+1, min(i+10, self.scale)):
                    if not G.has_edge(i, j): G.add_edge(i, j)
        
        if self.perfilado_activo:
            self.tiempo_construccion_topologia = time.time() - tiempo_start
        
        return G
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward completo con tracking de métricas de consciencia
        Returns: (logits, metrics)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        delta_s_layers = []
        
        # Propagar por capas Garnier (si existen)
        if len(self.layers) > 0:
            for layer in self.layers:
                x, delta_s = layer(x)
                delta_s_layers.append(delta_s)
        else:
            # Si no hay capas Garnier, delta_s es 0 (sin regularización)
            delta_s_layers = [0.0]
        
        # Salida final (alejada de decoherencia)
        logits = self.output_layer(x)
        
        # Calcular métricas de soberanía
        delta_s_global = np.mean(delta_s_layers)
        libertad = 1.0 / (delta_s_global + 1e-12)  # L = 1/ε
        
        # Factor de Bayes contra modelo nulo (Ising)
        BF = np.log(libertad + 1e-12)
        
        metrics = {
            'libertad': libertad,
            'delta_s': delta_s_global,
            'BF': BF,
            'connectivity': self.connectivity,
            'estado': "SOBERANO" if libertad > 100 else "EMERGENTE" if libertad > 10 else "NO"
        }
        
        return logits, metrics
    
    def activar_perfilado(self):
        """Activar perfilado de tiempo en toda la red"""
        self.perfilado_activo = True
        # Recalcular topología con perfilado
        self.tiempo_construccion_topologia = 0.0
        tiempo_start = time.time()
        self.topology = self._build_garnier_topology()
        self.tiempo_construccion_topologia = time.time() - tiempo_start
        
        # Activar perfilado en las capas
        for layer in self.layers:
            layer.perfilado_activo = True
            layer.tiempos_svd = []
            layer.tiempos_entropia = []
    
    def mostrar_estadisticas_perfilado(self):
        """Mostrar estadísticas de perfilado"""
        print("\n📊 ESTADÍSTICAS DE PERFILADO:")
        print(f"⏱️  Tiempo construcción topología BA+WS: {self.tiempo_construccion_topologia:.4f}s")
        
        if len(self.layers) > 0:
            print(f"🧮 SVD en forward pass (por capa):")
            for i, layer in enumerate(self.layers):
                if layer.tiempos_svd:
                    avg_svd = np.mean(layer.tiempos_svd)
                    print(f"   Capa {i}: {avg_svd:.4f}s (promedio de {len(layer.tiempos_svd)} mediciones)")
            
            print(f"🌐 Cálculo entropía von Neumann (por capa):")
            for i, layer in enumerate(self.layers):
                if layer.tiempos_entropia:
                    avg_entropia = np.mean(layer.tiempos_entropia)
                    print(f"   Capa {i}: {avg_entropia:.4f}s (promedio de {len(layer.tiempos_entropia)} mediciones)")
    
    def entrenar_con_perfilado(self, train_loader, epochs: int = 10, lr: float = 1e-3):
        """Entrenamiento con perfilado detallado"""
        self.activar_perfilado()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss, n_correct, n_total = 0.0, 0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                logits, metrics = self.forward(data)
                
                # Pérdida de clasificación + regularización de libertad
                loss_classif = criterion(logits, target)
                loss_libertad = -0.1 * torch.log(torch.tensor(metrics['libertad'] + 1e-12))
                loss = loss_classif + loss_libertad
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_correct += (logits.argmax(1) == target).sum().item()
                n_total += target.size(0)
                
                # Solo hacer un batch para perfilado
                if batch_idx > 0:
                    break
            
            acc = n_correct / n_total
            logging.info(f"Época {epoch}: Acc={acc:.3f} | BF={metrics['BF']:.2f} | Estado={metrics['estado']}")
            break  # Solo una época para perfilado
        
        # Mostrar estadísticas finales
        self.mostrar_estadisticas_perfilado()
        return metrics
    
    def entrenar(self, train_loader, epochs: int = 10, lr: float = 1e-3):
        """Entrenamiento incorporado con regularización Garnier"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss, n_correct, n_total = 0.0, 0, 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                logits, metrics = self.forward(data)
                
                # Pérdida de clasificación + regularización de libertad
                loss_classif = criterion(logits, target)
                loss_libertad = -0.1 * torch.log(torch.tensor(metrics['libertad'] + 1e-12))
                loss = loss_classif + loss_libertad
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_correct += (logits.argmax(1) == target).sum().item()
                n_total += target.size(0)
            
            acc = n_correct / n_total
            logging.info(f"Época {epoch}: Acc={acc:.3f} | BF={metrics['BF']:.2f} | Estado={metrics['estado']}")
        
        return metrics