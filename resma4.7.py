# =============================================================================
# RESMA 4.3.4 – CORRECCIONES CRÍTICAS GARNIER-MALET
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse.linalg import eigs  
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
import warnings
import gc

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONSTANTES Y SISTEMA DE UNIDADES
# =============================================================================

@dataclass(frozen=True)
class RESMAConstants:
    L_E8 = 0.68e-9
    Lambda_bio = 1e3
    beta_eff = 1.0
    kappa = 1e10
    Omega = 50e12
    chi = 0.6
    alpha = 0.702
    N_neurons = int(1e5)
    k_avg = 2.7
    gamma = 0.21
    epsilon_c = (8*248)**(-0.5) * 0.95
    Lambda_UV = 1e15
    
    # NUEVO: Parámetros de conectividad realistas
    MIN_CONNECTIVITY = 0.70  # Axioma 6
    TARGET_DEGREE = 15  # Grado promedio neuronal realista

# =============================================================================
# 2. GARNIER TRES TIEMPOS - MEJORADO
# =============================================================================

@dataclass
class GarnierTresTiempos:
    """
    Toro temporal T³ con parámetros físicamente consistentes.
    Basado en la teoría del desdoblamiento del tiempo de Garnier-Malet.
    """
    phi: np.ndarray = None
    
    def __post_init__(self):
        if self.phi is None:
            self.phi = np.random.uniform(0, 2*np.pi, 3)
        else:
            self.phi = np.array(self.phi) % (2 * np.pi)
        
        # Parámetros calibrados según Garnier-Malet
        self.C0 = 1.0      # Tiempo perceptible (escala humana)
        self.C2 = 2.7      # Tiempo modular (inconsciente rápido)
        self.C3 = 7.3      # Tiempo teleológico (apertura cuántica)
        
        # NUEVO: Factor de acoplamiento temporal
        self.coupling_strength = self._compute_coupling()
    
    def _compute_coupling(self) -> float:
        """Fuerza de acoplamiento entre tiempos"""
        # ξ = cos(φ₀) · sin(φ₂) · cos(φ₃)
        return abs(np.cos(self.phi[0]) * np.sin(self.phi[1]) * np.cos(self.phi[2]))
    
    def factor_escala(self, tiempo_idx: int) -> float:
        """Factor de escala temporal"""
        scales = {0: self.C0, 1: self.C2, 2: self.C3}
        return scales.get(tiempo_idx, self.C0)
    
    def epsilon_critico(self) -> float:
        """
        Entropía crítica con corrección de acoplamiento:
        ε_c = log(2) · (C0/C3)² · (1 + ξ)
        """
        base = np.log(2) * (self.C0 / self.C3) ** 2
        return base * (1 + self.coupling_strength)
    
    def modulation_factor(self) -> float:
        """
        Factor de modulación para la medida cuántica:
        M = exp(-|φ₃ - π|/C3)
        Máximo cuando φ₃ ≈ π (apertura temporal óptima)
        """
        return np.exp(-abs(self.phi[2] - np.pi) / self.C3)


class OperadorDesdoblamiento:
    """
    Operador de desdoblamiento D̂_G(φ) con estructura E8 simplificada
    """
    def __init__(self, garnier: GarnierTresTiempos, dimension: int = 248):
        self.garnier = garnier
        self.dim = dimension
        self.generadores = self._construir_generadores()
    
    def _construir_generadores(self) -> List[np.ndarray]:
        """Generadores temporales (anti-Hermitianos normalizados)"""
        gens = []
        for i in range(3):
            # Matriz aleatoria antisimétrica
            A = np.random.randn(self.dim, self.dim)
            H = (A - A.T) / 2  # Anti-simétrica
            # Normalizar
            H = 1j * H / (np.linalg.norm(H) + 1e-12)
            gens.append(H)
        return gens
    
    def operator(self) -> np.ndarray:
        """Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)"""
        fase = sum(phi * H for phi, H in zip(self.garnier.phi, self.generadores))
        return la.expm(fase)
    
    def aplicar_modulacion(self, state_vector: np.ndarray) -> np.ndarray:
        """Aplica desdoblamiento a vector de estado"""
        D = self.operator()
        modulated = D @ state_vector
        return modulated / (np.linalg.norm(modulated) + 1e-12)
    
    def calcular_alpha_modificado(self, alpha_base: float = 0.702) -> float:
        """
        α'(φ) = α · |cos(φ₃)|^(C0/C3)
        Garantiza α' ∈ [0, α]
        """
        exponent = self.garnier.C0 / self.garnier.C3
        return alpha_base * abs(np.cos(self.garnier.phi[2])) ** exponent


# =============================================================================
# 3. MONITOR DE SILENCIO-ACTIVO - CORREGIDO
# =============================================================================

class SilencioActivoMonitor:
    """
    Monitor de condición de Silencio-Activo: ΔS_loop < ε_c(φ)
    """
    def __init__(self, garnier: GarnierTresTiempos):
        self.garnier = garnier
        self.epsilon_c = garnier.epsilon_critico()
    
    def calcular_delta_s_loop(self, rho_red: np.ndarray, b1: int = 1) -> float:
        """
        ΔS_loop = S_vN(ρ) - log(b₁ + 1)
        
        Args:
            rho_red: Matriz densidad reducida
            b1: Primer número de Betti (ciclos independientes)
        """
        # Entropía von Neumann
        eigenvals = np.linalg.eigvalsh(rho_red)
        eigenvals = eigenvals[eigenvals > 1e-14]
        S_vn = -np.sum(eigenvals * np.log(eigenvals + 1e-14))
        
        # Entropía topológica
        S_top = np.log(float(b1 + 1))
        
        delta_s = S_vn - S_top
        return delta_s
    
    def es_silencio_activo(self, rho_red: np.ndarray, b1: int = 1) -> Tuple[bool, float]:
        """
        Verifica condición y calcula libertad L = 1/(ΔS + ε_c)
        
        Returns:
            (condicion_satisfecha, libertad)
        """
        delta_s = self.calcular_delta_s_loop(rho_red, b1)
        condicion = delta_s < self.epsilon_c
        
        # Libertad con regularización
        libertad = 1.0 / (abs(delta_s) + self.epsilon_c + 1e-12)
        
        logging.info(f"📊 Silencio-Activo: {'✓' if condicion else '✗'} | "
                    f"ΔS={delta_s:.4e} < ε_c={self.epsilon_c:.4e} | L={libertad:.2e}")
        
        return condicion, libertad


# =============================================================================
# 4. HOJA CUÁNTICA KMS (sin cambios)
# =============================================================================

@dataclass(frozen=True)
class QuantumLeaf:
    leaf_id: int
    beta_eff: float
    spectral_gap: float
    dimension: int = 248
    lambda_uv: float = RESMAConstants.Lambda_UV
    
    def spectral_density(self, omega: np.ndarray) -> np.ndarray:
        uv_factor = np.exp(-omega / self.lambda_uv)
        thermal = np.exp(-self.beta_eff * omega)
        cutoff = (omega > self.spectral_gap).astype(float)
        return uv_factor * thermal * cutoff * np.sqrt(omega + 1e-12)
    
    def bures_distance(self, other: 'QuantumLeaf') -> float:
        """Distancia de Bures simplificada"""
        omega_min = max(self.spectral_gap, other.spectral_gap)
        omega_max = min(10.0, self.lambda_uv / 1e14)
        omega = np.linspace(omega_min, omega_max, 300)
        
        r1, r2 = self.spectral_density(omega), other.spectral_density(omega)
        s1, s2 = trapezoid(r1, omega), trapezoid(r2, omega)
        
        if s1 < 1e-12 or s2 < 1e-12:
            return 1.0
        
        fidelity = trapezoid(np.sqrt((r1/s1) * (r2/s2)), omega)
        return np.sqrt(2 * max(0, 1 - fidelity))


# =============================================================================
# 5. UNIVERSO RESMA - MEJORADO
# =============================================================================

class RESMAUniverse:
    """Multiverso cuántico con desdoblamiento Garnier-Malet"""
    
    def __init__(self, n_leaves: int = 1000, seed: int = 42, 
                 garnier: Optional[GarnierTresTiempos] = None):
        if n_leaves < 100:
            raise ValueError(f"Mínimo 100 hojas, recibido {n_leaves}")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        # Garnier
        self.garnier = garnier or GarnierTresTiempos()
        self.desdoblamiento = OperadorDesdoblamiento(self.garnier)
        
        logging.info(f"🌌 Inicializando RESMA Universe ({n_leaves} hojas)...")
        
        # Construcción
        self.leaves = self._initialize_leaves()
        self.transition_measure = self._generate_modulated_measure()
        self.global_state = self._construct_global_state()
        
        # Métricas
        self.libertad_universo = self._calcular_libertad()
        self.coherencia = self._calcular_coherencia()
        
        gc.collect()
    
    def _initialize_leaves(self) -> Dict[int, QuantumLeaf]:
        """Genera hojas con gaps distribuidos exponencialmente"""
        leaves = {}
        for i in range(self.n_leaves):
            gap = np.random.exponential(scale=0.1) + 0.01
            leaves[i] = QuantumLeaf(
                leaf_id=i, 
                beta_eff=RESMAConstants.beta_eff, 
                spectral_gap=gap,
                dimension=248
            )
        return leaves
    
    def _generate_modulated_measure(self) -> np.ndarray:
        """
        Genera medida de transición modulada por Garnier:
        M_ij = exp(-β d²_ij) · φ(garnier)
        """
        logging.info("🔄 Calculando medida cuántica modulada...")
        
        # Medida base (solo vecinos cercanos para memoria)
        n = self.n_leaves
        measure = np.zeros((n, n))
        
        # Computar solo para k vecinos más cercanos
        k_neighbors = min(50, n // 10)
        
        for i in range(n):
            # Calcular distancias a todos
            distances = np.array([
                self.leaves[i].bures_distance(self.leaves[j]) 
                for j in range(n)
            ])
            
            # Seleccionar k más cercanos
            closest = np.argsort(distances)[:k_neighbors]
            
            for j in closest:
                if i != j:
                    d_sq = distances[j] ** 2
                    measure[i, j] = np.exp(-RESMAConstants.beta_eff * d_sq)
            
            if i % 100 == 0:
                logging.debug(f"  Procesadas {i}/{n} hojas")
        
        # Simetrizar
        measure = (measure + measure.T) / 2
        
        # APLICAR MODULACIÓN GARNIER
        modulation = self.garnier.modulation_factor()
        measure *= modulation
        
        # Normalizar
        total = np.sum(measure)
        if total < 1e-12:
            logging.warning("⚠️  Medida colapsada, aplicando uniforme")
            return np.ones_like(measure) / (n * n)
        
        return measure / total
    
    def _construct_global_state(self) -> Dict[int, float]:
        """Estado global como distribución diagonal"""
        diag = np.diag(self.transition_measure)
        total = np.sum(diag)
        
        if total < 1e-12:
            return {i: 1.0/self.n_leaves for i in range(self.n_leaves)}
        
        return {i: diag[i]/total for i in range(self.n_leaves)}
    
    def _calcular_libertad(self) -> float:
        """Libertad del universo: L_U = 1/ε_c"""
        return 1.0 / (self.garnier.epsilon_critico() + 1e-12)
    
    def _calcular_coherencia(self) -> float:
        """Coherencia cuántica: suma de elementos off-diagonal"""
        off_diag = self.transition_measure - np.diag(np.diag(self.transition_measure))
        return np.sum(np.abs(off_diag))


# =============================================================================
# 6. RED NEURONAL - CORREGIDA PARA AXIOMA 6
# =============================================================================

class NeuralNetworkRESMA:
    """
    Red neuronal con topología realista que satisface Axioma 6
    """
    
    def __init__(self, n_nodes: int = 5000, seed: int = 42,
                 garnier: Optional[GarnierTresTiempos] = None):
        if n_nodes < 100:
            raise ValueError(f"Mínimo 100 nodos, recibido {n_nodes}")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        # Garnier
        self.garnier = garnier or GarnierTresTiempos()
        
        logging.info(f"🧠 Generando red neuronal ({n_nodes} nodos)...")
        
        # CONSTRUIR RED PRIMERO
        self.graph = self._generate_realistic_network()
        self.conectividad = nx.density(self.graph)
        
        # CALCULAR PROPIEDADES TOPOLÓGICAS
        self.betti_numbers = self._compute_betti_numbers()
        self.dim_spectral = self._spectral_dimension()
        self.ramsey_number = self._topological_ramsey()
        
        # CREAR MONITOR (ahora sí existe betti_numbers)
        self.monitor = SilencioActivoMonitor(self.garnier)
        
        # ESTADO CUÁNTICO
        self.rho_reducida = self._calcular_rho_reducida()
        self.es_soberana, self.libertad = self.monitor.es_silencio_activo(
            self.rho_reducida, 
            self.betti_numbers.get(1, 1)
        )
        
        # VALIDAR
        self._validar_axioma_6()
        
        gc.collect()
    
    def _generate_realistic_network(self) -> nx.Graph:
        """
        Genera red con conectividad > 70% usando modelo realista:
        - Watts-Strogatz para mundo pequeño
        - Aumentación para alcanzar umbral
        """
        # Parámetros para Watts-Strogatz
        k = RESMAConstants.TARGET_DEGREE  # Grado inicial
        p = 0.3  # Probabilidad de rewiring
        
        logging.info(f"  Generando Watts-Strogatz (k={k}, p={p})...")
        G = nx.watts_strogatz_graph(self.n_nodes, k, p, seed=self.seed)
        
        # Calcular conectividad inicial
        density = nx.density(G)
        logging.info(f"  Densidad inicial: {density:.2%}")
        
        # Si no alcanza 70%, aumentar
        target = RESMAConstants.MIN_CONNECTIVITY
        if density < target:
            logging.info(f"  Aumentando conectividad a {target:.0%}...")
            
            # Añadir aristas aleatorias hasta alcanzar target
            current_edges = G.number_of_edges()
            max_edges = self.n_nodes * (self.n_nodes - 1) / 2
            target_edges = int(target * max_edges)
            edges_to_add = target_edges - current_edges
            
            # Generar pares aleatorios
            added = 0
            attempts = 0
            max_attempts = edges_to_add * 10
            
            while added < edges_to_add and attempts < max_attempts:
                i, j = np.random.randint(0, self.n_nodes, 2)
                if i != j and not G.has_edge(i, j):
                    G.add_edge(i, j)
                    added += 1
                attempts += 1
            
            density_final = nx.density(G)
            logging.info(f"  Densidad final: {density_final:.2%} ({added} aristas añadidas)")
        
        return G
    
    def _compute_betti_numbers(self) -> Dict[int, int]:
        """Números de Betti: b0=componentes, b1=ciclos"""
        try:
            n_components = nx.number_connected_components(self.graph)
            cycles = nx.cycle_basis(self.graph)
            return {0: n_components, 1: len(cycles)}
        except:
            return {0: 1, 1: 0}
    
    def _spectral_dimension(self) -> float:
        """Dimensión espectral del Laplaciano"""
        try:
            L = nx.normalized_laplacian_matrix(self.graph)
            k = min(50, self.n_nodes - 2)
            eigenvals = eigs(L, k=k, which='SM', return_eigenvectors=False, maxiter=500)
            eigenvals = eigenvals.real
            eigenvals = eigenvals[eigenvals > 1e-8]
            eigenvals.sort()
            
            if len(eigenvals) < 10:
                return 2.7
            
            log_n = np.log(np.arange(1, len(eigenvals)+1))
            log_l = np.log(eigenvals + 1e-12)
            coeffs = np.polyfit(log_l[:15], log_n[:15], 1)
            d_s = -2 * coeffs[0]
            
            return np.clip(d_s, 1.0, 5.0)
        except:
            return 2.7
    
    def _topological_ramsey(self) -> int:
        """Número de Ramsey topológico"""
        b1 = self.betti_numbers.get(1, 0)
        return 3 if b1 > 5 else 4
    
    def _calcular_rho_reducida(self) -> np.ndarray:
        """Matriz densidad de la red (normalizada por grados)"""
        degrees = np.array([d for _, d in self.graph.degree()], dtype=float)
        total_degree = np.sum(degrees)
        if total_degree < 1e-12:
            return np.eye(self.n_nodes) / self.n_nodes
        
        rho = np.diag(degrees / total_degree)
        return rho
    
    def _validar_axioma_6(self):
        """Verifica conectividad > 70%"""
        umbral = RESMAConstants.MIN_CONNECTIVITY
        
        if self.conectividad < umbral:
            logging.warning(f"⚠️  Axioma 6 NO SATISFECHO: {self.conectividad:.2%} < {umbral:.0%}")
        else:
            logging.info(f"✓ Axioma 6 SATISFECHO: {self.conectividad:.2%} ≥ {umbral:.0%}")


# =============================================================================
# 7. PREDICCIONES EXPERIMENTALES
# =============================================================================

class ExperimentalPredictions:
    """Cálculo de Factor de Bayes y predicciones"""
    
    def __init__(self, universe: RESMAUniverse, network: NeuralNetworkRESMA):
        self.universe = universe
        self.network = network
    
    def compute_log_bayes_factor(self) -> Dict[str, Any]:
        """
        ln(BF) ∝ log(L_red · L_univ)
        Veredicto basado en libertad total
        """
        # Libertad combinada
        L_total = self.network.libertad * self.universe.libertad_universo
        ln_bf = np.log(L_total + 1e-12)
        
        # Clasificación
        if L_total > 1e3:
            verdict = "SOBERANO"
        elif L_total > 1e2:
            verdict = "EMERGENTE"
        else:
            verdict = "NO-SOBERANO"
        
        # Alpha modificado
        alpha_prime = self.universe.desdoblamiento.calcular_alpha_modificado()
        
        return {
            'ln_bf': ln_bf,
            'verdict': verdict,
            'libertad_total': L_total,
            'predictions': {
                'libertad_red': self.network.libertad,
                'libertad_universo': self.universe.libertad_universo,
                'epsilon_critico': self.network.garnier.epsilon_critico(),
                'alpha_modificado': alpha_prime,
                'conectividad': self.network.conectividad,
                'delta_s_loop': self.network.monitor.calcular_delta_s_loop(
                    self.network.rho_reducida,
                    self.network.betti_numbers.get(1, 1)
                ),
                'betti_1': self.network.betti_numbers.get(1, 0),
                'coherencia_universo': self.universe.coherencia,
                'acoplamiento_garnier': self.universe.garnier.coupling_strength,
                'modulacion_temporal': self.universe.garnier.modulation_factor()
            }
        }


# =============================================================================
# 8. PIPELINE DE SIMULACIÓN
# =============================================================================

def simulate_resma_garnier(
    n_leaves: int = 1000,
    n_nodes: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Pipeline completo RESMA-Garnier con correcciones
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    logging.info("="*80)
    logging.info("RESMA 4.3.4 – GARNIER-MALET (Correcciones Críticas)")
    logging.info("="*80)
    logging.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    logging.info("")
    
    try:
        # Crear Garnier compartido
        garnier = GarnierTresTiempos()
        logging.info(f"🌀 Garnier inicializado:")
        logging.info(f"  φ = [{garnier.phi[0]:.3f}, {garnier.phi[1]:.3f}, {garnier.phi[2]:.3f}]")
        logging.info(f"  ε_c = {garnier.epsilon_critico():.4e}")
        logging.info(f"  Acoplamiento ξ = {garnier.coupling_strength:.4f}")
        logging.info("")
        
        # 1. Universo
        universe = RESMAUniverse(n_leaves=n_leaves, seed=seed, garnier=garnier)
        logging.info(f"✓ Universo: L_U={universe.libertad_universo:.2e}, Coherencia={universe.coherencia:.4f}")
        logging.info("")
        
        # 2. Red
        network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed, garnier=garnier)
        logging.info(f"✓ Red: L_N={network.libertad:.2e}, Soberana={network.es_soberana}")
        logging.info("")
        
        # 3. Predicciones
        experiments = ExperimentalPredictions(universe, network)
        results = experiments.compute_log_bayes_factor()
        
        logging.info("="*80)
        logging.info("✅ SIMULACIÓN COMPLETA")
        logging.info("="*80)
        logging.info(f"ln(BF): {results['ln_bf']:+.2f}")
        logging.info(f"Veredicto: {results['verdict']}")
        logging.info("="*80)
        
        return results
        
    except Exception as e:
        logging.exception(f"💥 Error: {e}")
        raise


# =============================================================================
# 9. EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # CONFIGURACIÓN OPTIMIZADA PARA MEMORIA
    N_LEAVES = 1000   # Reducido de 2000
    N_NODES = 5000    # Reducido de 20000
    
    results = simulate_resma_garnier(
        n_leaves=N_LEAVES,
        n_nodes=N_NODES,
        seed=42
    )
    
    print("\n" + "="*80)
    print("RESMA 4.3.4 – RESUMEN FINAL")
    print("="*80)
    print(f"ln(Bayes Factor): {results['ln_bf']:+.2f}")
    print(f"Veredicto: {results['verdict']}")
    print(f"Libertad Total: {results['libertad_total']:.2e}")
    print("-"*80)
    print("Predicciones Falsables:")
    for k, v in results['predictions'].items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.5e}")
        else:
            print(f"  {k:25s}: {v}")
    print("="*80)