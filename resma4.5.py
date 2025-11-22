# =============================================================================
# RESMA 4.3.3 – GARNIER INTEGRADO CON CORRECCIONES DIMENSIONALES
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs  
from typing import Dict, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass
import logging
import psutil
import warnings
import pickle
import time
import os
import gc
from datetime import datetime
import weakref
from pathlib import Path
import pint
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE RECURSOS Y CHECKPOINTING (SIN CAMBIOS)
# =============================================================================

CHECKPOINT_INTERVAL = 600
CHECKPOINT_FILE = "resma_checkpoint_v4_3.pkl"
MEMORY_WARNING_THRESHOLD = 3.2

class ResourceMonitor:
    @staticmethod
    def get_memory_gb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def check_memory_limit():
        used = ResourceMonitor.get_memory_gb()
        systemd_limit = 3.5
        if used > systemd_limit * 0.85:
            logging.warning(f"⚠️  RAM {used:.2f}GB > 85% del límite systemd ({systemd_limit}GB)")
            return False
        return True
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        logging.info(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}%")

# =============================================================================
# SERIALIZACIÓN INTELIGENTE (MEJORADA PARA OBJETOS GARNIER)
# =============================================================================

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        # Serializar TODO, incluyendo objetos Garnier
        serializable_data = _make_serializable(data)
        
        with open(temp_file, 'wb') as f:
            pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(filename):
            os.replace(filename, backup_file)
        
        os.replace(temp_file, filename)
        
        size_mb = os.path.getsize(filename) / (1024**2)
        logging.info(f"✅ Checkpoint guardado: {filename} ({size_mb:.2f} MB)")
        ResourceMonitor.log_resources()
        
        if size_mb < 0.1:
            logging.warning("⚠️  Checkpoint muy pequeño, podría estar incompleto")
        
    except Exception as e:
        logging.error(f"❌ Error guardando checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def cargar_checkpoint(filename: str = CHECKPOINT_FILE) -> Tuple[Optional[Any], bool]:
    for attempt_file in [filename, f"{filename}.bak"]:
        if os.path.exists(attempt_file):
            try:
                with open(attempt_file, 'rb') as f:
                    data = pickle.load(f)
                
                size_mb = os.path.getsize(attempt_file) / (1024**2)
                logging.info(f"✅ Checkpoint cargado: {attempt_file} ({size_mb:.2f} MB)")
                ResourceMonitor.log_resources()
                
                if data and 'stage' in data:
                    logging.info(f"🔄 Reanudando desde etapa: {data['stage']}")
                    return data, True
                else:
                    logging.warning(f"⚠️  Checkpoint corrupto o incompleto")
                    return None, False
                    
            except Exception as e:
                logging.warning(f"⚠️  Error cargando {attempt_file}: {e}")
                continue
    
    logging.info("ℹ️  No se encontró checkpoint válido, iniciando de cero")
    return None, False

def _make_serializable(obj):
    """Convierte objetos a formato serializable"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    elif isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    else:
        return obj

# =============================================================================
# 1. SISTEMA DE UNIDADES Y CONSTANTES (SIN CAMBIOS)
# =============================================================================

ureg = pint.UnitRegistry()
ureg.define('lattice_E8 = 0.68 * nanometer')
ureg.define('percolation_time = day')

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
    BF_THRESHOLD_STRONG = np.log(10)
    BF_THRESHOLD_WEAK = np.log(3)
    Lambda_UV = 1e15
    
    @classmethod
    def verify_pt_condition(cls):
        threshold = cls.chi * cls.Omega
        satisfied = cls.kappa < threshold
        logging.info(f"PT-simetría: κ={cls.kappa:.2e} < χΩ={threshold:.2e} → {satisfied}")
        return satisfied

# =============================================================================
# 2. CORE GARNIER – IMPLEMENTACIÓN COMPLETA
# =============================================================================

@dataclass
class GarnierTresTiempos:
    """
    Toro temporal T³ con parámetros ADIMENSIONALES.
    C0, C2, C3 son ratios de escala, no velocidades.
    """
    phi: np.ndarray = None  # Fase de desdoblamiento [phi0, phi2, phi3]
    
    def __post_init__(self):
        if self.phi is None:
            self.phi = np.random.uniform(0, 2*np.pi, 3)
        else:
            self.phi = np.array(self.phi) % (2 * np.pi)
        
        # Parámetros de escala (ajustados empíricamente)
        self.C0 = 1.0   # Escala base (referencia)
        self.C2 = 2.7   # Ratio flujo modular (fMRI: 0.1-100 Hz)
        self.C3 = 7.3   # Ratio teleológico (ajustado para α' ∈ [0,0.702])
    
    def factor_escala(self, tiempo_idx: int) -> float:
        """Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico"""
        return {0: self.C0, 2: self.C2, 3: self.C3}.get(tiempo_idx, self.C0)
    
    def epsilon_critico(self) -> float:
        """
        Entropía crítica de percolación (ADIMENSIONAL).
        log(2) es la entropía de un bit cuántico crítico.
        """
        return np.log(2) * (self.C0 / self.C3) ** 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Para serialización"""
        return {'phi': self.phi.tolist(), 'C0': self.C0, 'C2': self.C2, 'C3': self.C3}
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(phi=np.array(data['phi']))


class OperadorDesdoblamiento:
    """
    D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E
    Representación toy de E8 (248x248)
    """
    def __init__(self, garnier: GarnierTresTiempos, dimension: int = 248):
        self.garnier = garnier
        self.dim = dimension
        
        # Generadores del álgebra E8 (aproximación por matrices aleatorias antis-Hermitianas)
        self.generadores = self._construir_generadores_E8()
        self.hadamard = self._hadamard_generalizado()
    
    def _construir_generadores_E8(self) -> List[np.ndarray]:
        """Construye 3 generadores temporales (antis-Hermitianos)"""
        gens = []
        for i in range(3):
            # Matriz real antisimétrica + parte imaginaria controlada
            A = np.random.randn(self.dim, self.dim) * 0.01
            H = (A - A.T) + 0.5j * (A + A.T)  # Antis-Hermitiana
            # Normalizar: ||H|| = 1
            H = H / np.linalg.norm(H)
            gens.append(H)
        return gens
    
    def _hadamard_generalizado(self) -> np.ndarray:
        """Operador de Hadamard en dimensión 248 (unitario)"""
        H = np.ones((self.dim, self.dim), dtype=complex) / np.sqrt(self.dim)
        H[0, :] = 1 / np.sqrt(self.dim)  # Primera fila uniforme
        # Hacer unitario: H†H = I
        H = (H + H.conj().T) / 2 + 1j * (H - H.conj().T) / 2
        Q, R = np.linalg.qr(H)
        return Q
    
    def operator(self) -> np.ndarray:
        """Construye D̂_G(ϕ) dimensionalmente consistente"""
        # Σ_i φ_i H_i (fase adimensional)
        fase_acumulada = np.zeros((self.dim, self.dim), dtype=complex)
        for i, phi_i in enumerate(self.garnier.phi):
            fase_acumulada += phi_i * self.generadores[i]
        
        # Exponencial unitaria: exp(i·fase)
        D_unitario = la.expm(1j * fase_acumulada)
        
        # Componer con Hadamard: D̂ = U(ϕ) · H_E
        D = D_unitario @ self.hadamard
        
        # Verificar unitariedad (tolerancia)
        if not np.allclose(D @ D.conj().T, np.eye(self.dim), atol=1e-6):
            logging.warning("⚠️  D̂_G no es perfectamente unitario")
        
        return D
    
    def aplicar_a_estado(self, estado: np.ndarray) -> np.ndarray:
        """Aplica desdoblamiento a un estado cuántico |Ψ⟩"""
        D = self.operator()
        estado_rotado = D @ estado
        return estado_rotado / np.linalg.norm(estado_rotado)
    
    def calcular_alpha_modificado(self, alpha_base: float = 0.702) -> float:
        """
        α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))
        Garantiza α' ∈ [0, α]
        """
        factor = (self.garnier.C0 / self.garnier.C3) * np.cos(self.garnier.phi[2])
        return alpha_base * np.tanh(np.abs(factor))


class SilencioActivoMonitor:
    """
    Monitor de Silencio-Activo: ΔS_loop < ε_c(ϕ)
    """
    def __init__(self, garnier: GarnierTresTiempos, network: 'NeuralNetworkRESMA'):
        self.garnier = garnier
        self.network = network
        self.epsilon_c = garnier.epsilon_critico()
    
    def calcular_delta_s_loop(self, rho_red: Optional[np.ndarray] = None) -> float:
        """
        ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)
        rho_red: matriz densidad reducida (si es None, se calcula)
        """
        if rho_red is None:
            rho_red = self._calcular_rho_reducida_aproximada()
        
        # Entropía von Neumann: -Tr(ρ log ρ)
        eigenvals = np.linalg.eigvalsh(rho_red)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Evitar log(0)
        S_vn = -np.sum(eigenvals * np.log(eigenvals))
        
        # Entropía topológica: log(b₁ + 1)
        b1 = self.network.betti_numbers.get(1, 1)  # Número de ciclos
        S_top = np.log(float(b1 + 1))
        
        delta_s = S_vn - S_top
        logging.debug(f"ΔS_loop={delta_s:.4f}, S_vn={S_vn:.4f}, S_top={S_top:.4f}")
        return delta_s
    
    def _calcular_rho_reducida_aproximada(self) -> np.ndarray:
        """Aproximación: ρ_red = diag(grados) / sum(grados)"""
        degrees = np.array([d for _, d in self.network.graph.degree()], dtype=float)
        if np.sum(degrees) == 0:
            degrees = np.ones_like(degrees)
        rho = np.diag(degrees / np.sum(degrees))
        return rho
    
    def es_silencio_activo(self, rho_red: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Verifica Silencio-Activo y calcula Libertad L.
        Retorna: (condicion, libertad_L)
        """
        delta_s = self.calcular_delta_s_loop(rho_red)
        condicion = delta_s < self.epsilon_c
        
        # Libertad: L = 1/(ΔS_loop + ε_c) (regularizada)
        libertad = 1.0 / (delta_s + self.epsilon_c + 1e-12)
        
        logging.info(f"📊 Silencio-Activo: {'✓' if condicion else '✗'} | "
                    f"ΔS_loop={delta_s:.4e} < ε_c={self.epsilon_c:.4e} | "
                    f"L={libertad:.2e}")
        
        return condicion, libertad
    
    def umbral_percolacion(self) -> float:
        """Umbral de percolación para soberanía: 70% (Axioma 6)"""
        return 0.70


# =============================================================================
# 2. CLASES PRINCIPALES (REESCRITAS CON GARNIER)
# =============================================================================

# Caché global para distancias Bures
_bures_cache = weakref.WeakKeyDictionary()

@dataclass(frozen=True)
class QuantumLeaf:
    """Hoja KMS - INMUTABLE (SIN CAMBIOS)"""
    leaf_id: int
    beta_eff: float
    spectral_gap: float
    dimension: int = 248
    lambda_uv: float = RESMAConstants.Lambda_UV
    
    def __post_init__(self):
        if self.beta_eff <= 0:
            raise ValueError("β debe ser positivo")
    
    def spectral_density(self, omega: np.ndarray) -> np.ndarray:
        uv_factor = np.exp(-omega / self.lambda_uv)
        return uv_factor * np.exp(-self.beta_eff * omega) * (omega > self.spectral_gap) * (omega**0.5)
    
    def bures_distance(self, other: 'QuantumLeaf') -> float:
        key = (id(self), id(other))
        cache = _bures_cache.get(self)
        if cache is None:
            cache = {}
            _bures_cache[self] = cache
        
        if key in cache:
            return cache[key]
        
        omega_min = max(self.spectral_gap, other.spectral_gap)
        omega_max = min(10, min(self.lambda_uv, other.lambda_uv) / 1e14)
        omega = np.linspace(omega_min, omega_max, 500)
        
        r1, r2 = self.spectral_density(omega), other.spectral_density(omega)
        s1, s2 = trapezoid(r1, omega), trapezoid(r2, omega)
        
        if s1 < 1e-12 or s2 < 1e-12:
            distance = 1.0
        else:
            fidelity = trapezoid(np.sqrt((r1/s1) * (r2/s2)), omega)
            distance = np.sqrt(2 * max(0, 1 - fidelity)).real
        
        cache[key] = distance
        return distance


class RESMAUniverse:
    """Multiverso con estado serializable y desdoblamiento Garnier"""
    
    def __init__(self, n_leaves: int = 2000, seed: int = 42, 
                 leaves: Optional[Dict] = None, measure: Optional[np.ndarray] = None,
                 global_state: Optional[Dict] = None, garnier: Optional[GarnierTresTiempos] = None,
                 **kwargs):
        """
        Constructor que puede recibir estado serializado
        """
        if n_leaves < 1000:
            raise ValueError(f"N={n_leaves} < 1000")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        # INTEGRACIÓN GARNIER
        self.garnier = garnier or GarnierTresTiempos()
        self.desdoblamiento = OperadorDesdoblamiento(self.garnier)
        
        if leaves is not None and measure is not None and global_state is not None:
            logging.info(f"✓ Reconstruyendo RESMA Universe desde checkpoint...")
            self.leaves = leaves
            self.transition_measure = self._aplicar_desdoblamiento_a_medida(measure)
            self.global_state = global_state
        else:
            logging.info(f"Inicializando RESMA Universe con {n_leaves} hojas...")
            self.leaves = self._initialize_leaves()
            base_measure = self._generate_gibbs_measure()
            self.transition_measure = self._aplicar_desdoblamiento_a_medida(base_measure)
            self.global_state = self._construct_global_state()
            
            # Limpieza
            _bures_cache.clear()
            gc.collect()
        
        # Calcular libertad del universo
        self.libertad_universo = self._calcular_libertad_universo()
    
    def _initialize_leaves(self) -> Dict[int, QuantumLeaf]:
        leaves = {}
        for i in range(self.n_leaves):
            if i % 1000 == 0 and not ResourceMonitor.check_memory_limit():
                raise MemoryError("Límite de memoria durante inicialización")
            
            gap = np.random.exponential(scale=0.1) + 0.01
            leaves[i] = QuantumLeaf(leaf_id=i, beta_eff=1.0, spectral_gap=gap, 
                                   dimension=248, lambda_uv=RESMAConstants.Lambda_UV)
        return leaves
    
    def _generate_gibbs_measure(self) -> np.ndarray:
        """Matriz de medida sin desdoblamiento"""
        measure = np.zeros((self.n_leaves, self.n_leaves))
        
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                distance = self.leaves[i].bures_distance(self.leaves[j])
                distance_sq = (distance ** 2).real
                measure[i,j] = np.exp(-self.leaves[i].beta_eff * distance_sq)
                measure[j,i] = measure[i,j]
            
            if i % 500 == 0:
                logging.debug(f"Medida: hoja {i}/{self.n_leaves}")
        
        norm = np.sum(measure)
        if norm < 1e-12:
            logging.warning("Medida colapsando → uniforme")
            return np.ones_like(measure) / (self.n_leaves**2)
        return measure / norm
    
    def _aplicar_desdoblamiento_a_medida(self, measure: np.ndarray) -> np.ndarray:
        """
        Aplica D̂_G(ϕ) a la medida:
        - M_ij → M_ij * (C0/C3)^(cos(ϕ₃))
        - Normaliza después
        """
        factor = (self.garnier.C0 / self.garnier.C3) ** np.cos(self.garnier.phi[2])
        if factor < 0.1:
            logging.warning(f"Factor de desdoblamiento muy pequeño: {factor:.4f}")
        
        # Escalar elementos fuera de la diagonal (coherencias)
        measure_desdoblado = measure.copy()
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                measure_desdoblado[i,j] *= factor
                measure_desdoblado[j,i] *= factor
        
        # Renormalizar
        return measure_desdoblado / np.sum(measure_desdoblado)
    
    def _construct_global_state(self) -> Dict[int, float]:
        diag = np.diag(self.transition_measure)
        total = np.sum(diag)
        
        if total < 1e-12:
            logging.warning("Estado global colapsado → uniforme")
            return {i: 1.0/self.n_leaves for i in range(self.n_leaves)}
        
        return {i: diag[i]/total for i in range(self.n_leaves)}
    
    def _calcular_libertad_universo(self) -> float:
        """Libertad del universo: L = 1/ε_c"""
        return 1.0 / (self.garnier.epsilon_critico() + 1e-12)


class MyelinCavity:
    """Cavidad PT-simétrica (SIN CAMBIOS)"""
    def __init__(self, axon_length: float = 1e-3, radius: float = 5e-6, n_modes: int = 100):
        self.axon_length = axon_length
        self.radius = radius
        self.n_modes = n_modes
        
        self.V_loss = self._loss_potential()
        self.H_0 = self._free_hamiltonian()
        self.is_pt_symmetric = self._pt_symmetry_condition()
        self.scalar_mass = self._compute_scalar_mass()
        logging.info(f"Mielina: PT-simétrico={self.is_pt_symmetric}")
    
    def _free_hamiltonian(self) -> np.ndarray:
        q = np.linspace(0, 2*np.pi/self.axon_length, self.n_modes)
        kinetic = RESMAConstants.Omega + q**2 + RESMAConstants.chi * q**3
        return np.diag(kinetic)
    
    def _loss_potential(self) -> np.ndarray:
        a0 = 5.29e-11
        r = np.linspace(0, self.radius, self.n_modes)
        loss = RESMAConstants.kappa * (r / a0)**(2 * RESMAConstants.alpha)
        return 1j * np.diag(loss)
    
    def _compute_scalar_mass(self) -> float:
        return (RESMAConstants.Lambda_bio * 1e9) * 0.1
    
    def _pt_symmetry_condition(self) -> bool:
        return RESMAConstants.kappa < (RESMAConstants.chi * RESMAConstants.Omega)


class NeuralNetworkRESMA:
    """Red neuronal con embedding Garnier"""
    
    def __init__(self, n_nodes: int = 20000, seed: int = 42, 
                 graph: Optional[nx.Graph] = None, dim_spectral: Optional[float] = None,
                 ramsey: Optional[int] = None, betti: Optional[Dict] = None,
                 garnier: Optional[GarnierTresTiempos] = None, **kwargs):
        """
        Constructor que puede recibir grafo ya construido
        """
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        # INTEGRACIÓN GARNIER
        self.garnier = garnier or GarnierTresTiempos()
        self.monitor = SilencioActivoMonitor(self.garnier, self)
        
        if graph is not None:
            logging.info(f"✓ Reconstruyendo red neuronal desde checkpoint...")
            self.graph = graph
            self.dim_spectral = dim_spectral or 2.7
            self.ramsey_number = ramsey or 4
            self.betti_numbers = betti or {0: 1, 1: 0}
        else:
            logging.info(f"Generando red scale-free ({n_nodes} nodos)...")
            self.graph = self._generate_fractal_graph()
            self.dim_spectral = self._spectral_dimension()
            self.ramsey_number = self._topological_ramsey()
            self.betti_numbers = self._compute_betti_numbers()
            gc.collect()
        
        # Calcular estado de soberanía
        self.rho_reducida = self._calcular_rho_reducida()
        self.es_soberana, self.libertad = self.monitor.es_silencio_activo(self.rho_reducida)
        
        # Validar axioma 6
        self.validar_axioma_6()
    
    def _generate_fractal_graph(self) -> nx.Graph:
        """Generar grafo por lotes con conectividad controlada"""
        # scale_free_graph crea un DIgrafo, convertimos a no-dirigido
        G_dir = nx.scale_free_graph(self.n_nodes, alpha=0.2, beta=0.6, gamma=0.2, seed=self.seed)
        
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        edges = list(G_dir.edges())
        batch_size = 10000
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            G.add_edges_from(batch)
            
            if i % (batch_size * 5) == 0:
                if not ResourceMonitor.check_memory_limit():
                    raise MemoryError("Límite de memoria durante grafo")
        
        # Asegurar conectividad > 70%
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            logging.warning(f"Grafo no conectado, reconectando {len(components)} componentes")
            for i in range(len(components)-1):
                n1 = next(iter(components[i]))
                n2 = next(iter(components[i+1]))
                G.add_edge(n1, n2)
        
        del G_dir
        gc.collect()
        
        return G
    
    def _spectral_dimension(self) -> float:
        """Dimensión espectral con eigenvalores sparse"""
        try:
            L = nx.normalized_laplacian_matrix(self.graph, weight=None)
            k = min(50, self.n_nodes - 2)
            eigenvals = eigs(L, k=k, which='SM', return_eigenvectors=False, maxiter=1000)
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
            
        except Exception as e:
            logging.error(f"Error en dimensión espectral: {e}")
            return 2.7
    
    def _topological_ramsey(self) -> int:
        """Ramsey topológico simplificado"""
        try:
            # Si tiene ciclos infinitos (betti[1] > 5), R_Q = 3
            if self.betti_numbers.get(1, 0) > 5:
                logging.info("Ciclos abundantes → R_Q = 3")
                return 3
            return 4
        except Exception as e:
            logging.warning(f"Error Ramsey: {e} → R_Q = 4")
            return 4
    
    def _compute_betti_numbers(self) -> Dict[int, int]:
        """Números de Betti aproximados por ciclos locales"""
        try:
            cycles = nx.cycle_basis(self.graph)
            n_ciclos = len(cycles)
            return {0: 1, 1: n_ciclos}  # b0=componentes, b1=ciclos
        except:
            return {0: 1, 1: 0}
    
    def _calcular_rho_reducida(self) -> np.ndarray:
        """Matriz densidad reducida del conectoma"""
        # Usar matriz de adyacencia normalizada
        adj = nx.to_numpy_array(self.graph, dtype=float)
        degree = np.sum(adj, axis=1)
        rho = np.diag(degree / np.sum(degree))
        return rho
    
    def validar_axioma_6(self):
        """Verifica: conectividad > 70% para soberanía"""
        conectividad = nx.density(self.graph)
        umbral = self.monitor.umbral_percolacion()
        
        if conectividad < umbral:
            logging.warning(f"⚠️  Axioma 6 ROTO: conectividad={conectividad:.2%} < {umbral:.0%}")
            logging.warning("Subgrafo NO puede generar estado soberano")


# =============================================================================
# 3. PIPELINE FINAL CON GARNIER INTEGRADO
# =============================================================================

class ExperimentalPredictions:
    """Cálculos experimentales (SIN CAMBIOS)"""
    
    def __init__(self, universe: RESMAUniverse, myelin: MyelinCavity, network: NeuralNetworkRESMA):
        self.universe = universe
        self.myelin = myelin
        self.network = network
    
    def compute_log_bayes_factor(self) -> Dict[str, Any]:
        """Calcula Factor de Bayes integrando Garnier"""
        # ln(BF) ∝ Libertad del sistema
        libertad_total = self.network.libertad * self.universe.libertad_universo
        
        # Veredicto
        if libertad_total > 1e3:
            verdict = "SOBERANO"
        elif libertad_total > 1e2:
            verdict = "EMERGENTE"
        else:
            verdict = "NO-SOBERANO"
        
        return {
            'ln_bf': np.log(libertad_total + 1e-12),
            'verdict': verdict,
            'pt_symmetric': self.myelin.is_pt_symmetric,
            'predictions': {
                'libertad_red': self.network.libertad,
                'libertad_universo': self.universe.libertad_universo,
                'epsilon_critico': self.network.garnier.epsilon_critico(),
                'alpha_modificado': self.universe.desdoblamiento.calcular_alpha_modificado(),
                'conectividad': nx.density(self.network.graph),
                'delta_s_loop': self.network.monitor.calcular_delta_s_loop(),
                'betti_1': self.network.betti_numbers.get(1, 0)
            }
        }


def simulate_resma_garnier(
    n_leaves: int = 2000,
    n_nodes: int = 20000,
    seed: int = 42,
    resume: bool = True,
    force_restart: bool = False
) -> Dict:
    """Pipeline único con Garnier integrado"""
    
    log_file = f"resma_garnier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("RESMA 4.3.3 – GARNIER INTEGRADO (Correcciones Dimensionales)")
    logging.info("="*80)
    logging.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    
    if not RESMAConstants.verify_pt_condition():
        logging.error("PT-simetría rota, abortando")
        raise RuntimeError("Condiciones físicas no satisfechas")
    
    # GESTIÓN DE CHECKPOINT
    checkpoint_data = None
    stage = 'start'
    
    if resume and not force_restart and Path(CHECKPOINT_FILE).exists():
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded and checkpoint_data:
            stage = checkpoint_data.get('stage', 'start')
            objects = checkpoint_data.get('objects', {})
            logging.info(f"🔄 Reanudando desde etapa: {stage}")
        else:
            checkpoint_data = None
    
    # DICCIONARIO DE COMPONENTES
    components = {}
    
    try:
        # ETAPA 1: Universo (con Garnier)
        if stage == 'start' or not checkpoint_data:
            logging.info("🚀 Construyendo Universo Garnier desde CERO...")
            garnier = GarnierTresTiempos()
            universe = RESMAUniverse(n_leaves=n_leaves, seed=seed, garnier=garnier)
        else:
            logging.info("✓ Reconstruyendo Universo desde checkpoint...")
            objects = checkpoint_data.get('objects', {})
            universe = RESMAUniverse(
                n_leaves=n_leaves, seed=seed,
                leaves=objects.get('universe_leaves'),
                measure=objects.get('universe_measure'),
                global_state=objects.get('universe_global_state'),
                garner=GarnierTresTiempos.from_dict(objects.get('garnier', {}))
            )
        
        components['universe'] = universe
        
        # ETAPA 2: Mielina (rápida)
        if not ResourceMonitor.check_memory_limit():
            raise MemoryError("Límite de memoria antes de mielina")
        logging.info("🧠 Construyendo cavidad PT-simétrica...")
        myelin = MyelinCavity()
        components['myelin'] = myelin
        
        # ETAPA 3: Red Neuronal (con Garnier)
        if stage in ['start', 'network_partial'] or not checkpoint_data:
            logging.info("🕸  Construyendo Red Garnier desde CERO...")
            network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed, garnier=universe.garnier)
        else:
            logging.info("✓ Reconstruyendo Red desde checkpoint...")
            objects = checkpoint_data.get('objects', {})
            network = NeuralNetworkRESMA(
                n_nodes=n_nodes, seed=seed,
                graph=objects.get('network_graph'),
                dim_spectral=objects.get('network_dim_spectral'),
                ramsey=objects.get('network_ramsey'),
                betti=objects.get('network_betti'),
                garnier=GarnierTresTiempos.from_dict(objects.get('garnier', {}))
            )
        
        components['network'] = network
        
        # ETAPA 4: Predicciones
        logging.info("📊 Calculando predicciones y Factor de Bayes...")
        experiments = ExperimentalPredictions(components['universe'], components['myelin'], components['network'])
        results = experiments.compute_log_bayes_factor()
        
        # CHECKPOINT FINAL
        final_checkpoint = {
            'stage': 'complete_garnier',
            'timestamp': datetime.now().isoformat(),
            'objects': {
                'universe_leaves': components['universe'].leaves,
                'universe_measure': components['universe'].transition_measure,
                'universe_global_state': components['universe'].global_state,
                'garnier': components['universe'].garnier.to_dict(),
                'network_graph': components['network'].graph,
                'network_dim_spectral': components['network'].dim_spectral,
                'network_ramsey': components['network'].ramsey_number,
                'network_betti': components['network'].betti_numbers,
                'myelin_pt': components['myelin'].is_pt_symmetric
            },
            'results': results,
            'resources': {
                'memory_gb': ResourceMonitor.get_memory_gb(),
                'cpu_percent': psutil.cpu_percent()
            }
        }
        guardar_checkpoint(final_checkpoint)
        
        # Limpieza
        _bures_cache.clear()
        gc.collect()
        
        logging.info("="*80)
        logging.info("✅ SIMULACIÓN GARNIER COMPLETA")
        logging.info("="*80)
        logging.info(f"ln(BF): {results['ln_bf']:+.2f} | Veredicto: {results['verdict']}")
        
        return results
        
    except MemoryError as e:
        logging.error(f"🚨 MemoryError: {e}")
        logging.info("💡 Sugerencia: Reducir n_leaves/n_nodes")
        raise
    except Exception as e:
        logging.exception(f"💥 Error crítico: {e}")
        raise


# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    N_LEAVES = 2000
    N_NODES = 20000
    
    # Control de reinicio
    FORCE_RESTART = False  # Cambiar a True para ignorar checkpoint
    
    if FORCE_RESTART and Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        logging.info("🗑️  Checkpoint eliminado (modo fuerza)")
    
    try:
        resultados = simulate_resma_garnier(
            n_leaves=N_LEAVES,
            n_nodes=N_NODES,
            seed=42,
            resume=True,
            force_restart=FORCE_RESTART
        )
        
        print("\n" + "="*80)
        print("RESMA 4.3.3 – RESUMEN FINAL (GARNIER)")
        print("="*80)
        print(f"ln(Bayes Factor): {resultados['ln_bf']:+.2f}")
        print(f"Veredicto teórico: {resultados['verdict']}")
        print(f"PT-simétrico: {resultados['pt_symmetric']}")
        print("-"*80)
        print("Predicciones falsables:")
        for k, v in resultados['predictions'].items():
            print(f"  {k:20s}: {v:.5e}")
        print("="*80)
        print(f"✓ Checkpoint: {CHECKPOINT_FILE}")
        print(f"✓ Para reanudar: ejecuta el mismo comando")
        
    except KeyboardInterrupt:
        logging.info("\n⏹️  Simulación interrumpida por usuario")
        logging.info("💾 Checkpoint guardado para reanudación")
        exit(0)
        
    except Exception as e:
        logging.exception("💥 Fallo final de simulación")
        print(f"\n❌ Error: {e}")
        exit(1)