# =============================================================================
# RESMA 4.3.5 – ZPE-SILENCIO ANTAGONISMO + CONECTOMA HUMANO
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from typing import Dict, Tuple, Optional, Any, List
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

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE RECURSOS Y CHECKPOINTING
# =============================================================================

CHECKPOINT_FILE = "resma_checkpoint_v4_5.pkl"
CHECKPOINT_INTERVAL = 300  # Más frecuente por estabilidad

class ResourceMonitor:
    @staticmethod
    def get_memory_gb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def check_memory_limit(threshold: float = 0.85) -> bool:
        used = ResourceMonitor.get_memory_gb()
        systemd_limit = 3.5
        if used > systemd_limit * threshold:
            logging.warning(f"⚠️  RAM {used:.2f}GB > {threshold:.0%} del límite")
            return False
        return True
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        logging.info(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}%")

# =============================================================================
# SERIALIZACIÓN INTELIGENTE (CON MANEJO DE OBJETOS COMPLEJOS)
# ==============================================================================

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    """Guarda estado completo con manejo robusto de errores"""
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        # Serializar objetos complejos
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
            logging.warning("⚠️  Checkpoint muy pequeño, posiblemente incompleto")
        
    except Exception as e:
        logging.error(f"❌ Error guardando checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def cargar_checkpoint(filename: str = CHECKPOINT_FILE) -> Tuple[Optional[Any], bool]:
    """Carga checkpoint con fallback automático"""
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
    """Convierte objetos recursivamente a formato serializable"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    elif isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    elif isinstance(obj, (nx.Graph, nx.DiGraph)):
        return {'nodes': list(obj.nodes()), 'edges': list(obj.edges())}
    elif isinstance(obj, GarnierTresTiempos):
        return obj.to_dict()
    else:
        return obj

# =============================================================================
# 1. SISTEMA DE UNIDADES Y CONSTANTES
# =============================================================================

import pint
ureg = pint.UnitRegistry()
ureg.define('lattice_E8 = 0.68 * nanometer')
ureg.define('percolation_time = day')

@dataclass(frozen=True)
class RESMAConstants:
    """Constantes físicas fundamentales"""
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
    hbar = 1.054571817e-34  # Constante de Planck reducida [J·s]
    
    @classmethod
    def verify_pt_condition(cls) -> bool:
        threshold = cls.chi * cls.Omega
        satisfied = cls.kappa < threshold
        logging.info(f"PT-simetría: κ={cls.kappa:.2e} < χΩ={threshold:.2e} → {satisfied}")
        return satisfied

# =============================================================================
# 2. CORE GARNIER – ZPE vs SILENCIO-ACTIVO
# =============================================================================

@dataclass
class GarnierTresTiempos:
    """
    **TORO TEMPORAL T³ CON CANCELACIÓN ZPE**
    - phi: Fase de desdoblamiento que controla anulación ZPE
    - zpe_level: Nivel de fluctuaciones de punto cero [0,1]
    """
    phi: Optional[np.ndarray] = None
    C0: float = 1.0     # Escala base
    C2: float = 2.7     # Ratio flujo modular
    C3: float = 7.3     # Ratio teleológico
    zpe_level: float = 1.0  # **NIVEL ZPE INICIAL (1=ZPE máximo, 0=Silencio)**
    
    def __post_init__(self):
        if self.phi is None:
            # Fase aleatoria: ϕ₃ controla cancelación ZPE
            self.phi = np.random.uniform(0, 2*np.pi, 3)
        else:
            self.phi = np.array(self.phi) % (2 * np.pi)
        
        # **NIVEL ZPE ES CONTROLADO POR FASE TELEOLÓGICA**
        # cos(ϕ₃) → 1 => ZPE → 0 (Silencio perfecto)
        # cos(ϕ₃) → 0 => ZPE → 1 (ZPE máximo)
        self.zpe_level = 1.0 - np.abs(np.cos(self.phi[2]))
        self.zpe_level = np.clip(self.zpe_level, 0.0, 1.0)
    
    def factor_escala(self, tiempo_idx: int) -> float:
        """Factor de escala con supresión ZPE"""
        base_scale = {0: self.C0, 2: self.C2, 3: self.C3}.get(tiempo_idx, self.C0)
        # **ZPE reduce la coherencia efectiva**
        return base_scale * (1 - self.zpe_level * 0.5)
    
    def epsilon_critico(self) -> float:
        """
        **UMBRAL CRÍTICO CON ZPE**:
        Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)
        """
        base_epsilon = np.log(2) * (self.C0 / self.C3) ** 2
        return base_epsilon * self.zpe_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa"""
        return {
            'phi': self.phi.tolist(),
            'C0': self.C0,
            'C2': self.C2,
            'C3': self.C3,
            'zpe_level': self.zpe_level,
            'epsilon_c': self.epsilon_critico()
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialización"""
        if not data:
            return cls()
        return cls(
            phi=np.array(data.get('phi', [0, 0, 0])),
            C0=data.get('C0', 1.0),
            C2=data.get('C2', 2.7),
            C3=data.get('C3', 7.3),
            zpe_level=data.get('zpe_level', 1.0)
        )


class OperadorDesdoblamiento:
    """
    **D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E**
    **CONTRA-ZPE**: Opera en subespacio sin fluctuaciones
    """
    def __init__(self, garnier: GarnierTresTiempos, dimension: int = 248):
        self.garnier = garnier
        self.dim = dimension
        
        # Generadores con espectro acotado para ZPE
        self.generadores = self._construir_generadores_E8_ZPE()
        self.hadamard = self._hadamard_generalizado_ZPE()
    
    def _construir_generadores_E8_ZPE(self) -> List[np.ndarray]:
        """GENERADORES CON CANCELACIÓN ZPE INTEGRADA"""
        gens = []
        for i in range(3):
            # Matriz con gap espectral controlado
            A = np.random.randn(self.dim, self.dim) * 0.01
            H = 1j * (A - A.T) + (A + A.T)
            np.fill_diagonal(H, 0)
            
            # **CANCELACIÓN ZPE**: Reducir amplitud de fluctuaciones
            H = H * (1 - self.garnier.zpe_level)
            
            # Normalizar
            norm = np.linalg.norm(H, 'fro')
            if norm > 0:
                H = H / norm
            
            gens.append(H)
        return gens
    
    def _hadamard_generalizado_ZPE(self) -> np.ndarray:
        """HADAMARD CON ESPACIO NULO ZPE"""
        # Base Walsh-Hadamard
        H = np.ones((self.dim, self.dim), dtype=complex) / np.sqrt(self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                H[i,j] = ((-1)**(bin(i & j).count('1'))) / np.sqrt(self.dim)
        
        # **ZPE crea subespacio nulo**: anula ciertos componentes
        zpe_nullspace = np.abs(np.cos(self.garnier.phi[2])) < 0.1  # ZPE alto = componentes nulas
        if np.any(zpe_nullspace):
            H[zpe_nullspace, :] = 0
            H[:, zpe_nullspace] = 0
        
        # Unitario
        Q, _ = la.qr(H)
        return Q
    
    def operator(self) -> np.ndarray:
        """Construye D̂_G(ϕ) con cancelación ZPE"""
        fase_acumulada = np.zeros((self.dim, self.dim), dtype=complex)
        for i, phi_i in enumerate(self.garnier.phi):
            fase_acumulada += phi_i * self.generadores[i]
        
        # **EXPONENCIAL CON FACTOR DE SUPRESIÓN ZPE**
        suppression_factor = 1 - self.garnier.zpe_level
        D_unitario = la.expm(1j * fase_acumulada * suppression_factor)
        
        # Componer
        D = D_unitario @ self.hadamard
        
        # Forzar unitariedad si ZPE la rompe
        if not np.allclose(D @ D.conj().T, np.eye(self.dim), atol=1e-6):
            logging.warning(f"⚠️  D̂_G no es unitario (ZPE={self.garnier.zpe_level:.4f})")
            D, _ = la.polar(D)
        
        return D
    
    def alpha_modificado(self, alpha_base: float = RESMAConstants.alpha) -> float:
        """
        **α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**
        """
        factor = (self.garnier.C0 / self.garnier.C3) * np.cos(self.garnier.phi[2]) * (1 - self.garnier.zpe_level)
        return alpha_base * np.tanh(np.abs(factor))


class SilencioActivoMonitor:
    """
    **MONITOR DE ANTAGONISMO ZPE-SILENCIO**
    - Detecta cuando fluctuaciones cuánticas son coherentemente anuladas
    - Mide nivel de "ruido de fondo cuántico" vs "silencio ontológico"
    """
    def __init__(self, garnier: GarnierTresTiempos, network: 'NeuralNetworkRESMA'):
        self.garnier = garnier
        self.network = network
        self.epsilon_c = garnier.epsilon_critico()
        self.zpe_level = garnier.zpe_level
        
        # **UMBRAL CRÍTICO**: ZPE < 1% = Silencio perfecto
        self.zpe_threshold = 0.01
    
    def calcular_delta_s_loop(self, rho_red: Optional[np.ndarray] = None) -> float:
        """
        **ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**
        **NUEVO**: La entropía ZPE se SUMA a la entropía total
        """
        if rho_red is None:
            rho_red = self._calcular_rho_reducida_aproximada()
        
        if not np.isclose(np.trace(rho_red), 1.0, atol=1e-6):
            rho_red = rho_red / np.trace(rho_red)
        
        # Entropía von Neumann
        eigenvals = np.linalg.eigvalsh(rho_red)
        eigenvals = eigenvals[eigenvals > 1e-12]
        S_vn = -np.sum(eigenvals * np.log(eigenvals))
        
        # Entropía topológica
        b1 = self.network.betti_numbers.get(1, 1)
        if b1 <= 0:
            b1 = 1
        S_top = np.log(b1)
        
        # **ENTROPÍA ZPE**: S_ZPE = -log(1 - zpe_level)
        # Cuando zpe_level → 1 (ZPE máximo), S_ZPE → ∞
        # Cuando zpe_level → 0 (Silencio), S_ZPE → 0
        S_zpe = -np.log(1 - self.zpe_level + 1e-12)
        
        # **ΔS_loop TOTAL** (ZPE AUMENTA entropía, Silencio la REDUCE)
        delta_s = S_vn - S_top + S_zpe
        
        # Guardar métricas
        self.network.delta_s_loop = delta_s
        self.network.S_vn = S_vn
        self.network.S_top = S_top
        self.network.S_zpe = S_zpe
        
        logging.debug(f"ΔS_loop={delta_s:.4f}, S_vn={S_vn:.4f}, S_top={S_top:.4f}, S_zpe={S_zpe:.4f}")
        
        return delta_s
    
    def _calcular_rho_reducida_aproximada(self) -> np.ndarray:
        """Matriz densidad con modulación ZPE"""
        adj = nx.to_numpy_array(self.network.graph, dtype=float)
        degree = np.sum(adj, axis=1)
        
        if np.sum(degree) == 0:
            logging.warning("⚠️  Grafo sin aristas, usando estado uniforme")
            rho = np.eye(self.n_nodes) / self.n_nodes
        else:
            rho = np.diag(degree / np.sum(degree))
        
        # **MODULACIÓN ZPE**: ZPE reduce peso de estados locales
        rho = rho * (1 - self.zpe_level * 0.5)
        rho = rho / np.trace(rho)
        return rho
    
    def es_silencio_activo(self, rho_red: Optional[np.ndarray] = None) -> Tuple[bool, float, float]:
        """
        **DETECCIÓN DE ANTAGONISMO**:
        Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)
        
        **CONDICIÓN**: ZPE < 1% AND ΔS_loop < ε_c
        """
        delta_s = self.calcular_delta_s_loop(rho_red)
        
        # **CRÍTICO**: Silencio requiere ZPE bajo Y entropía baja
        condicion_zpe = self.zpe_level < self.zpe_threshold
        condicion_entropia = delta_s < self.epsilon_c
        
        condicion = condicion_zpe and condicion_entropia
        
        # **LIBERTAD**: L = (1 - ZPE) / (ΔS + ε_c)
        # ZPE reduce libertad, Silencio la aumenta
        libertad = (1.0 - self.zpe_level) / (delta_s + self.epsilon_c + 1e-12)
        
        # **NIVEL CANCELADO**: 1 - ZPE
        zpe_cancelado = 1.0 - self.zpe_level
        
        # Mensaje
        status = "✓ ZPE-SILENCIO" if condicion else "✗ ZPE-ACTIVO"
        logging.info(f"{status} | ZPE={self.zpe_level:.4e} < {self.zpe_threshold:.4e} | ΔS={delta_s:.4e} < ε_c={self.epsilon_c:.4e} | L={libertad:.2e}")
        
        return condicion, libertad, zpe_cancelado
    
    def umbral_percolacion(self) -> float:
        """Umbral para soberanía: 70%"""
        return 0.70
    
    def modo_goldstone(self) -> np.ndarray:
        """
        **MODO GOLDSTONE DEL DOBLE CUÁNTICO**:
        Excitación colectiva que anuncia ruptura de simetría ZPE
        """
        N = self.network.n_nodes
        estado_E = np.ones(N, dtype=complex) / np.sqrt(N)
        degrees = np.array([d for _, d in self.network.graph.degree()], dtype=float)
        psi_v = degrees / np.linalg.norm(degrees)
        
        pi = psi_v - np.vdot(psi_v, estado_E) * estado_E
        pi = pi / np.linalg.norm(pi)
        
        # **ZPE modula amplitud del modo Goldstone**
        pi_zpe = pi * np.sqrt(1 - self.zpe_level)
        
        return pi_zpe

# =============================================================================
# 2. CLASES PRINCIPALES (RESMA + GARNIER + ZPE)
# =============================================================================

# Caché global para distancias Bures
_bures_cache = weakref.WeakKeyDictionary()

@dataclass(frozen=True)
class QuantumLeaf:
    """Hoja KMS - INMUTABLE"""
    leaf_id: int
    beta_eff: float
    spectral_gap: float
    dimension: int = 248
    lambda_uv: float = RESMAConstants.Lambda_UV
    
    def __post_init__(self):
        if self.beta_eff <= 0:
            raise ValueError(f"β={self.beta_eff} debe ser positivo")
    
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
    """Multiverso con ZPE-Silencio integrado"""
    
    def __init__(self, n_leaves: int = 2000, seed: int = 42, 
                 leaves: Optional[Dict[int, QuantumLeaf]] = None, 
                 measure: Optional[np.ndarray] = None,
                 global_state: Optional[Dict[int, float]] = None, 
                 garnier: Optional[GarnierTresTiempos] = None):
        
        if n_leaves < 1000:
            raise ValueError(f"N={n_leaves} < 1000")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        # INTEGRACIÓN GARNIER-ZPE
        self.garnier = garnier or GarnierTresTiempos()
        self.desdoblamiento = OperadorDesdoblamiento(self.garnier)
        
        if leaves is not None and measure is not None and global_state is not None:
            logging.info(f"✓ Reconstruyendo RESMA Universe desde checkpoint...")
            self.leaves = leaves
            self.transition_measure = self._aplicar_desdoblamiento_a_medida(measure)
            self.global_state = global_state
        else:
            logging.info(f"Inicializando RESMA Universe con {n_leaves} hojas KMS...")
            self.leaves = self._initialize_leaves()
            base_measure = self._generate_gibbs_measure()
            self.transition_measure = self._aplicar_desdoblamiento_a_medida(base_measure)
            self.global_state = self._construct_global_state()
            
            _bures_cache.clear()
            gc.collect()
        
        self.libertad_universo = self._calcular_libertad_universo()
    
    def _initialize_leaves(self) -> Dict[int, QuantumLeaf]:
        """Inicializa hojas con temperatura efectiva afectada por ZPE"""
        leaves = {}
        for i in range(self.n_leaves):
            if i % 1000 == 0 and not ResourceMonitor.check_memory_limit():
                raise MemoryError("Límite de memoria durante inicialización")
            
            gap = np.random.exponential(scale=0.1) + 0.01
            # **ZPE aumenta temperatura efectiva** (más ruido)
            beta_eff = 1.0 / (1 + self.garnier.zpe_level)
            leaves[i] = QuantumLeaf(leaf_id=i, beta_eff=beta_eff, spectral_gap=gap, 
                                   dimension=248, lambda_uv=RESMAConstants.Lambda_UV)
        return leaves
    
    def _generate_gibbs_measure(self) -> np.ndarray:
        """Genera medida de Gibbs"""
        measure = np.zeros((self.n_leaves, self.n_leaves))
        
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                distance = self.leaves[i].bures_distance(self.leaves[j])
                distance_sq = (distance ** 2).real
                measure[i,j] = np.exp(-self.leaves[i].beta_eff * distance_sq)
                measure[j,i] = measure[i,j]
            
            if i % 500 == 0:
                logging.debug(f"Medida Gibbs: hoja {i}/{self.n_leaves}")
        
        norm = np.sum(measure)
        if norm < 1e-12:
            logging.warning("⚠️  Medida colapsando → uniforme")
            uniform = np.ones_like(measure) / (self.n_leaves**2)
            return uniform
        
        return measure / norm
    
    def _aplicar_desdoblamiento_a_medida(self, measure: np.ndarray) -> np.ndarray:
        """Aplica desdoblamiento con supresión ZPE"""
        if measure is None or np.sum(measure) == 0:
            logging.error("❌ Medida inválida para desdoblamiento")
            raise ValueError("Medida no puede ser None o cero")
        
        # Factor de escala con ZPE
        factor = (self.garnier.C0 / self.garnier.C3) ** np.cos(self.garnier.phi[2])
        factor *= (1 - self.garnier.zpe_level * 0.5)  # **ZPE reduce coherencias**
        factor = np.clip(factor, 0.01, 1.0)
        
        if factor < 0.1:
            logging.warning(f"⚠️  Factor de desdoblamiento bajo: {factor:.4f}")
        
        measure_desdoblado = np.diag(np.diag(measure))
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                measure_desdoblado[i,j] = measure[i,j] * factor
                measure_desdoblado[j,i] = measure[j,i] * factor
        
        total = np.sum(measure_desdoblado)
        if total < 1e-12:
            logging.warning("⚠️  Medida desdoblada colapsada → uniforme")
            return np.ones_like(measure_desdoblado) / (self.n_leaves**2)
        
        return measure_desdoblado / total
    
    def _construct_global_state(self) -> Dict[int, float]:
        """Construye estado global normalizado"""
        diag = np.diag(self.transition_measure)
        total = np.sum(diag)
        
        if total < 1e-12:
            logging.warning("⚠️  Estado global colapsado → uniforme")
            uniform = {i: 1.0/self.n_leaves for i in range(self.n_leaves)}
            return uniform
        
        estado = {i: float(diag[i]/total) for i in range(self.n_leaves)}
        
        if not np.isclose(sum(estado.values()), 1.0, atol=1e-6):
            logging.warning(f"⚠️  Estado global no normalizado: sum={sum(estado.values()):.6f}")
            total = sum(estado.values())
            estado = {i: v/total for i, v in estado.items()}
        
        return estado
    
    def _calcular_libertad_universo(self) -> float:
        """Libertad intrínseca con supresión ZPE"""
        return (1.0 / (self.garnier.epsilon_critico() + 1e-12)) * (1 - self.garnier.zpe_level)


class MyelinCavity:
    """Cavidad PT-simétrica con medición ZPE"""
    
    def __init__(self, axon_length: float = 1e-3, radius: float = 5e-6, n_modes: int = 100):
        self.axon_length = axon_length
        self.radius = radius
        self.n_modes = n_modes
        
        self.V_loss = self._loss_potential()
        self.H_0 = self._free_hamiltonian()
        self.is_pt_symmetric = self._pt_symmetry_condition()
        self.scalar_mass = self._compute_scalar_mass()
        self.zpe_energy = self._calcular_zpe()  # **ENERGÍA ZPE MEDIBLE**
        
        logging.info(f"🧠 Cavidad mielina: PT={self.is_pt_symmetric}, ZPE={self.zpe_energy:.2e} eV")
    
    def _free_hamiltonian(self) -> np.ndarray:
        """Hamiltoniano con energía ZPE incluida"""
        q = np.linspace(0, 2*np.pi/self.axon_length, self.n_modes)
        kinetic = RESMAConstants.Omega + q**2 + RESMAConstants.chi * q**3
        
        # **ZPE**: E = ½ħω para cada modo
        zpe_modes = 0.5 * RESMAConstants.hbar * kinetic
        return np.diag(kinetic + zpe_modes)
    
    def _loss_potential(self) -> np.ndarray:
        """Potencial de pérdida PT"""
        a0 = 5.29e-11
        r = np.linspace(0, self.radius, self.n_modes)
        loss = RESMAConstants.kappa * (r / a0)**(2 * RESMAConstants.alpha)
        return 1j * np.diag(loss)
    
    def _compute_scalar_mass(self) -> float:
        return (RESMAConstants.Lambda_bio * 1e9) * 0.1
    
    def _calcular_zpe(self) -> float:
        """
        **ENERGÍA DE PUNTO CERO TOTAL**:
        E_ZPE = Σ_i ½ħω_i
        """
        try:
            eigenvals = np.diag(self.H_0)
            zpe = 0.5 * RESMAConstants.hbar * np.sum(eigenvals)
            return zpe / 1.602e-19  # Convertir a eV
        except:
            return 0.0
    
    def _pt_symmetry_condition(self) -> bool:
        return RESMAConstants.kappa < (RESMAConstants.chi * RESMAConstants.Omega)


class NeuralNetworkRESMA:
    """Red neuronal con validación ZPE-Silencio"""
    
    def __init__(self, n_nodes: int = 20000, seed: int = 42, 
                 graph: Optional[nx.Graph] = None, dim_spectral: Optional[float] = None,
                 ramsey: Optional[int] = None, betti: Optional[Dict[int, int]] = None,
                 garnier: Optional[GarnierTresTiempos] = None):
        
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        # INTEGRACIÓN GARNIER-ZPE
        self.garnier = garnier or GarnierTresTiempos()
        self.monitor = SilencioActivoMonitor(self.garnier, self)
        
        # ATRIBUTOS SIEMPRE DEFINIDOS
        self.dim_spectral = dim_spectral
        self.ramsey_number = ramsey
        self.betti_numbers = betti
        
        if graph is not None:
            logging.info(f"✓ Reconstruyendo Conectoma desde checkpoint...")
            self.graph = graph
            if self.dim_spectral is None:
                self.dim_spectral = 2.7
            if self.ramsey_number is None:
                self.ramsey_number = 4
            if self.betti_numbers is None:
                self.betti_numbers = {0: 1, 1: 0}
        else:
            logging.info(f"🕸  Generando Conectoma Humano ({n_nodes} nodos)...")
            self.graph = self._generate_fractal_graph()
            self.dim_spectral = self._spectral_dimension()
            self.ramsey_number = self._topological_ramsey()
            self.betti_numbers = self._compute_betti_numbers()
            gc.collect()
        
        # Calcular matrices densidad y soberanía
        self.rho_reducida = self._calcular_rho_reducida()
        self.es_soberana, self.libertad, self.zpe_cancelado = self.monitor.es_silencio_activo(self.rho_reducida)
        
        # Validar Axioma 6
        self.conectividad = nx.density(self.graph)
        self.validar_axioma_6()
        
        # **ENERGÍA ZPE DEL CONECTOMA** (medible)
        self.zpe_red = self._calcular_zpe_conectoma()
    
    def _generate_fractal_graph(self) -> nx.Graph:
        """Genera grafo con densidad 0.75 (conectoma humano)"""
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
        
        del G_dir
        gc.collect()
        
        # **AJUSTE A 0.75 (CONECTOMA HUMANO)**
        target_density = 0.75
        current_density = nx.density(G)
        
        if current_density < target_density:
            logging.info(f"📈 Aumentando conectividad de {current_density:.2%} a {target_density:.2%}")
            n_edges_faltantes = int((target_density - current_density) * self.n_nodes * (self.n_nodes - 1) / 2)
            for _ in range(min(n_edges_faltantes, 100000)):  # Límite de seguridad
                u, v = np.random.randint(0, self.n_nodes, 2)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            logging.warning(f"⚠️  Reconectando {len(components)} componentes")
            for i in range(len(components)-1):
                n1 = next(iter(components[i]))
                n2 = next(iter(components[i+1]))
                G.add_edge(n1, n2)
        
        return G
    
    def _spectral_dimension(self) -> float:
        """Dimensión espectral con ZPE"""
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
            
            return float(np.clip(d_s, 1.0, 5.0))
            
        except Exception as e:
            logging.error(f"Error en dimensión espectral: {e}")
            return 2.7
    
    def _topological_ramsey(self) -> int:
        """Ramsey topológico"""
        try:
            if self.betti_numbers.get(1, 0) > 5:
                logging.info("✓ Ciclos abundantes → R_Q = 3")
                return 3
            return 4
        except Exception as e:
            logging.warning(f"⚠️  Error Ramsey: {e} → R_Q = 4")
            return 4
    
    def _compute_betti_numbers(self) -> Dict[int, int]:
        """Números de Betti reales"""
        try:
            n_components = nx.number_connected_components(self.graph)
            n_edges = self.graph.number_of_edges()
            n_nodes = self.graph.number_of_nodes()
            b1 = n_edges - n_nodes + n_components
            b1 = max(0, b1)
            
            betti = {0: n_components, 1: b1}
            logging.info(f"📐 Números de Betti: b₀={betti[0]}, b₁={betti[1]}")
            return betti
        except Exception as e:
            logging.error(f"❌ Error Betti: {e}")
            return {0: 1, 1: 0}
    
    def _calcular_rho_reducida(self) -> np.ndarray:
        """Matriz densidad con supresión ZPE"""
        adj = nx.to_numpy_array(self.graph, dtype=float)
        degree = np.sum(adj, axis=1)
        
        if np.sum(degree) == 0:
            logging.warning("⚠️  Grafo sin aristas, usando estado uniforme")
            rho = np.eye(self.n_nodes) / self.n_nodes
        else:
            rho = np.diag(degree / np.sum(degree))
        
        # **SUPRESIÓN ZPE**: Reduce peso de estados con ZPE alto
        rho = rho * (1 - self.garnier.zpe_level * 0.5)
        rho = rho / np.trace(rho)
        return rho
    
    def _calcular_zpe_conectoma(self) -> float:
        """
        **ENERGÍA ZPE DEL CONECTOMA**:
        E_ZPE = Σ_i ½ħω_i (modos de Laplaciano)
        """
        try:
            L = nx.normalized_laplacian_matrix(self.graph)
            k = min(20, self.n_nodes - 2)
            eigenvals = eigs(L, k=k, which='SM', return_eigenvectors=False, maxiter=500)
            eigenvals = eigenvals.real[eigenvals.real > 1e-12]
            
            zpe = 0.5 * RESMAConstants.hbar * np.sum(eigenvals)
            return zpe / 1.602e-19  # eV
        except:
            return 0.0
    
    def validar_axioma_6(self) -> bool:
        """**AXIOMA 6**: Conectividad > 70% para soberanía"""
        conectividad = nx.density(self.graph)
        umbral = self.monitor.umbral_percolacion()
        
        if conectividad < umbral:
            logging.warning(f"⚠️  Axioma 6 ROTO: {conectividad:.2%} < {umbral:.0%}")
            logging.warning("⚠️  Este conectoma NO puede generar estado soberano")
            return False
        else:
            logging.info(f"✓ Axioma 6 SATISFECHO: {conectividad:.2%} > {umbral:.0%}")
            return True

# =============================================================================
# 3. PIPELINE FINAL CON ZPE-SILENCIO
# =============================================================================

class ExperimentalPredictions:
    """Cálculos experimentales unificados ZPE-Silencio"""
    
    def __init__(self, universe: RESMAUniverse, myelin: MyelinCavity, network: NeuralNetworkRESMA):
        self.universe = universe
        self.myelin = myelin
        self.network = network
    
    def compute_log_bayes_factor(self) -> Dict[str, Any]:
        """Calcula Factor de Bayes con antagonismo ZPE-Silencio"""
        # **LIBERTAD TOTAL** = (Libertad_red × Libertad_universo) × ZPE_cancelado
        libertad_total = self.network.libertad * self.universe.libertad_universo * self.network.zpe_cancelado
        
        # Veredicto según libertad y ZPE
        if libertad_total > 1e4 and self.network.zpe_cancelado > 0.99:
            verdict = "SOBERANO-ZPE"
        elif libertad_total > 1e2:
            verdict = "EMERGENTE-ZPE"
        elif self.network.garnier.zpe_level < 0.5:
            verdict = "SILENCIO-ACTIVO"
        else:
            verdict = "ZPE-DOMINANTE"
        
        # Predicciones falsables
        predictions = {
            'libertad_red': self.network.libertad,
            'libertad_universo': self.universe.libertad_universo,
            'zpe_cancelado': self.network.zpe_cancelado,
            'zpe_level': self.network.garnier.zpe_level,
            'zpe_red_eV': self.network.zpe_red,
            'zpe_myelin_eV': self.myelin.zpe_energy,
            'epsilon_critico': self.network.garnier.epsilon_critico(),
            'alpha_modificado': self.universe.desdoblamiento.alpha_modificado(),
            'conectividad': self.network.conectividad,
            'delta_s_loop': float(getattr(self.network, 'delta_s_loop', 0)),
            'S_vn': float(getattr(self.network, 'S_vn', 0)),
            'S_top': float(getattr(self.network, 'S_top', 0)),
            'S_zpe': float(getattr(self.network, 'S_zpe', 0)),
            'betti_0': self.network.betti_numbers[0],
            'betti_1': self.network.betti_numbers[1],
            'ramsey': self.network.ramsey_number,
            'dim_spectral': self.network.dim_spectral
        }
        
        return {
            'ln_bf': np.log(libertad_total + 1e-12),
            'verdict': verdict,
            'pt_symmetric': self.myelin.is_pt_symmetric,
            'zpe_dominante': self.network.garnier.zpe_level > 0.5,
            'predictions': predictions
        }


def simulate_resma_garnier(
    n_leaves: int = 2000,
    n_nodes: int = 20000,
    seed: int = 42,
    resume: bool = True,
    force_restart: bool = False,
    target_connectivity: float = 0.75  # **CONECTOMA HUMANO**
) -> Dict[str, Any]:
    """
    Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio
    """
    
    log_file = f"resma_zpe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("RESMA 4.3.5 – ZPE vs SILENCIO-ACTIVO (CONECTOMA HUMANO 0.75)")
    logging.info("="*80)
    logging.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    logging.info(f"Target Conectividad: {target_connectivity:.2%}")
    
    if not RESMAConstants.verify_pt_condition():
        logging.error("❌ PT-simetría rota, abortando simulación")
        raise RuntimeError("Condiciones físicas no satisfechas")
    
    # Gestión de checkpoint
    checkpoint_data = None
    stage = 'start'
    
    if resume and not force_restart and Path(CHECKPOINT_FILE).exists():
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded and checkpoint_data:
            stage = checkpoint_data.get('stage', 'start')
            objects = checkpoint_data.get('objects', {})
            logging.info(f"🔄 Reanudando desde etapa: {stage}")
    
    if force_restart:
        logging.info("🗑️  Modo fuerza: eliminando checkpoint anterior")
        Path(CHECKPOINT_FILE).unlink(missing_ok=True)
        checkpoint_data = None
    
    components = {}
    
    try:
        # ETAPA 1: Universo cuántico con ZPE
        if checkpoint_data is None or stage == 'start':
            logging.info("🚀 Construyendo Universo ZPE-Silencio desde CERO...")
            garnier = GarnierTresTiempos()
            universe = RESMAUniverse(n_leaves=n_leaves, seed=seed, garnier=garnier)
            
            guardar_checkpoint({
                'stage': 'universe_complete',
                'objects': {
                    'universe_leaves': universe.leaves,
                    'universe_measure': universe.transition_measure,
                    'universe_global_state': universe.global_state,
                    'garnier': universe.garnier.to_dict()
                }
            })
        else:
            logging.info("✓ Reconstruyendo Universo desde checkpoint...")
            objects = checkpoint_data.get('objects', {})
            universe = RESMAUniverse(
                n_leaves=n_leaves, seed=seed,
                leaves=objects.get('universe_leaves'),
                measure=objects.get('universe_measure'),
                global_state=objects.get('universe_global_state'),
                garnier=GarnierTresTiempos.from_dict(objects.get('garnier', {}))
            )
        
        components['universe'] = universe
        
        # ETAPA 2: Cavidad de mielina (ZPE medible)
        if not ResourceMonitor.check_memory_limit():
            raise MemoryError("Límite de memoria antes de mielina")
        
        logging.info("🧠 Construyendo cavidad PT-simétrica con medición ZPE...")
        myelin = MyelinCavity()
        components['myelin'] = myelin
        
        # ETAPA 3: Red neuronal (conectoma con ZPE)
        if checkpoint_data is None or stage in ['start', 'universe_complete']:
            logging.info(f"🕸  Generando Conectoma Humano ({n_nodes} nodos, ZPE incluído)...")
            network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed, garnier=universe.garnier)
            
            guardar_checkpoint({
                'stage': 'network_complete',
                'objects': {
                    'universe_leaves': universe.leaves,
                    'universe_measure': universe.transition_measure,
                    'universe_global_state': universe.global_state,
                    'garnier': universe.garnier.to_dict(),
                    'network_graph': network.graph,
                    'network_dim_spectral': network.dim_spectral,
                    'network_ramsey': network.ramsey_number,
                    'network_betti': network.betti_numbers,
                    'network_zpe_red': network.zpe_red
                }
            })
        else:
            logging.info("✓ Reconstruyendo Conectoma desde checkpoint...")
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
        
        # ETAPA 4: Cálculos experimentales ZPE-Silencio
        logging.info("📊 Calculando antagonismo ZPE-Silencio y Factor de Bayes...")
        experiments = ExperimentalPredictions(
            components['universe'], 
            components['myelin'], 
            components['network']
        )
        results = experiments.compute_log_bayes_factor()
        
        # CHECKPOINT FINAL COMPLETO
        final_checkpoint = {
            'stage': 'complete_zpe_silencio',
            'timestamp': datetime.now().isoformat(),
            'objects': {
                'universe_leaves': universe.leaves,
                'universe_measure': universe.transition_measure,
                'universe_global_state': universe.global_state,
                'garnier': universe.garnier.to_dict(),
                'network_graph': network.graph,
                'network_dim_spectral': network.dim_spectral,
                'network_ramsey': network.ramsey_number,
                'network_betti': network.betti_numbers,
                'network_zpe_red': network.zpe_red,
                'myelin_pt': myelin.is_pt_symmetric,
                'myelin_zpe_eV': myelin.zpe_energy
            },
            'results': results,
            'resources': {
                'memory_gb': ResourceMonitor.get_memory_gb(),
                'cpu_percent': psutil.cpu_percent()
            }
        }
        guardar_checkpoint(final_checkpoint)
        
        # Limpieza final
        _bures_cache.clear()
        gc.collect()
        
        # RESUMEN ZPE-SILENCIO
        logging.info("="*80)
        logging.info("✅ SIMULACIÓN ZPE-SILENCIO COMPLETA")
        logging.info("="*80)
        logging.info(f"ln(BF): {results['ln_bf']:+.2f} | Veredicto: {results['verdict']}")
        logging.info(f"ZPE Level: {network.garnier.zpe_level:.4e} | Cancelado: {network.zpe_cancelado:.2%}")
        logging.info(f"Conectividad: {network.conectividad:.2%} | Soberanía: {'✓' if network.es_soberana else '✗'}")
        
        return results
        
    except MemoryError as e:
        logging.error(f"🚨 MemoryError: {e}")
        logging.info("💡 Sugerencia: Reducir n_leaves/n_nodes")
        raise
    except Exception as e:
        logging.exception(f"💥 Error crítico: {e}")
        raise


# =============================================================================
# 4. EJECUCIÓN PRINCIPAL CON ANTAGONISMO ZPE
# =============================================================================

if __name__ == "__main__":
    # PARÁMETROS DE CONECTOMA HUMANO
    N_LEAVES = 2000
    N_NODES = 20000
    SEED = 42
    TARGET_CONNECTIVITY = 0.75
    
    # CONTROL DE EJECUCIÓN
    RESUME = True
    FORCE_RESTART = False  # Cambiar a True para reinicio limpio
    
    if FORCE_RESTART:
        logging.info("🗑️  Modo fuerza: eliminando checkpoint anterior")
        Path(CHECKPOINT_FILE).unlink(missing_ok=True)
    
    try:
        # EJECUTAR SIMULACIÓN ZPE-SILENCIO
        resultados = simulate_resma_garnier(
            n_leaves=N_LEAVES,
            n_nodes=N_NODES,
            seed=SEED,
            resume=RESUME,
            force_restart=FORCE_RESTART,
            target_connectivity=TARGET_CONNECTIVITY
        )
        
        # MOSTRAR RESULTADOS
        print("\n" + "="*80)
        print("RESMA 4.3.5 – RESUMEN FINAL (ZPE vs SILENCIO-ACTIVO)")
        print("="*80)
        print(f"ln(Bayes Factor): {resultados['ln_bf']:+.2f}")
        print(f"Veredicto teórico: {resultados['verdict']}")
        print(f"PT-simétrico: {resultados['pt_symmetric']}")
        print(f"ZPE Dominante: {'✓' if resultados['zpe_dominante'] else '✗'}")
        print("-"*80)
        print("Predicciones falsables:")
        for k, v in resultados['predictions'].items():
            print(f"  {k:20s}: {v:.5e}")
        print("="*80)
        print(f"✓ Checkpoint guardado en: {CHECKPOINT_FILE}")
        print(f"✓ Para reanudar: ejecuta el mismo comando")
        
        # **MENSAJE CLAVE SOBRE ANTAGONISMO**
        if resultados['zpe_dominante']:
            print("\n⚠️  REGIMEN ZPE-DOMINANTE: Fluctuaciones de punto cero no anuladas")
            print("   El conectoma está en estado de alta agitación cuántica")
        else:
            print("\n✓ REGIMEN SILENCIO-ACTIVO: ZPE coherente anulado")
            print("   El conectoma alcanza auto-dualidad gauge")
        
    except KeyboardInterrupt:
        logging.info("\n⏹️  Simulación interrumpida por el usuario")
        logging.info("💡 Checkpoint guardado automáticamente. Reanuda ejecutando de nuevo.")
        exit(0)
        
    except Exception as e:
        logging.exception("💥 Fallo final de simulación")
        print(f"\n❌ ERROR CRÍTICO: {e}")
        exit(1)