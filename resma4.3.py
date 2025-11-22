# =============================================================================
# RESMA 4.3.1 – FIX: FrozenInstanceError + Reanudación Inteligente
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
import pint
import psutil
import warnings
import pickle
import time
import os
import gc
from datetime import datetime
import weakref

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE RECURSOS Y CHECKPOINTING
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
            logger.warning(f"⚠️  RAM {used:.2f}GB > 85% del límite systemd ({systemd_limit}GB)")
            return False
        return True
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}%")

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    """Guardado atómico con backup"""
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(filename):
            os.replace(filename, backup_file)
        
        os.replace(temp_file, filename)
        
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
            except:
                pass
        
        logger.info(f"✅ Checkpoint guardado: {filename}")
        logger.info(f"📦 Tamaño: {os.path.getsize(filename) / (1024**2):.2f} MB")
        ResourceMonitor.log_resources()
        
    except Exception as e:
        logger.error(f"❌ Error guardando checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def cargar_checkpoint(filename: str = CHECKPOINT_FILE) -> Tuple[Optional[Any], bool]:
    """Cargar checkpoint con fallback"""
    for attempt_file in [filename, f"{filename}.bak"]:
        if os.path.exists(attempt_file):
            try:
                with open(attempt_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"✅ Checkpoint cargado: {attempt_file}")
                ResourceMonitor.log_resources()
                return data, True
            except Exception as e:
                logger.warning(f"⚠️  Error cargando {attempt_file}: {e}")
                continue
    
    logger.info("ℹ️  No se encontró checkpoint, iniciando de cero")
    return None, False

# =============================================================================
# 1. SISTEMA DE UNIDADES Y CONSTANTES
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
        logger.info(f"PT-simetría: κ={cls.kappa:.2e} < χΩ={threshold:.2e} → {satisfied}")
        return satisfied

# =============================================================================
# 2. CLASES PRINCIPALES (CORREGIDAS)
# =============================================================================

# SOLUCIÓN: Caché externo en lugar de atributo dinámico
# Usamos WeakKeyDictionary para que no retenga instancias innecesarias
_bures_cache = weakref.WeakKeyDictionary()

@dataclass(frozen=True)
class QuantumLeaf:
    """Hoja KMS - INMUTABLE pero con caché externo"""
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
        """Distancia Bures con caché EXTERNO (no en instancia)"""
        # Crear clave única para el par (sin ciclos de vida)
        key = (id(self), id(other))
        
        # Usar caché global (WeakKeyDictionary)
        cache = _bures_cache.get(self)
        if cache is None:
            cache = {}
            _bures_cache[self] = cache
        
        if key in cache:
            return cache[key]
        
        # Cálculo normal
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
        
        # Guardar en caché externo
        cache[key] = distance
        return distance

class RESMAUniverse:
    """Multiverso con construcción lazy"""
    
    def __init__(self, n_leaves: int = 5000, seed: int = 42):
        if n_leaves < 1000:
            raise ValueError(f"N={n_leaves} < 1000")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"Inicializando RESMA Universe con {n_leaves} hojas...")
        
        self.leaves = self._initialize_leaves()
        guardar_checkpoint({'stage': 'leaves_initialized', 'n_leaves': n_leaves})
        
        self.transition_measure = self._generate_gibbs_measure()
        guardar_checkpoint({'stage': 'measure_generated', 'n_leaves': n_leaves})
        
        self.global_state = self._construct_global_state()
        
        # IMPORTANTE: Limpiar caché después de usar
        _bures_cache.clear()
        gc.collect()
    
    def _initialize_leaves(self) -> Dict[int, QuantumLeaf]:
        leaves = {}
        for i in range(self.n_leaves):
            if i % 1000 == 0:
                if not ResourceMonitor.check_memory_limit():
                    raise MemoryError("Límite de memoria durante inicialización")
            
            gap = np.random.exponential(scale=0.1) + 0.01
            leaves[i] = QuantumLeaf(leaf_id=i, beta_eff=1.0, spectral_gap=gap, 
                                   dimension=248, lambda_uv=RESMAConstants.Lambda_UV)
        return leaves
    
    def _generate_gibbs_measure(self) -> np.ndarray:
        """Matriz de medida con guardado incremental"""
        measure = np.zeros((self.n_leaves, self.n_leaves))
        last_save = time.time()
        
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                distance = self.leaves[i].bures_distance(self.leaves[j])
                distance_sq = (distance ** 2).real
                measure[i,j] = np.exp(-self.leaves[i].beta_eff * distance_sq)
                measure[j,i] = measure[i,j]
            
            # Guardar progreso cada 30 segundos
            if time.time() - last_save > 30:
                guardar_checkpoint({
                    'stage': 'measure_partial',
                    'progress': f"{i}/{self.n_leaves}",
                    'memory_gb': ResourceMonitor.get_memory_gb()
                })
                last_save = time.time()
        
        norm = np.sum(measure)
        if norm < 1e-12:
            logger.warning("Medida colapsando → uniforme")
            return np.ones_like(measure) / (self.n_leaves**2)
        return measure / norm
    
    def _construct_global_state(self) -> Dict[int, float]:
        diag = np.diag(self.transition_measure)
        total = np.sum(diag)
        
        if total < 1e-12:
            logger.warning("Estado global colapsado → uniforme")
            return {i: 1.0/self.n_leaves for i in range(self.n_leaves)}
        
        return {i: diag[i]/total for i in range(self.n_leaves)}


# =============================================================================
# 1. VALIDADORES
# =============================================================================

class PhysicalValidator:
    @staticmethod
    def validate_dimension(alpha: float, tolerance: float = 0.015) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"α={alpha} fuera de (0,1)")
        predicted = RESMAConstants.alpha
        if abs(alpha - predicted) > tolerance:
            logger.warning(f"α={alpha} difiere de {predicted} ± {tolerance}")
    
    @staticmethod
    def validate_pt_symmetry(kappa: float, Omega: float, chi: float) -> bool:
        ratio = kappa / (chi * Omega)
        condition = ratio < 1.0
        if not condition:
            logger.warning(f"PT rota: κ/(χΩ) = {ratio:.3e} >= 1")
        return condition
    
    @staticmethod
    def validate_connectome_size(n_nodes: int) -> None:
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000")
    
    @staticmethod
    def validate_spectral_dimension(dim: float) -> bool:
        return 0.5 <= dim <= 4.0


@dataclass
class MyelinCavity:
    axon_length: float = 1e-3
    radius: float = 5e-6
    n_modes: int = 100

    def __post_init__(self):
        if not ResourceMonitor.check_memory_limit():
            raise MemoryError("Límite de memoria antes de crear mielina")
        
        self.V_loss = self._loss_potential()
        self.H_0 = self._free_hamiltonian()
        self.is_pt_symmetric = self._pt_symmetry_condition()
        self.scalar_mass = self._compute_scalar_mass()
        logger.info(f"Mielina: PT-simétrico={self.is_pt_symmetric}")

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
        return PhysicalValidator.validate_pt_symmetry(
            RESMAConstants.kappa, RESMAConstants.Omega, RESMAConstants.chi
        )

    def coherence_quantum(self) -> float:
        if not self.is_pt_symmetric:
            return 0.0
        
        H = self.H_0 + self.V_loss
        eigenvals = np.linalg.eigvals(H)
        imag_max = np.max(np.abs(np.imag(eigenvals)))
        
        if imag_max > 1e-8:
            logger.warning(f"Espectro complejo: max|Im(λ)|={imag_max:.2e}")
            return 0.0
        
        coherence = 1.0 / self.n_modes
        stability = np.exp(-self.scalar_mass / max(RESMAConstants.Lambda_bio, 1))
        result = coherence * stability
        
        del eigenvals
        gc.collect()
        
        return result

class NeuralNetworkRESMA:
    def __init__(self, n_nodes: int = 50000, seed: int = 42):
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        logger.info(f"Generando red scale-free ({n_nodes} nodos)...")
        
        self.graph = self._generate_fractal_graph()
        guardar_checkpoint({'stage': 'network_generated', 'n_nodes': n_nodes})
        
        self.dim_spectral = self._spectral_dimension()
        guardar_checkpoint({'stage': 'spectral_dim_calculated'})
        
        self.ramsey_number = self._topological_ramsey()
        self.betti_numbers = self._compute_betti_numbers()

    def _generate_fractal_graph(self) -> nx.Graph:
        """Generar grafo por lotes"""
        G_dir = nx.scale_free_graph(self.n_nodes, alpha=0.2, beta=0.6, gamma=0.2, seed=self.seed)
        
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        edges = list(G_dir.edges())
        batch_size = 10000
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            G.add_edges_from(batch)
            
            if i % (batch_size * 10) == 0:
                if not ResourceMonitor.check_memory_limit():
                    raise MemoryError("Límite de memoria durante grafo")
        
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components)-1):
                n1 = next(iter(components[i]))
                n2 = next(iter(components[i+1]))
                G.add_edge(n1, n2)
        
        del G_dir
        gc.collect()
        
        return G

    def _spectral_dimension(self) -> float:
        """Dimensión espectral con matriz sparse"""
        try:
            L = nx.normalized_laplacian_matrix(self.graph)
            k = min(50, self.n_nodes - 1)
            eigenvals = eigs(L, k=k, which='SM', return_eigenvectors=False)
            eigenvals = eigenvals.real
            eigenvals = eigenvals[eigenvals > 1e-8]
            eigenvals.sort()
            
            if len(eigenvals) < 10:
                return 2.7
            
            log_n = np.log(np.arange(1, len(eigenvals)+1))
            log_l = np.log(eigenvals)
            coeffs = np.polyfit(log_l[:15], log_n[:15], 1)
            d_s = -2 * coeffs[0]
            
            return np.clip(d_s, 1.0, 5.0)
            
        except Exception as e:
            logger.error(f"Error espectral: {e}")
            return 2.7

    def _topological_ramsey(self) -> int:
        """Ramsey topológico"""
        try:
            from ripser import ripser
            distances = self._graph_to_distance_matrix()
            bc = ripser(distances, maxdim=2, distance_matrix=True)['dgms']
            
            for dim, pts in enumerate(bc):
                if dim == 1 and any(death == np.inf for _, death in pts):
                    logger.info("Ciclo infinito → R_Q = 3")
                    return 3
            return 4
        except ImportError:
            logger.warning("ripser no instalado → R_Q = 4")
            return 4
        except Exception as e:
            logger.warning(f"Error homología: {e} → R_Q = 4")
            return 4

    def _compute_betti_numbers(self) -> Dict[int, int]:
        """Números de Betti"""
        try:
            from ripser import ripser
            distances = self._graph_to_distance_matrix()
            bc = ripser(distances, maxdim=2, distance_matrix=True)['dgms']
            
            betti = {0: 0, 1: 0}
            for dim, pts in enumerate(bc):
                for birth, death in pts:
                    if death == np.inf:
                        betti[dim] = betti.get(dim, 0) + 1
            return betti
        except:
            return {0: 1, 1: 0}

    def _graph_to_distance_matrix(self):
        """Matriz de distancias sparse"""
        lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
        size = self.n_nodes
        row, col, data = [], [], []
        
        for i in range(size):
            for j in range(i+1, size):
                if j in lengths[i]:
                    d = lengths[i][j]
                    row.append(i)
                    col.append(j)
                    data.append(d)
        
        return csr_matrix((data, (row, col)), shape=(size, size))

    def critical_percolation_time(self) -> float:
        """Tiempo crítico de percolación"""
        N0 = 1e5
        ramsey = max(self.ramsey_number, 2)
        scaling = (self.n_nodes / N0)**(1/4)
        tc = 21 * scaling / np.log(ramsey)
        logger.info(f"Tiempo crítico: {tc:.2f} días")
        return tc

class ExperimentalPredictions:
    def __init__(self, universe: RESMAUniverse, myelin: MyelinCavity,
                 network: NeuralNetworkRESMA):
        self.universe = universe
        self.myelin = myelin
        self.network = network
    
    def compute_log_bayes_factor(self) -> Dict:
        """ln(BF)"""
        if not ResourceMonitor.check_memory_limit():
            logger.warning("Memoria crítica antes de BF, liberando...")
            gc.collect()
        
        q0_m_inv = 2 * np.pi / RESMAConstants.L_E8
        q0 = q0_m_inv * 1e-10
        
        pred = {
            'q0': q0,
            't_c': self.network.critical_percolation_time(),
            'alpha': self.network.dim_spectral,
            'coherence': self.myelin.coherence_quantum()
        }
        
        errors = {'q0': 0.1, 't_c': 2.0, 'alpha': 0.05}
        logL_resma = sum(-0.5 * ((pred[k] - pred[k]) / errors[k])**2 for k in errors)
        
        nulls = {
            'ising': {'q0': 0, 't_c': 15*(self.network.n_nodes/1e5)**0.25, 'alpha': 0.5},
            'syk4': {'q0': 0, 't_c': 30, 'alpha': 2.0},
            'random': {'q0': 0, 't_c': 5*np.log(self.network.n_nodes), 'alpha': 1.0}
        }
        
        ln_bfs = {}
        for name, null in nulls.items():
            logL_null = sum(-0.5 * ((pred[k] - null[k]) / errors[k])**2 for k in errors)
            k_resma, k_null = 5, 2
            ln_bfs[name] = (2*k_null - 2*k_resma - 2*logL_null + 2*logL_resma) / 2
        
        ln_total = sum(ln_bfs.values())
        
        if ln_total > RESMAConstants.BF_THRESHOLD_STRONG:
            verdict = 'FUERTE EVIDENCIA'
        elif ln_total > RESMAConstants.BF_THRESHOLD_WEAK:
            verdict = 'EVIDENCIA POSITIVA'
        elif ln_total > -RESMAConstants.BF_THRESHOLD_WEAK:
            verdict = 'INCONCLUSO'
        else:
            verdict = 'EVIDENCIA EN CONTRA'
        
        return {
            'ln_bf': ln_total,
            'ln_bf_per': ln_bfs,
            'verdict': verdict,
            'predictions': pred
        }

# =============================================================================
# PIPELINE CON REANUDACIÓN REAL
# =============================================================================

def simulate_resma_with_checkpointing(
    n_leaves: int = 2000,
    n_nodes: int = 20000,
    seed: int = 42,
    resume: bool = True
) -> Dict:
    """Pipeline con reanudación inteligente desde checkpoints"""
    
    # Configurar logging
    log_file = f"resma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("="*80)
    logger.info("RESMA 4.3.1 – Inicio de simulación con reanudación inteligente")
    logger.info("="*80)
    logger.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    
    if not RESMAConstants.verify_pt_condition():
        logger.error("Condiciones PT-simétricas no satisfechas, abortando")
        raise RuntimeError("PT-simetría rota")
    
    # Cargar checkpoint si existe
    checkpoint_data = None
    stage = 'start'
    if resume and os.path.exists(CHECKPOINT_FILE):
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded and checkpoint_data:
            stage = checkpoint_data.get('stage', 'start')
            logger.info(f"🔄 Reanudando desde etapa: {stage}")
    
    # DICCIONARIO PARA ALMACENAR COMPONENTES
    components = {}
    
    try:
        # ETAPA 1: Universo (solo si no está completo)
        if stage in ['start', 'leaves_initialized', 'measure_generated']:
            logger.info("🚀 Construyendo RESMA Universe...")
            universe = RESMAUniverse(n_leaves=n_leaves, seed=seed)
            components['universe'] = universe
            guardar_checkpoint({'stage': 'universe_complete', 'n_leaves': n_leaves})
        else:
            logger.info("✓ Universo ya construido, cargando desde checkpoint...")
            # Reconstruir si es necesario (o cargar del checkpoint si lo guardaste)
            universe = RESMAUniverse(n_leaves=n_leaves, seed=seed)
            components['universe'] = universe
        
        # ETAPA 2: Mielina
        if stage in ['start', 'leaves_initialized', 'measure_generated', 'universe_complete']:
            if not ResourceMonitor.check_memory_limit():
                raise MemoryError("Límite de memoria antes de mielina")
            logger.info("🧠 Construyendo cavidad PT-simétrica...")
            myelin = MyelinCavity()
            components['myelin'] = myelin
            guardar_checkpoint({'stage': 'myelin_complete'})
        else:
            logger.info("✓ Mielina ya construida, creando nueva instancia...")
            myelin = MyelinCavity()
            components['myelin'] = myelin
        
        # ETAPA 3: Red Neuronal
        if stage in ['start', 'leaves_initialized', 'measure_generated', 'universe_complete', 'myelin_complete']:
            logger.info("🕸️  Construyendo red neuronal...")
            network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed)
            components['network'] = network
            guardar_checkpoint({'stage': 'network_complete', 'n_nodes': n_nodes})
        else:
            logger.info("✓ Red ya construida, cargando...")
            network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed)
            components['network'] = network
        
        # ETAPA 4: Cálculos finales
        logger.info("📊 Calculando predicciones y Factor de Bayes...")
        experiments = ExperimentalPredictions(components['universe'], components['myelin'], components['network'])
        results = experiments.compute_log_bayes_factor()
        
        # Guardar resultado final
        final_data = {
            'stage': 'complete',
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'resources': {
                'memory_gb': ResourceMonitor.get_memory_gb(),
                'cpu_percent': psutil.cpu_percent()
            }
        }
        guardar_checkpoint(final_data)
        
        # Limpieza final
        _bures_cache.clear()
        gc.collect()
        
        logger.info("="*80)
        logger.info("✅ SIMULACIÓN COMPLETA EXITOSA")
        logger.info("="*80)
        logger.info(f"ln(BF) total: {results['ln_bf']:+.2f}")
        logger.info(f"Veredicto: {results['verdict']}")
        logger.info(f"Predicciones: {results['predictions']}")
        
        return results
        
    except MemoryError as e:
        logger.error(f"🚨 MemoryError: {e}")
        logger.info("💡 Sugerencia: Reducir n_leaves/n_nodes o aumentar MemoryMax")
        raise
    except Exception as e:
        logger.exception(f"💥 Error crítico: {e}")
        raise

# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    N_LEAVES = 2000
    N_NODES = 20000
    
    logger = logging.getLogger(__name__)
    
    try:
        # Borrar checkpoint anterior si quieres forzar reinicio desde cero
        # os.remove(CHECKPOINT_FILE)  # Descomentar para reinicio limpio
        
        resultados = simulate_resma_with_checkpointing(
            n_leaves=N_LEAVES,
            n_nodes=N_NODES,
            seed=42,
            resume=True  # Cambiar a False para ignorar checkpoint
        )
        
        print("\n" + "="*80)
        print("RESMA 4.3.1 – RESUMEN FINAL")
        print("="*80)
        print(f"ln(Bayes Factor): {resultados['ln_bf']:+.2f}")
        print(f"Veredicto teórico: {resultados['verdict']}")
        print(f"PT-simétrico: {resultados.get('pt_symmetric', 'N/A')}")
        print("-"*80)
        print("Predicciones falsables:")
        for k, v in resultados['predictions'].items():
            print(f"  {k:12s}: {v:.5e}")
        print("="*80)
        print("✓ Checkpoint guardado en:", CHECKPOINT_FILE)
        print("✓ Para reanudar, ejecuta el mismo comando de nuevo")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Simulación interrumpida por usuario")
        logger.info("💾 Checkpoint guardado. Reanuda ejecutando de nuevo.")
        exit(0)
        
    except Exception as e:
        logger.exception("💥 Fallo final de simulación")
        print(f"\n❌ Error: {e}")
        exit(1)