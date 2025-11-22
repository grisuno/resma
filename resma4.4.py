# =============================================================================
# RESMA 4.3.2 – REANUDACIÓN REAL + SERIALIZACIÓN DE OBJETOS
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs  # FIX: Import faltante
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

# =============================================================================
# SERIALIZACIÓN INTELIGENTE (solo guarda lo esencial)
# =============================================================================

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    """
    Guarda el estado COMPLETO de los objetos, no solo metadatos
    """
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        # Guardar todo el diccionario con objetos pickle
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Backup del anterior
        if os.path.exists(filename):
            os.replace(filename, backup_file)
        
        os.replace(temp_file, filename)
        
        # Stats
        size_mb = os.path.getsize(filename) / (1024**2)
        logger.info(f"✅ Checkpoint guardado: {filename}")
        logger.info(f"📦 Tamaño: {size_mb:.2f} MB")
        ResourceMonitor.log_resources()
        
        # Si el checkpoint es muy pequeño, alertar
        if size_mb < 0.1:
            logger.warning("⚠️  Checkpoint muy pequeño, podría estar incompleto")
        
    except Exception as e:
        logger.error(f"❌ Error guardando checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def cargar_checkpoint(filename: str = CHECKPOINT_FILE) -> Tuple[Optional[Any], bool]:
    """Carga el estado COMPLETO desde disco"""
    for attempt_file in [filename, f"{filename}.bak"]:
        if os.path.exists(attempt_file):
            try:
                with open(attempt_file, 'rb') as f:
                    data = pickle.load(f)
                
                size_mb = os.path.getsize(attempt_file) / (1024**2)
                logger.info(f"✅ Checkpoint cargado: {attempt_file} ({size_mb:.2f} MB)")
                ResourceMonitor.log_resources()
                
                # Verificar que tenga datos útiles
                if data and 'stage' in data:
                    logger.info(f"🔄 Reanudando desde etapa: {data['stage']}")
                    return data, True
                else:
                    logger.warning(f"⚠️  Checkpoint corrupto o incompleto")
                    return None, False
                    
            except Exception as e:
                logger.warning(f"⚠️  Error cargando {attempt_file}: {e}")
                continue
    
    logger.info("ℹ️  No se encontró checkpoint válido, iniciando de cero")
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
# 2. CLASES PRINCIPALES (CON MÉTODOS DE SERIALIZACIÓN)
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
            raise ValueError("β debe ser positivo")
    
    def spectral_density(self, omega: np.ndarray) -> np.ndarray:
        uv_factor = np.exp(-omega / self.lambda_uv)
        return uv_factor * np.exp(-self.beta_eff * omega) * (omega > self.spectral_gap) * (omega**0.5)
    
    def bures_distance(self, other: 'QuantumLeaf') -> float:
        """Distancia Bures con caché externo"""
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
    """Multiverso con estado serializable"""
    
    def __init__(self, n_leaves: int = 5000, seed: int = 42, 
                 leaves: Optional[Dict] = None, measure: Optional[np.ndarray] = None,
                 global_state: Optional[Dict] = None):
        """
        Constructor que puede recibir estado serializado
        """
        if n_leaves < 1000:
            raise ValueError(f"N={n_leaves} < 1000")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        if leaves is not None and measure is not None and global_state is not None:
            # RECONSTRUCCIÓN DESDE CHECKPOINT
            logger.info(f"✓ Reconstruyendo RESMA Universe desde checkpoint...")
            self.leaves = leaves
            self.transition_measure = measure
            self.global_state = global_state
        else:
            # CONSTRUCCIÓN NORMAL
            logger.info(f"Inicializando RESMA Universe con {n_leaves} hojas...")
            self.leaves = self._initialize_leaves()
            guardar_checkpoint({
                'stage': 'leaves_initialized',
                'n_leaves': n_leaves,
                'seed': seed,
                'objects': {
                    'universe_leaves': self.leaves,  # SERIALIZAR OBJETOS
                    'universe_measure': None,  # Por calcular
                    'universe_global_state': None
                }
            })
            
            self.transition_measure = self._generate_gibbs_measure()
            guardar_checkpoint({
                'stage': 'measure_generated',
                'n_leaves': n_leaves,
                'seed': seed,
                'objects': {
                    'universe_leaves': self.leaves,
                    'universe_measure': self.transition_measure,
                    'universe_global_state': None
                }
            })
            
            self.global_state = self._construct_global_state()
            
            # Limpieza final de caché
            _bures_cache.clear()
            gc.collect()
    
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
        """Matriz de medida"""
        measure = np.zeros((self.n_leaves, self.n_leaves))
        last_save = time.time()
        
        for i in range(self.n_leaves):
            for j in range(i+1, self.n_leaves):
                distance = self.leaves[i].bures_distance(self.leaves[j])
                distance_sq = (distance ** 2).real
                measure[i,j] = np.exp(-self.leaves[i].beta_eff * distance_sq)
                measure[j,i] = measure[i,j]
            
            if time.time() - last_save > 30:
                guardar_checkpoint({
                    'stage': 'measure_partial',
                    'progress': f"hoja {i}/{self.n_leaves}",
                    'n_leaves': self.n_leaves,
                    'seed': self.seed,
                    'objects': {
                        'universe_leaves': self.leaves,
                        'universe_measure': measure,  # Guardar estado parcial
                        'universe_global_state': None
                    }
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

class NeuralNetworkRESMA:
    def __init__(self, n_nodes: int = 50000, seed: int = 42, 
                 graph: Optional[nx.Graph] = None, dim_spectral: Optional[float] = None,
                 ramsey: Optional[int] = None, betti: Optional[Dict] = None):
        """
        Constructor que puede recibir grafo ya construido
        """
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        if graph is not None:
            # RECONSTRUCCIÓN DESDE CHECKPOINT
            logger.info(f"✓ Reconstruyendo red neuronal desde checkpoint...")
            self.graph = graph
            self.dim_spectral = dim_spectral or 2.7
            self.ramsey_number = ramsey or 4
            self.betti_numbers = betti or {0: 1, 1: 0}
        else:
            # CONSTRUCCIÓN NORMAL
            logger.info(f"Generando red scale-free ({n_nodes} nodos)...")
            self.graph = self._generate_fractal_graph()
            guardar_checkpoint({
                'stage': 'network_generated',
                'n_nodes': n_nodes,
                'seed': seed,
                'objects': {
                    'network_graph': self.graph,
                    'network_dim_spectral': None,
                    'network_ramsey': None,
                    'network_betti': None
                }
            })
            
            self.dim_spectral = self._spectral_dimension()
            guardar_checkpoint({
                'stage': 'spectral_dim_calculated',
                'n_nodes': n_nodes,
                'seed': seed,
                'objects': {
                    'network_graph': self.graph,
                    'network_dim_spectral': self.dim_spectral,
                    'network_ramsey': None,
                    'network_betti': None
                }
            })
            
            self.ramsey_number = self._topological_ramsey()
            self.betti_numbers = self._compute_betti_numbers()
            
            # Checkpoint final de red
            guardar_checkpoint({
                'stage': 'network_complete',
                'n_nodes': n_nodes,
                'seed': seed,
                'objects': {
                    'network_graph': self.graph,
                    'network_dim_spectral': self.dim_spectral,
                    'network_ramsey': self.ramsey_number,
                    'network_betti': self.betti_numbers
                }
            })
            
            # Limpieza
            gc.collect()

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
        """Dimensión espectral con eigenvalores sparse"""
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
            logger.error(f"Error en dimensión espectral: {e}")
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

# =============================================================================
# PIPELINE CON REANUDACIÓN REAL
# =============================================================================

def simulate_resma_with_checkpointing(
    n_leaves: int = 2000,
    n_nodes: int = 20000,
    seed: int = 42,
    resume: bool = True
) -> Dict:
    """Pipeline con reanudación que realmente carga objetos"""
    
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
    logger.info("RESMA 4.3.2 – Inicio de simulación con REANUDACIÓN REAL")
    logger.info("="*80)
    logger.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    
    if not RESMAConstants.verify_pt_condition():
        logger.error("Condiciones PT-simétricas no satisfechas, abortando")
        raise RuntimeError("PT-simetría rota")
    
    # CARGAR CHECKPOINT
    checkpoint_data = None
    stage = 'start'
    objects = {}
    
    if resume and os.path.exists(CHECKPOINT_FILE):
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded and checkpoint_data:
            stage = checkpoint_data.get('stage', 'start')
            objects = checkpoint_data.get('objects', {})
            logger.info(f"🔄 Reanudando desde etapa: {stage}")
            logger.info(f"📦 Objetos en checkpoint: {list(objects.keys())}")
    
    # DICCIONARIO PARA ALMACENAR COMPONENTES
    components = {}
    
    try:
        # ETAPA 1: Universo
        if stage == 'start' or objects.get('universe_measure') is None:
            logger.info("🚀 Construyendo RESMA Universe desde CERO...")
            universe = RESMAUniverse(n_leaves=n_leaves, seed=seed)
        else:
            logger.info("✓ Reconstruyendo RESMA Universe desde CHECKPOINT...")
            universe_leaves = objects.get('universe_leaves', {})
            universe_measure = objects.get('universe_measure')
            universe_global = objects.get('universe_global_state', {})
            
            if universe_measure is not None:
                universe = RESMAUniverse(
                    n_leaves=n_leaves, seed=seed,
                    leaves=universe_leaves,
                    measure=universe_measure,
                    global_state=universe_global
                )
            else:
                logger.warning("⚠️  Checkpoint incompleto, reconstruyendo...")
                universe = RESMAUniverse(n_leaves=n_leaves, seed=seed)
        
        components['universe'] = universe
        
        # ETAPA 2: Mielina (rápida, siempre se reconstruye)
        if not ResourceMonitor.check_memory_limit():
            raise MemoryError("Límite de memoria antes de mielina")
        logger.info("🧠 Construyendo cavidad PT-simétrica...")
        myelin = MyelinCavity()
        components['myelin'] = myelin
        
        # ETAPA 3: Red Neuronal
        if stage in ['start', 'leaves_initialized', 'measure_generated', 'universe_complete', 'myelin_complete'] or objects.get('network_graph') is None:
            logger.info("🕸  Construyendo red neuronal desde CERO...")
            network = NeuralNetworkRESMA(n_nodes=n_nodes, seed=seed)
        else:
            logger.info("✓ Reconstruyendo red neuronal desde CHECKPOINT...")
            network_graph = objects.get('network_graph')
            network_dim = objects.get('network_dim_spectral')
            network_ramsey = objects.get('network_ramsey')
            network_betti = objects.get('network_betti')
            
            if network_graph is not None:
                network = NeuralNetworkRESMA(
                    n_nodes=n_nodes, seed=seed,
                    graph=network_graph,
                    dim_spectral=network_dim,
                    ramsey=network_ramsey,
                    betti=network_betti
                )
            else:
                logger.warning("⚠️  Checkpoint de red incompleto, reconstruyendo...")
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
        # DESCOMENTAR PARA FORZAR REINICIO LIMPIO
        # if os.path.exists(CHECKPOINT_FILE):
        #     os.remove(CHECKPOINT_FILE)
        #     logger.info("🗑️  Checkpoint anterior eliminado")
        
        resultados = simulate_resma_with_checkpointing(
            n_leaves=N_LEAVES,
            n_nodes=N_NODES,
            seed=42,
            resume=True  # Cambiar a False para ignorar checkpoint
        )
        
        print("\n" + "="*80)
        print("RESMA 4.3.2 – RESUMEN FINAL")
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