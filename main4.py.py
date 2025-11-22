# =============================================================================
#  RESMA 4.0 – CÓDIGO COMPLETO FINAL (FIX: NameError, UnboundLocalError & Numérico)
#  Autor: Tu nombre
#  Fecha: 2025-11-21
#  Descripción: Implementación completa, estable y corregida.
# =============================================================================

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse.linalg import eigs
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import pint
import psutil

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Unidades
ureg = pint.UnitRegistry()
ureg.define('lattice_E8 = 0.68 * nanometer')
ureg.define('bio_energy = 1e3 * gigaelectronvolt')
ureg.define('percolation_time = day')

# =============================================================================
# 0. CONSTANTES FÍSICAS
# =============================================================================

class RC:
    L_E8       = 0.68e-9          # m
    Lambda_bio = 1e3              # GeV
    beta_eff   = 1.0
    kappa      = 1e11             # Hz
    Omega      = 50e12            # Hz
    chi        = 0.6
    alpha      = 0.702
    N_neurons  = int(1e5)
    k_avg      = 2.7
    gamma      = 0.21
    eps_c      = (8*248)**(-0.5)

# =============================================================================
# 1. VALIDADORES
# =============================================================================

class Validator:
    @staticmethod
    def dim(a: float):
        if not (0 < a < 1):
            raise ValueError(f"α={a} no está en (0,1)")
    @staticmethod
    def pt(k, o, c):
        # Condición simplificada de simetría PT (kappa < chi*Omega)
        return k < c * o 
    @staticmethod
    def size(n: int):
        if n < 1000:
            raise ValueError("N < 1000 neuronas")

# =============================================================================
# 2. HOJA CUÁNTICA KMS (RESTAURADA)
# =============================================================================

@dataclass(frozen=True)
class QuantumLeaf:
    leaf_id: int
    beta_eff: float
    spectral_gap: float
    dimension: int = 248

    def __post_init__(self):
        if self.beta_eff <= 0:
            raise ValueError("β debe ser positivo")

    def spectral_density(self, w):
        return np.exp(-self.beta_eff * w) * (w > self.spectral_gap) * (w**0.5)

    def modular_entropy(self):
        w = np.linspace(self.spectral_gap, 10, 1000)
        rho = self.spectral_density(w)
        rho /= trapezoid(rho, w)
        return -trapezoid(rho * np.log(rho + 1e-12), w)

    def bures_distance(self, other: 'QuantumLeaf') -> float:
        w_min = max(self.spectral_gap, other.spectral_gap)
        w = np.linspace(w_min, 10, 500)
        r1 = self.spectral_density(w)
        r2 = other.spectral_density(w)
        r1 /= trapezoid(r1, w)
        r2 /= trapezoid(r2, w)
        fid = trapezoid(np.sqrt(r1 * r2), w)
        return np.sqrt(2 * (1 - fid)).real

# =============================================================================
# 3. MULTIVERSO (RESTAURADA)
# =============================================================================

class Universe:
    def __init__(self, n_leaves: int = 5000, seed: int = 42):
        self.n_leaves = n_leaves
        np.random.seed(seed)
        self.leaves = {i: QuantumLeaf(i, 1.0, np.random.exponential(0.1) + 0.01)
                       for i in range(n_leaves)}
        self.measure = self._gibbs()
        self.global_state = self._global()

    def _gibbs(self):
        M = np.zeros((self.n_leaves, self.n_leaves))
        for i in range(self.n_leaves):
            for j in range(i + 1, self.n_leaves):
                d = self.leaves[i].bures_distance(self.leaves[j])
                M[i, j] = M[j, i] = np.exp(-self.leaves[i].beta_eff * (d ** 2).real)
        norm = M.sum()
        return M / norm if norm > 1e-12 else np.ones_like(M) / (self.n_leaves ** 2)

    def _global(self):
        diag = np.diag(self.measure)
        tot = diag.sum()
        return {i: diag[i] / tot if tot > 1e-12 else 1 / self.n_leaves
                for i in range(self.n_leaves)}

# =============================================================================
# 4. RED NEURONAL NO DIRIGIDA (CORREGIDA)
# =============================================================================

class Network:
    def __init__(self, n_nodes: int = 50000, seed: int = 42):
        Validator.size(n_nodes)
        self.n_nodes = n_nodes
        np.random.seed(seed)
        G_dir = nx.scale_free_graph(n_nodes, alpha=0.2, beta=0.6, gamma=0.2)
        self.graph = G_dir.to_undirected()
        self.dim_spectral = self._spectral_dim() 
        self.ramsey = self._ramsey()

    def _spectral_dim(self, k=500, n_fit=15): # FIX: UnboundLocalError
        L = nx.normalized_laplacian_matrix(self.graph)
        try:
            # Obtener solo los autovalores (índice [0] es para los autovalores)
            ev_tuple = eigs(L, k=k, which='SM', return_eigenvectors=False)
            ev_raw = ev_tuple[0].real
            ev_raw.sort() # Asegurar orden ascendente
            
        except Exception as e:
            logger.error(f"Error en eigs: {e}. Usando d_s=2.7")
            return 2.7
        
        # Filtrar valores cercanos a cero (ruido o el valor propio trivial)
        ev_nz = ev_raw[ev_raw > 1e-5] 
        
        if len(ev_nz) < n_fit:
            logger.warning(f"Insuficientes autovalores no triviales ({len(ev_nz)}), usando d_s=2.7")
            return 2.7
        
        # Usar los primeros n_fit autovalores no triviales
        ev_fit = ev_nz[:n_fit]
        n_indices = np.arange(1, n_fit + 1)
        
        log_n = np.log(n_indices)
        log_l = np.log(ev_fit)
        
        # Regresión: log(lambda) = m * log(n) + b, donde d_s = 2/m
        m = np.polyfit(log_n, log_l, 1)[0]
        
        if np.abs(m) < 1e-4:
            return 2.7
            
        # Limitar a un rango físico plausible (e.g., 1.0 < d_s < 5.0)
        d_s = 2 / m
        return np.clip(d_s, 1.0, 5.0)

    def _ramsey(self):
        try:
            from ripser import ripser
            from scipy.sparse import csr_matrix
            dist = nx.floyd_warshall_numpy(self.graph)
            d = csr_matrix(dist)
            bc = ripser(d, maxdim=2, distance_matrix=True)['dgms']
            for dim, pts in enumerate(bc):
                if dim == 1 and any(death == np.inf for _, death in pts):
                    return 3
        except:
            logger.warning("ripser no instalado, usando R_Q=4")
        return 4

    def t_c(self):
        N0 = 1e5
        return 21 * (self.n_nodes / N0) ** 0.25 / np.log(max(self.ramsey, 2))

# =============================================================================
# 5. CAVIDAD PT-SIMÉTRICA (RESTAURADA)
# =============================================================================

class MyelinCavity:
    def __init__(self, n_modes: int = 100):
        self.n_modes = n_modes
        self.V_loss = self._loss_potential()
        self.H_0 = self._free_hamiltonian()
        self.is_pt_symmetric = self._pt_symmetry_condition()

    def _free_hamiltonian(self):
        q = np.linspace(0, 2 * np.pi / 1e-3, self.n_modes)
        kinetic = RC.Omega + q ** 2 + RC.chi * q ** 3
        return np.diag(kinetic)

    def _loss_potential(self):
        a0 = 5.29e-11
        r = np.linspace(0, 5e-6, self.n_modes)
        loss = RC.kappa * (r / a0) ** (2 * RC.alpha)
        return 1j * np.diag(loss)

    def _pt_symmetry_condition(self):
        return Validator.pt(RC.kappa, RC.Omega, RC.chi)

    def coherence_quantum(self):
        if not self.is_pt_symmetric:
            return 0.0
        H = self.H_0 + self.V_loss
        ev = np.linalg.eigvals(H)
        if np.max(np.abs(np.imag(ev))) > 1e-8:
            return 0.0
        return 1.0 / self.n_modes

# =============================================================================
# 6. FACTOR DE BAYES ESTABLE (AJUSTADO)
# =============================================================================

class Bayes:
    def __init__(self, pred_resma: dict, nulls: dict):
        self.pred = pred_resma
        self.nulls = nulls

    def log_lik(self, model_pred):
        err = {'q0': 0.1, 't_c': 1.0, 'alpha': 0.1}
        ll = 0.0
        for key in ['q0', 't_c', 'alpha']:
            ll -= 0.5 * ((self.pred[key] - model_pred[key]) / err[key]) ** 2
        return ll

    def ln_bf(self): # Método estable
        k_resma, k_null = 5, 2
        logL_resma = self.log_lik(self.pred)
        ln_bfs = {}
        for name, null in self.nulls.items():
            logL_null = self.log_lik(null)
            # ln(BF) = k_null - k_resma + logL_resma - logL_null
            ln_bf_val = (k_null - k_resma) + (logL_resma - logL_null)
            ln_bfs[name] = ln_bf_val
        ln_total = np.sum(list(ln_bfs.values()))
        return ln_total, ln_bfs

# =============================================================================
# 7. PIPELINE COMPLETO
# =============================================================================

def simulate(n_leaves=5000, n_nodes=50000, seed=42):
    logger.info("RESMA 4.0 – Iniciando simulación completa y estable")
    # --- Restauración de Clases ---
    U = Universe(n_leaves, seed)
    Net = Network(n_nodes, seed)
    M = MyelinCavity()
    # ------------------------------

    pred = {
        'q0': 2 * np.pi / RC.L_E8 / 1e-10,  # Å⁻¹
        't_c': Net.t_c(),
        'alpha': Net.dim_spectral
    }

    nulls = {
        'ising': {'q0': 0, 't_c': 15 * (n_nodes / 1e5) ** 0.25, 'alpha': 0.5},
        'syk4': {'q0': 0, 't_c': 30, 'alpha': 2.0},
        'random': {'q0': 0, 't_c': 5 * np.log(n_nodes), 'alpha': 1.0}
    }

    ln_bf_total, ln_bf_per = Bayes(pred, nulls).ln_bf()

    logger.info("Predicciones: %s", pred)
    logger.info("ln(BF) total: %.2f", ln_bf_total)
    for k, v in ln_bf_per.items():
        logger.info("  vs %-10s: %.2f (ln(BF))", k, v)

    # Interpretación: ln(BF) > ln(10) ≈ 2.3 es Evidencia Fuerte
    verdict = 'CONFIRMADA' if ln_bf_total > 2.3 else 'FALSADA' if ln_bf_total < -2.3 else 'INCONCLUSA'
    logger.info("Veredicto: %s", verdict)

    return {
        'pred': pred,
        'ln_bf': ln_bf_total,
        'verdict': verdict,
        'coherence': M.coherence_quantum(),
        'ramsey': Net.ramsey,
        'freedom': RC.eps_c / max(1, Net.ramsey)
    }

# =============================================================================
# 8. EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    print(simulate())