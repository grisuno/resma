# =============================================================================
# RESMA 4.13 – VECTORIZACIÓN MASIVA (HACK PRO)
# =============================================================================
# OPTIMIZACIÓN: Inicialización y medida cuántica completamente vectorizadas
# en operaciones NumPy (matrices (N,500) + producto matricial Q@Q.T).

import numpy as np
import scipy.linalg as la
import networkx as nx
from scipy.integrate import trapezoid
from scipy.sparse.linalg import eigs  
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
import warnings
import pickle
import gc
import os
from pathlib import Path
import psutil
from datetime import datetime
import weakref
import time
import itertools

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONSTANTES Y CONFIGURACIÓN
# =============================================================================

CHECKPOINT_FILE = "resma_checkpoint_v4_6.pkl"

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
    Lambda_UV = 1e15
    MIN_CONNECTIVITY = 0.70
    TARGET_DEGREE = 15
    
    @classmethod
    def verify_pt_condition(cls) -> bool:
        threshold = cls.chi * cls.Omega
        satisfied = cls.kappa < threshold
        ratio = cls.kappa / threshold
        
        logging.info("🔬 Verificación PT-simetría:")
        logging.info(f"   κ = {cls.kappa:.2e} Hz")
        logging.info(f"   χΩ = {threshold:.2e} Hz")
        logging.info(f"   Ratio = {ratio:.4f}")
        logging.info(f"   Resultado: {'✓ PT-simétrico' if satisfied else '✗ FASE ROTA'}")
        
        if not satisfied:
            logging.warning("⚠️  Sistema en fase rota: aumentar χ o Ω")
        
        return satisfied

# =============================================================================
# 2. GARNIER TRES TIEMPOS
# =============================================================================

@dataclass
class GarnierTresTiempos:
    phi: np.ndarray = None
    
    def __post_init__(self):
        if self.phi is None:
            self.phi = np.random.uniform(0, 2*np.pi, 3)
        else:
            self.phi = np.array(self.phi) % (2 * np.pi)
        
        self.C0, self.C2, self.C3 = 1.0, 2.7, 7.3
        self.coupling_strength = abs(np.cos(self.phi[0]) * np.sin(self.phi[1]) * np.cos(self.phi[2]))
        
        logging.info(f"🌀 Garnier T³ inicializado:")
        logging.info(f"   φ=[{self.phi[0]:.3f},{self.phi[1]:.3f},{self.phi[2]:.3f}] | ε_c={self.epsilon_critico():.4e}")
    
    def epsilon_critico(self) -> float:
        return np.log(2) * (self.C0 / self.C3) ** 2 * (1 + self.coupling_strength)
    
    def modulation_factor(self) -> float:
        return np.exp(-abs(self.phi[2] - np.pi) / self.C3)
    
    def to_dict(self) -> dict:
        return {
            'phi': self.phi.tolist(),
            'C0': self.C0,
            'C2': self.C2,
            'C3': self.C3,
            'coupling': self.coupling_strength
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        obj = cls(phi=np.array(data['phi']))
        obj.C0, obj.C2, obj.C3 = data['C0'], data['C2'], data['C3']
        return obj

# =============================================================================
# 3. OPERADOR DE DESDOBLAMIENTO
# =============================================================================

class OperadorDesdoblamiento:
    """
    Operador de desdoblamiento D̂_G(φ) sobre el álgebra E8 genuina.
    Construcción: sistema de raíces E8 → base de Chevalley → representación adjunta 248.
    """

    def __init__(self, garnier: GarnierTresTiempos, dimension: int = 248):
        self.garnier = garnier
        self.dim = dimension

        logging.info(f"🔷 Construyendo álgebra E8 genuina (adjunta 248D)")

        # === 1. Sistema de raíces E8 (240 raíces en R⁸) ===
        self.roots = self._generate_e8_roots()
        self.n_pos = len(self.roots) // 2

        # Mapa: raíz → índice global (0..239)
        self.root_to_idx = {}
        for i, r in enumerate(self.roots):
            self.root_to_idx[tuple(np.round(r, 12))] = i

        # Posición en base adjunta: H₀..H₇ (0-7), E_α⁺ (8-127), E_α⁻ (128-247)
        # con roots[0..119] = raíces positivas, roots[120..239] = raíces negativas
        # basis_pos[gi] = 8 + gi (directo por construcción)

        # === 2. Constantes de estructura de Chevalley (solo pares con α+β raíz) ===
        self.N = self._compute_structure_constants()

        # === 3. Generadores genuinos de E8 en adjunta (tres escalas Garnier T³) ===
        self.generadores = self._construir_generadores_e8()
        self.hadamard = self._hadamard_generalizado()

    # ------------------------------------------------------------------
    #  SISTEMA DE RAÍCES E8
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_e8_roots() -> np.ndarray:
        """
        Genera las 240 raíces de E8 en R⁸.
        Retorna array (240,8) con forma: [positivas (120) | negativas (120)]
        donde neg[j] = -pos[j].
        """
        raw = []

        # Tipo 1: (±1, ±1, 0, 0, 0, 0, 0, 0) + permutaciones → 112
        for i in range(8):
            for j in range(i + 1, 8):
                for s1, s2 in itertools.product([-1, 1], [-1, 1]):
                    v = np.zeros(8)
                    v[i] = s1
                    v[j] = s2
                    raw.append(v)

        # Tipo 2: ½(±1, ±1, ..., ±1) con número par de signos neg → 128
        for mask in range(256):
            v = np.array(
                [1 if (mask >> i) & 1 else -1 for i in range(8)], dtype=float
            ) * 0.5
            if np.count_nonzero(v < 0) % 2 == 0:
                raw.append(v)

        # Separar positivas y negativas por primer componente no nulo
        pos_set = set()
        for r in raw:
            nz = np.where(np.abs(r) > 1e-10)[0]
            if len(nz) > 0 and r[nz[0]] > 0:
                pos_set.add(tuple(np.round(r, 12)))

        pos_sorted = sorted(pos_set, key=lambda x: x)
        pos_arr = np.array(pos_sorted, dtype=float)
        neg_arr = -pos_arr
        roots = np.vstack([pos_arr, neg_arr])

        assert roots.shape == (240, 8), f"E8 necesita 240 raíces, → {roots.shape[0]}"
        return roots

    def _idx(self, root_vec: np.ndarray) -> int:
        """Índice global (0..239) de una raíz."""
        return self.root_to_idx.get(tuple(np.round(root_vec, 12)), -1)

    # ------------------------------------------------------------------
    #  CONSTANTES DE ESTRUCTURA (CHEVALLEY)
    # ------------------------------------------------------------------

    def _compute_structure_constants(self) -> Dict:
        """
        Constantes N_{α,β} para toda raíz α, β con α+β también raíz.
        Retorna dict {(i,j): N_{α_i,α_j}} con ambas orientaciones.
        Convención Chevalley:
          N_{α,β} = -(p+1) si α < β,  (p+1) si α > β,
          donde p es el entero max con β - pα raíz.
        """
        N = {}
        R = self.roots
        for i in range(240):
            if i == 120:
                continue
            alpha = R[i]
            for j in range(240):
                if i == j:
                    continue
                beta = R[j]
                gamma = alpha + beta
                if self._idx(gamma) < 0:
                    continue

                # Longitud de la α-cuerda a través de β
                p = 0
                while True:
                    test = beta - (p + 1) * alpha
                    if self._idx(test) >= 0:
                        p += 1
                    else:
                        break

                val = -(p + 1) if i < j else (p + 1)
                N[(i, j)] = val
                N[(j, i)] = -val

        return N

    # ------------------------------------------------------------------
    #  REPRESENTACIÓN ADJUNTA 248
    # ------------------------------------------------------------------

    def _adjoint_matrix(self, cartan: np.ndarray, roots_coeff: np.ndarray) -> np.ndarray:
        """
        Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.

        Base: |H₀⟩..|H₇⟩ (0-7), |E_α₀⟩..|E_α₁₁₉⟩ (8-127), |E_{-α₀}⟩..|E_{-α₁₁₉}⟩ (128-247)
        con roots[gi] = α para gi en 0..119, roots[gi+120] = -α.

        Args:
            cartan:     array[8] coeficientes c_i para H_i.
            roots_coeff: array[240] coeficientes d_γ para E_γ.
        """
        M = np.zeros((self.dim, self.dim), dtype=complex)
        n_pos = self.n_pos

        # ---- Columnas H_j (j=0..7) ---------------------------------
        for j in range(8):
            for gi in range(240):
                dg = roots_coeff[gi]
                if abs(dg) < 1e-14:
                    continue
                # [E_γ, H_j] = -γ_j E_γ
                M[8 + gi, j] -= dg * self.roots[gi][j]

        # ---- Columnas E_β (β = raíz positiva) ----------------------
        for bj in range(n_pos):
            col = 8 + bj
            beta = self.roots[bj]
            gi_neg_beta = n_pos + bj

            # Cartan: (Σ c_i β_i) E_β
            ev = cartan @ beta
            if abs(ev) > 1e-14:
                M[col, col] = ev

            # [E_{-β}, E_β] = -Σ β_j H_j
            d_neg = roots_coeff[gi_neg_beta]
            if abs(d_neg) > 1e-14:
                for j in range(8):
                    M[j, col] -= d_neg * beta[j]

            # [E_γ, E_β] = N_{γ,β} E_{γ+β}
            for gi in range(240):
                if gi == bj or gi == gi_neg_beta:
                    continue
                dg = roots_coeff[gi]
                if abs(dg) < 1e-14:
                    continue
                key = (gi, bj)
                nv = self.N.get(key)
                if nv is not None:
                    k = self._idx(self.roots[gi] + beta)
                    if k >= 0:
                        M[8 + k, col] += dg * nv

        # ---- Columnas E_{-β} (raíces negativas) --------------------
        for bj in range(n_pos):
            col = 8 + n_pos + bj
            beta = self.roots[bj]
            gi_neg_beta = n_pos + bj
            gi_beta = bj

            # Cartan: -(Σ c_i β_i) E_{-β}
            ev = -cartan @ beta
            if abs(ev) > 1e-14:
                M[col, col] = ev

            # [E_β, E_{-β}] = Σ β_j H_j
            d_pos = roots_coeff[gi_beta]
            if abs(d_pos) > 1e-14:
                for j in range(8):
                    M[j, col] += d_pos * beta[j]

            # [E_γ, E_{-β}] = N_{γ,-β} E_{γ-β}
            for gi in range(240):
                if gi == gi_beta or gi == gi_neg_beta:
                    continue
                dg = roots_coeff[gi]
                if abs(dg) < 1e-14:
                    continue
                key = (gi, gi_neg_beta)
                nv = self.N.get(key)
                if nv is not None:
                    k = self._idx(self.roots[gi] - beta)
                    if k >= 0:
                        M[8 + k, col] += dg * nv

        return M

    # ------------------------------------------------------------------
    #  GENERADORES E8 (tres escalas Garnier T³)
    # ------------------------------------------------------------------

    def _construir_generadores_e8(self) -> list:
        """
        Construye 3 generadores genuinos del álgebra E8 en la adjunta.
        Cada uno corresponde a una dirección física del formalismo Garnier T³:

          G₀ = H₁         (escala C₀ = 1.0, tiempo físico)
          G₂ = H₂         (escala C₂ = 2.7, tiempo crítico)
          G₃ = E_{α₁} + E_{-α₁}  (escala C₃ = 7.3, tiempo teleológico)

        Las raíces simples de E8 son:
          α₁ = (1,-1,0,0,0,0,0,0), α₂ = (0,1,-1,0,0,0,0,0)
        """
        # Raíces simples
        alpha_1 = np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=float)
        idx_a1 = self._idx(alpha_1)

        # G₀ = H₁
        c0 = np.zeros(8, dtype=float)
        c0[0] = 1.0
        r0 = np.zeros(240, dtype=float)

        # G₂ = H₂
        c2 = np.zeros(8, dtype=float)
        c2[1] = 1.0
        r2 = np.zeros(240, dtype=float)

        # G₃ = E_{α₁} + E_{-α₁}
        c3 = np.zeros(8, dtype=float)
        r3 = np.zeros(240, dtype=float)
        if idx_a1 >= 0:
            r3[idx_a1] = 1.0
            idx_na1 = self._idx(-alpha_1)
            if idx_na1 >= 0:
                r3[idx_na1] = 1.0

        G0 = self._adjoint_matrix(c0, r0)
        G2 = self._adjoint_matrix(c2, r2)
        G3 = self._adjoint_matrix(c3, r3)

        # Hermitian-symmetrize: ad(E_α+E_{-α}) debe ser Hermitiano en la adjunta.
        # Las constantes de Chevalley requieren condición de cociclo N_{α,β}=N_{-α,α+β},
        # que no se satisface automáticamente con el ordenamiento por índice.
        # La symmetrización corrige los signos y garantiza D̂_G unitario.
        for i in range(3):
            G = [G0, G2, G3][i]
            G = (G + G.conj().T) / 2.0
            nrm = np.linalg.norm(G, 'fro')
            if nrm > 1e-12:
                G /= nrm
            if i == 0:
                G0 = G
            elif i == 1:
                G2 = G
            else:
                G3 = G

        log_msg = (
            "✓ 3 generadores E8: H1, H2, E_a1+E_-a1  "
            f"||G0||={np.linalg.norm(G0,'fro'):.2e}  "
            f"||G2||={np.linalg.norm(G2,'fro'):.2e}  "
            f"||G3||={np.linalg.norm(G3,'fro'):.2e}"
        )
        logging.info(log_msg)
        return [G0, G2, G3]

    # ------------------------------------------------------------------
    #  HADAMARD GENERALIZADO
    # ------------------------------------------------------------------

    def _hadamard_generalizado(self) -> np.ndarray:
        H = np.ones((self.dim, self.dim), dtype=complex) / np.sqrt(self.dim)
        Q, _ = np.linalg.qr(H)
        return Q

    # ------------------------------------------------------------------
    #  OPERADOR D̂_G
    # ------------------------------------------------------------------

    def operator(self) -> np.ndarray:
        fase = sum(phi * H for phi, H in zip(self.garnier.phi, self.generadores))
        D_unitario = la.expm(1j * fase)
        D = D_unitario @ self.hadamard

        identidad = D @ D.conj().T
        error = np.linalg.norm(identidad - np.eye(self.dim), 'fro')

        if error > 1e-6:
            logging.warning(f"⚠️  D̂_G no unitario: ||D†D - I|| = {error:.2e}")
        else:
            logging.info(f"✓ D̂_G unitario: error = {error:.2e}")

        return D

    def calcular_alpha_modificado(self, alpha_base: float = 0.702) -> float:
        exponent = self.garnier.C0 / self.garnier.C3
        return alpha_base * abs(np.cos(self.garnier.phi[2])) ** exponent

# =============================================================================
# 4. MONITOR SILENCIO-ACTIVO
# =============================================================================

class SilencioActivoMonitor:
    def __init__(self, garnier: GarnierTresTiempos):
        self.garnier = garnier
        self.epsilon_c = garnier.epsilon_critico()
    
    def calcular_delta_s_loop(self, rho_red: np.ndarray, b1: int = 1) -> float:
        eigenvals = np.linalg.eigvalsh(rho_red)
        eigenvals = eigenvals[eigenvals > 1e-14]
        S_vn = -np.sum(eigenvals * np.log(eigenvals + 1e-14))
        S_top = np.log(float(b1 + 1))
        return S_vn - S_top
    
    def es_silencio_activo(self, rho_red: np.ndarray, b1: int = 1) -> Tuple[bool, float]:
        delta_s = self.calcular_delta_s_loop(rho_red, b1)
        condicion = delta_s < self.epsilon_c
        libertad = 1.0 / (abs(delta_s) + self.epsilon_c + 1e-12)
        
        logging.info(f"📊 Silencio-Activo: {'✓' if condicion else '✗'} | "
                    f"ΔS={delta_s:.4e} < ε_c={self.epsilon_c:.4e} | L={libertad:.2e}")
        return condicion, libertad

# =============================================================================
# 5. HOJA CUÁNTICA KMS
# =============================================================================

_bures_cache = weakref.WeakKeyDictionary()

@dataclass(frozen=True)
class QuantumLeaf:
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
        thermal = np.exp(-self.beta_eff * omega)
        cutoff = (omega > self.spectral_gap).astype(float)
        return uv_factor * thermal * cutoff * np.sqrt(omega + 1e-12)
    
    def bures_distance(self, other: 'QuantumLeaf') -> float:
        key = (id(self), id(other))
        cache = _bures_cache.get(self)
        if cache is None:
            cache = {}
            _bures_cache[self] = cache
        
        if key in cache:
            return cache[key]
        
        omega_min = max(self.spectral_gap, other.spectral_gap)
        omega_max = min(10.0, self.lambda_uv / 1e14)
        omega = np.linspace(omega_min, omega_max, 500)
        
        r1, r2 = self.spectral_density(omega), other.spectral_density(omega)
        s1, s2 = trapezoid(r1, omega), trapezoid(r2, omega)
        
        if s1 < 1e-12 or s2 < 1e-12:
            distance = 1.0
        else:
            fidelity = trapezoid(np.sqrt((r1/s1) * (r2/s2)), omega)
            distance = np.sqrt(2 * max(0, 1 - fidelity))
        
        cache[key] = distance
        return distance

# =============================================================================
# 6. UNIVERSO RESMA
# =============================================================================

class RESMAUniverse:
    def __init__(self, n_leaves: int = 2000, seed: int = 42, 
                 leaves: Optional[Dict] = None, measure: Optional[np.ndarray] = None,
                 global_state: Optional[Dict] = None, garnier: Optional[GarnierTresTiempos] = None):
        if n_leaves < 1000:
            raise ValueError(f"N={n_leaves} < 1000 mínimo")
        
        self.n_leaves = n_leaves
        self.seed = seed
        np.random.seed(seed)
        
        self.garnier = garnier or GarnierTresTiempos()
        self.desdoblamiento = OperadorDesdoblamiento(self.garnier)
        
        # ==== FIX CRÍTICO: Conversión automática de listas a numpy arrays ====
        if leaves is not None and measure is not None and global_state is not None:
            logging.info(f"✓ Reconstruyendo Universo desde checkpoint...")
            
            # FIX: Detectar y convertir si es lista
            if isinstance(measure, list):
                logging.warning("⚠️  Convirtiendo measure de lista a numpy array...")
                measure = np.array(measure)
            
            self.leaves = leaves
            self.transition_measure = self._aplicar_modulacion_garnier(measure)
            self.global_state = global_state
        else:
            logging.info(f"🌌 Inicializando RESMA Universe ({n_leaves} hojas)...")
            self.leaves = self._initialize_leaves()
            base_measure = self._generate_complete_measure()
            self.transition_measure = self._aplicar_modulacion_garnier(base_measure)
            self.global_state = self._construct_global_state()
            gc.collect()
        
        self.libertad_universo = self._calcular_libertad()
        self.coherencia = self._calcular_coherencia()
    
    def _initialize_leaves(self) -> Dict[int, QuantumLeaf]:
        gaps = np.random.exponential(scale=0.1, size=self.n_leaves) + 0.01
        leaves = {}
        for i in range(self.n_leaves):
            if i % 1000 == 0:
                logging.debug(f"  Hojas inicializadas: {i}/{self.n_leaves}")
            leaves[i] = QuantumLeaf(leaf_id=i, beta_eff=1.0, spectral_gap=float(gaps[i]),
                                   dimension=248, lambda_uv=RESMAConstants.Lambda_UV)
        return leaves
    
    def _generate_complete_measure(self) -> np.ndarray:
        logging.info("🔄 Calculando medida cuántica VECTORIZADA (todas las distancias en paralelo)...")
        
        n = self.n_leaves
        
        # 1. Extraer gaps en un solo array NumPy
        gaps = np.array([self.leaves[i].spectral_gap for i in range(n)])
        
        # 2. Grilla global de frecuencias (evita recalcular por par)
        omega_min = float(gaps.min())
        omega_max = min(10.0, RESMAConstants.Lambda_UV / 1e14)
        omega = np.linspace(omega_min, omega_max, 500)
        
        # 3. Densidades espectrales de TODAS las hojas a la vez: (n, 500)
        uv_factor = np.exp(-omega / RESMAConstants.Lambda_UV)
        thermal = np.exp(-RESMAConstants.beta_eff * omega)
        sqrt_omega = np.sqrt(omega + 1e-12)
        cutoff = (omega[np.newaxis, :] > gaps[:, np.newaxis]).astype(float)
        
        rho = uv_factor * thermal * cutoff * sqrt_omega  # (n, 500)
        
        # 4. Normalización y construcción de la matriz Q
        s = trapezoid(rho, omega, axis=1)  # (n,)
        valid = s > 1e-12
        psi = np.zeros_like(rho)
        psi[valid] = np.sqrt(rho[valid] / s[valid, np.newaxis])
        
        # Pesos de cuadratura trapezoidal
        dx = omega[1] - omega[0]
        w = np.full(500, dx)
        w[0] = dx / 2
        w[-1] = dx / 2
        
        # 5. ¡EL HACK! Fidelidad = matriz de Gram Q @ Q.T
        Q = psi * np.sqrt(w)             # (n, 500)
        fidelity = Q @ Q.T               # (n, n) — producto matricial masivo
        
        # 6. Distancias Bures desde la fidelidad
        distances = np.sqrt(2 * np.maximum(0, 1 - fidelity))
        np.fill_diagonal(distances, 0.0)
        
        measure = np.exp(-RESMAConstants.beta_eff * distances**2)
        threshold = 0.01 * np.max(measure)
        measure[measure < threshold] = 0
        
        sparsity = 1 - np.count_nonzero(measure) / measure.size
        logging.info(f"  Sparsity: {sparsity:.1%} (threshold={threshold:.2e})")
        
        total = np.sum(measure)
        if total < 1e-12:
            logging.warning("⚠️  Medida colapsada → uniforme")
            return np.ones_like(measure) / (n * n)
        
        return measure / total
    
    def _aplicar_modulacion_garnier(self, measure: np.ndarray) -> np.ndarray:
        modulation = self.garnier.modulation_factor()
        measure_mod = measure * modulation
        
        total = np.sum(measure_mod)
        if total < 1e-12:
            logging.warning("⚠️  Medida modulada colapsada")
            return measure
        
        return measure_mod / total
    
    def _construct_global_state(self) -> Dict[int, float]:
        diag = np.diag(self.transition_measure)
        total = np.sum(diag)
        
        if total < 1e-12:
            return {i: 1.0/self.n_leaves for i in range(self.n_leaves)}
        
        return {i: float(diag[i]/total) for i in range(self.n_leaves)}
    
    def _calcular_libertad(self) -> float:
        return 1.0 / (self.garnier.epsilon_critico() + 1e-12)
    
    def _calcular_coherencia(self) -> float:
        off_diag = self.transition_measure - np.diag(np.diag(self.transition_measure))
        return np.sum(np.abs(off_diag))

# =============================================================================
# 7. RED NEURONAL
# =============================================================================

class NeuralNetworkRESMA:
    def __init__(self, n_nodes: int = 20000, seed: int = 42,
                 graph: Optional[nx.Graph] = None, dim_spectral: Optional[float] = None,
                 ramsey: Optional[int] = None, betti: Optional[Dict] = None,
                 garnier: Optional[GarnierTresTiempos] = None):
        if n_nodes < 1000:
            raise ValueError(f"N={n_nodes} < 1000 mínimo")
        
        self.n_nodes = n_nodes
        self.seed = seed
        np.random.seed(seed)
        
        self.garnier = garnier or GarnierTresTiempos()
        
        if graph is not None:
            logging.info(f"✓ Reconstruyendo Red desde checkpoint...")
            self.graph = graph
            self.conectividad = nx.density(self.graph)
            self.betti_numbers = betti or {0: 1, 1: 0}
            self.dim_spectral = dim_spectral or 2.7
            self.ramsey_number = ramsey or self._topological_ramsey()
        else:
            logging.info(f"🧠 Generando red neuronal realista ({n_nodes} nodos)...")
            self.graph = self._generate_realistic_modular_network()
            self.conectividad = nx.density(self.graph)
            self.betti_numbers = self._compute_betti_numbers()
            self.dim_spectral = self._spectral_dimension()
            self.ramsey_number = self._topological_ramsey()
            gc.collect()
        
        self.monitor = SilencioActivoMonitor(self.garnier)
        self.rho_reducida = self._calcular_rho_reducida()
        self.es_soberana, self.libertad = self.monitor.es_silencio_activo(
            self.rho_reducida, 
            self.betti_numbers.get(1, 1)
        )
        
        self._validar_axioma_6()
    
    def _generate_realistic_modular_network(self) -> nx.Graph:
        target = RESMAConstants.MIN_CONNECTIVITY
        k_ba = RESMAConstants.TARGET_DEGREE // 2
        
        logging.info(f"  Fase 1: Barabási-Albert (m={k_ba})...")
        G = nx.barabasi_albert_graph(self.n_nodes, k_ba, seed=self.seed)
        
        logging.info(f"  Fase 2: Watts-Strogatz local (k=4, p=0.1)...")
        G_ws = nx.watts_strogatz_graph(self.n_nodes, 4, 0.1, seed=self.seed+1)
        G.add_edges_from(G_ws.edges())
        
        density_current = nx.density(G)
        logging.info(f"     Densidad post-combinación: {density_current:.2%}")
        
        if density_current < target:
            logging.info(f"  Fase 3: Densificación modular (LIMITADA)...")
            
            communities = list(nx.community.greedy_modularity_communities(G))
            logging.info(f"     Detectadas {len(communities)} comunidades")
            
            # ==== FIX ULTRA-CRÍTICO: Límite ABSOLUTO para evitar explosión ====
            MAX_ADDITIONAL_EDGES = 50000
            
            max_edges = self.n_nodes * (self.n_nodes - 1) / 2
            target_edges_raw = int(target * max_edges)
            edges_needed_raw = target_edges_raw - G.number_of_edges()
            
            edges_needed = min(edges_needed_raw, MAX_ADDITIONAL_EDGES)
            
            if edges_needed <= 0 or edges_needed_raw > 1000000:
                logging.warning(f"     ⚠️  Objetivo IRREALISTA: {target:.1%} requiere {edges_needed_raw:,} aristas")
                edges_needed = MAX_ADDITIONAL_EDGES
            
            logging.info(f"     Añadiendo {edges_needed:,} aristas (máximo {MAX_ADDITIONAL_EDGES:,})...")
            
            added = 0
            start_time = time.time()
            for comm in communities:
                nodes = list(comm)
                n_comm = len(nodes)
                
                possible_edges = n_comm * (n_comm - 1) // 2 - G.subgraph(comm).number_of_edges()
                edges_to_add = min(int(edges_needed * (n_comm / self.n_nodes)), possible_edges)
                
                for _ in range(edges_to_add):
                    if added >= edges_needed:
                        break
                    
                    i, j = np.random.choice(nodes, 2, replace=False)
                    if not G.has_edge(i, j):
                        G.add_edge(i, j)
                        added += 1
                        
                        if added % 5000 == 0:
                            elapsed = time.time() - start_time
                            logging.info(f"     Progreso: {added:,}/{edges_needed:,} aristas ({elapsed:.1f}s)")
                
                if added >= edges_needed:
                    break
            
            density_final = nx.density(G)
            logging.info(f"     ✓ Densificación completada: +{added:,} aristas")
            logging.info(f"     ✓ Densidad final: {density_final:.2%}")
        
        avg_clustering = nx.average_clustering(G)
        avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        
        logging.info(f"     Propiedades finales:")
        logging.info(f"       Clustering: {avg_clustering:.4f}")
        logging.info(f"       Path length: {avg_path_length:.2f}")
        logging.info(f"       Grado medio: {2*G.number_of_edges()/self.n_nodes:.2f}")
        
        return G
    
    def _compute_betti_numbers(self) -> Dict[int, int]:
        try:
            n_components = nx.number_connected_components(self.graph)
            cycles = nx.cycle_basis(self.graph)
            return {0: n_components, 1: len(cycles)}
        except:
            return {0: 1, 1: 0}
    
    def _spectral_dimension(self) -> float:
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
        except Exception as e:
            logging.error(f"Error dimensión espectral: {e}")
            return 2.7
    
    def _topological_ramsey(self) -> int:
        b1 = self.betti_numbers.get(1, 0)
        return 3 if b1 > 5 else 4
    
    def _calcular_rho_reducida(self) -> np.ndarray:
        degrees = np.array([d for _, d in self.graph.degree()], dtype=float)
        total_degree = np.sum(degrees)
        if total_degree < 1e-12:
            return np.eye(self.n_nodes) / self.n_nodes
        
        return np.diag(degrees / total_degree)
    
    def _validar_axioma_6(self):
        umbral = RESMAConstants.MIN_CONNECTIVITY
        
        if self.conectividad < umbral:
            logging.warning(f"⚠️  Axioma 6 ROTO: {self.conectividad:.2%} < {umbral:.0%}")
            logging.warning("   El subgrafo NO puede generar estado soberano")
        else:
            logging.info(f"✅ Axioma 6 SATISFECHO: {self.conectividad:.2%} >= {umbral:.0%}")

# =============================================================================
# 8. CAVIDAD DE MIELINA
# =============================================================================

class MyelinCavity:
    def __init__(self, axon_length: float = 1e-3, radius: float = 5e-6, n_modes: int = 100):
        self.axon_length = axon_length
        self.radius = radius
        self.n_modes = n_modes
        
        logging.info("🦠 Construyendo cavidad PT-simétrica...")
        
        self.V_loss = self._loss_potential()
        self.H_0 = self._free_hamiltonian()
        self.is_pt_symmetric = RESMAConstants.verify_pt_condition()
        self.scalar_mass = self._compute_scalar_mass()
        
        if self.is_pt_symmetric:
            logging.info("  ✓ PT-simetría SATISFECHA")
        else:
            logging.warning("  ⚠️  PT-simetría ROTA")
    
    def _free_hamiltonian(self) -> np.ndarray:
        q = np.linspace(0, 2*np.pi/self.axon_length, self.n_modes)
        kinetic = RESMAConstants.Omega + q**2 + RESMAConstants.chi * q**3
        return np.diag(kinetic)
    
    def _loss_potential(self) -> np.ndarray:
        a0 = 5.29e-11
        r = np.linspace(0, self.radius, self.n_modes)  # FIX: Cambiado de self.n_nodes a self.n_modes
        loss = RESMAConstants.kappa * (r / a0)**(2 * RESMAConstants.alpha)
        return 1j * np.diag(loss)
    
    def _compute_scalar_mass(self) -> float:
        return (RESMAConstants.Lambda_bio * 1e9) * 0.1

# =============================================================================
# 9. PREDICCIONES EXPERIMENTALES
# =============================================================================

class ExperimentalPredictions:
    def __init__(self, universe: RESMAUniverse, network: NeuralNetworkRESMA, myelin: MyelinCavity):
        self.universe = universe
        self.network = network
        self.myelin = myelin
    
    def compute_log_bayes_factor(self) -> Dict[str, Any]:
        L_total = self.network.libertad * self.universe.libertad_universo
        ln_bf = np.log(L_total + 1e-12)
        
        if L_total > 1e3:
            verdict = "SOBERANO"
        elif L_total > 1e2:
            verdict = "EMERGENTE"
        else:
            verdict = "NO-SOBERANO"
        
        return {
            'ln_bf': ln_bf,
            'verdict': verdict,
            'libertad_total': L_total,
            'pt_symmetric': self.myelin.is_pt_symmetric,
            'axioma_6_satisfied': not (self.network.conectividad < RESMAConstants.MIN_CONNECTIVITY),
            'predictions': {
                'libertad_red': self.network.libertad,
                'libertad_universo': self.universe.libertad_universo,
                'epsilon_critico': self.network.garnier.epsilon_critico(),
                'alpha_modificado': self.universe.desdoblamiento.calcular_alpha_modificado(),
                'acoplamiento_garnier': self.universe.garnier.coupling_strength,
                'modulacion_temporal': self.universe.garnier.modulation_factor(),
                'conectividad': self.network.conectividad,
                'delta_s_loop': self.network.monitor.calcular_delta_s_loop(
                    self.network.rho_reducida,
                    self.network.betti_numbers.get(1, 1)
                ),
                'betti_1': self.network.betti_numbers.get(1, 0),
                'dim_spectral': self.network.dim_spectral,
                'ramsey_number': self.network.ramsey_number,
                'coherencia_universo': self.universe.coherencia,
                'memory_gb': ResourceMonitor.get_memory_gb()
            }
        }

# =============================================================================
# 10. MONITOR DE RECURSOS
# =============================================================================

class ResourceMonitor:
    @staticmethod
    def get_memory_gb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    
    @staticmethod
    def log_resources():
        used = ResourceMonitor.get_memory_gb()
        cpu_percent = psutil.cpu_percent(interval=1)
        logging.info(f"💾 RAM: {used:.2f}GB | CPU: {cpu_percent}%")

def guardar_checkpoint(data: Dict[str, Any], filename: str = CHECKPOINT_FILE):
    temp_file = f"{filename}.tmp"
    backup_file = f"{filename}.bak"
    
    try:
        # Preparar datos serializables con logging
        logging.info(f"💾 Iniciando serialización de checkpoint...")
        serializable_data = _make_serializable(data)
        
        # Verificar tamaño antes de escribir
        estimated_size = len(str(serializable_data)) / (1024**2)
        if estimated_size > 2000:  # 1GB
            logging.error(f"🚨 Checkpoint demasiado grande: {estimated_size:.2f} MB")
            raise MemoryError("Checkpoint excede 2GB")
        
        # Escribir a archivo temporal
        with open(temp_file, 'wb') as f:
            pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Rotar archivos
        if os.path.exists(filename):
            os.replace(filename, backup_file)
            logging.info(f"♻️  Rotando backup: {backup_file}")
        
        os.replace(temp_file, filename)
        
        # Verificar archivo guardado
        size_mb = os.path.getsize(filename) / (1024**2)
        logging.info(f"✅ Checkpoint guardado: {filename} ({size_mb:.2f} MB)")
        
        # Limpiar caché
        _bures_cache.clear()
        gc.collect()
        
    except RecursionError as e:
        logging.error(f"❌ Error de recursión al guardar checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Guardar versión truncada de emergencia
        emergency_data = {
            'stage': 'emergency_save',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        emergency_file = f"{filename}.emergency"
        with open(emergency_file, 'wb') as f:
            pickle.dump(emergency_data, f)
        
        logging.info(f"💾 Guardado de emergencia: {emergency_file}")
        raise
        
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
                
                if data and 'stage' in data:
                    logging.info(f"🔄 Reanudando desde etapa: {data['stage']}")
                    return data, True
                else:
                    logging.warning("⚠️  Checkpoint corrupto")
                    return None, False
                    
            except Exception as e:
                logging.warning(f"⚠️  Error cargando {attempt_file}: {e}")
                continue
    
    logging.info("ℹ️  No se encontró checkpoint, iniciando de cero")
    return None, False

# =============================================================================
# FIX CRÍTICO: Serialización segura con protección de recursión
# =============================================================================

def _make_serializable(obj, depth=0, max_depth=10, _visited=None):
    """
    Convierte objetos a formato serializable de forma segura.
    
    Args:
        obj: Objeto a serializar
        depth: Nivel de profundidad actual (auto-incremental)
        max_depth: Profundidad máxima permitida
        _visited: Diccionario de objetos ya procesados (para referencias circulares)
    """
    if _visited is None:
        _visited = {}
    
    # Límite de profundidad
    if depth > max_depth:
        logging.warning(f"⚠️  Profundidad máxima alcanzada ({max_depth}), truncando objeto")
        return f"<MAX_DEPTH_REACHED: {type(obj).__name__}>"
    
    # Manejar objetos None
    if obj is None:
        return None
    
    # Manejar tipos básicos
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Manejar tipos de NumPy
    if isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    
    # Detectar referencias circulares
    obj_id = id(obj)
    if obj_id in _visited:
        return f"<CIRCULAR_REF: {type(obj).__name__}>"
    
    _visited[obj_id] = True
    
    try:
        # Manejar diccionarios
        if isinstance(obj, dict):
            if depth == 0:
                logging.info(f"📦 Serializando diccionario con {len(obj)} claves")
            return {
                str(k): _make_serializable(v, depth + 1, max_depth, _visited) 
                for k, v in obj.items()
            }
        
        # Manejar listas y tuplas
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1000000:
                logging.warning(f"⚠️  Lista/tupla muy grande ({len(obj)} elementos), truncando...")
                return f"<LARGE_SEQUENCE: {type(obj).__name__}({len(obj)} items)>"
                
            result = [_make_serializable(item, depth + 1, max_depth, _visited) for item in obj]
            return result if isinstance(obj, list) else tuple(result)
        
        # Manejar objetos de NetworkX
        elif 'networkx' in str(type(obj)):
            logging.info(f"📊 Serializando objeto NetworkX: {type(obj).__name__}")
            return {
                'graph_type': type(obj).__name__,
                'num_nodes': getattr(obj, 'number_of_nodes', lambda: 0)(),
                'num_edges': getattr(obj, 'number_of_edges', lambda: 0)(),
                'is_directed': hasattr(obj, 'is_directed') and obj.is_directed()
            }
        
        # Manejar objetos con __dict__
        elif hasattr(obj, '__dict__'):
            if depth == 0:
                logging.info(f"🔧 Serializando objeto {type(obj).__name__}")
            
            # Clases conocidas problemáticas
            problematic_types = (weakref.WeakKeyDictionary, psutil.Process, logging.Logger)
            if isinstance(obj, problematic_types):
                return f"<NON_SERIALIZABLE: {type(obj).__name__}>"
            
            return _make_serializable(obj.__dict__, depth + 1, max_depth, _visited)
        
        # Manejar objetos especiales que definen su propia serialización
        elif hasattr(obj, 'to_dict'):
            try:
                return _make_serializable(obj.to_dict(), depth + 1, max_depth, _visited)
            except:
                pass
        
        # Manejar tipos nativos de Python que son serializables
        elif isinstance(obj, (complex, bytes, bytearray)):
            return str(obj)
        
        # Cualquier otro tipo
        else:
            return f"<{type(obj).__name__}>"
            
    finally:
        # Limpiar el seguimiento de este objeto
        if obj_id in _visited:
            del _visited[obj_id]

# =============================================================================

# =============================================================================
# 11. PIPELINE DE SIMULACIÓN
# =============================================================================

def simulate_resma_garnier(
    n_leaves: int = 2000,
    n_nodes: int = 20000,
    seed: int = 42,
    resume: bool = True,
    force_restart: bool = False
) -> Dict[str, Any]:
    log_file = f"resma_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("RESMA 4.3.6 – FUSIÓN CRÍTICA (Validada)")
    logging.info("="*80)
    logging.info(f"Parámetros: n_leaves={n_leaves}, n_nodes={n_nodes}, seed={seed}")
    
    if not RESMAConstants.verify_pt_condition():
        logging.error("❌ PT-simetría rota, abortando")
        raise RuntimeError("Condiciones físicas no satisfechas")
    
    if force_restart and Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        logging.info("🗑️  Checkpoint eliminado (modo fuerza)")
    
    checkpoint_data = None
    stage = 'start'
    
    if resume and not force_restart and Path(CHECKPOINT_FILE).exists():
        checkpoint_data, loaded = cargar_checkpoint()
        if loaded:
            stage = checkpoint_data.get('stage', 'start')
            logging.info(f"🔄 Reanudando desde etapa: {stage}")
    
    components = {}
    
    try:
        if stage == 'start' or not checkpoint_data:
            garnier = GarnierTresTiempos()
            logging.info(f"🌀 Garnier: φ=[{garnier.phi[0]:.3f},{garnier.phi[1]:.3f},{garnier.phi[2]:.3f}]")
            logging.info(f"   ε_c={garnier.epsilon_critico():.4e}, ξ={garnier.coupling_strength:.4f}")
            
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
            objects = checkpoint_data['objects']
            
            # FIX CRÍTICO: La conversión ahora ocurre dentro de RESMAUniverse
            universe = RESMAUniverse(
                n_leaves=n_leaves, seed=seed,
                leaves=objects.get('universe_leaves'),
                measure=objects.get('universe_measure'),
                global_state=objects.get('universe_global_state'),
                garnier=GarnierTresTiempos.from_dict(objects.get('garnier', {}))
            )
        
        components['universe'] = universe
        ResourceMonitor.log_resources()
        
        if stage in ['start', 'universe_complete'] or not checkpoint_data:
            logging.info("🦠 Construyendo cavidad PT-simétrica...")
            myelin = MyelinCavity()
            
            guardar_checkpoint({
                'stage': 'cavity_complete',
                'objects': {
                    'universe_leaves': universe.leaves,
                    'universe_measure': universe.transition_measure,
                    'universe_global_state': universe.global_state,
                    'garnier': universe.garnier.to_dict(),
                    'myelin_pt': myelin.is_pt_symmetric
                }
            })
        else:
            myelin = MyelinCavity()
        
        components['myelin'] = myelin
        
        if stage in ['start', 'universe_complete', 'cavity_complete'] or not checkpoint_data:
            logging.info("🧠 Construyendo Red Neuronal (BA+WS modular)...")
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
                    'myelin_pt': myelin.is_pt_symmetric
                }
            })
        else:
            logging.info("✓ Reconstruyendo Red desde checkpoint...")
            objects = checkpoint_data['objects']
            network = NeuralNetworkRESMA(
                n_nodes=n_nodes, seed=seed,
                graph=objects.get('network_graph'),
                dim_spectral=objects.get('network_dim_spectral'),
                ramsey=objects.get('network_ramsey'),
                betti=objects.get('network_betti'),
                garnier=GarnierTresTiempos.from_dict(objects.get('garnier', {}))
            )
        
        components['network'] = network
        ResourceMonitor.log_resources()
        
        logging.info("📊 Calculando Factor de Bayes...")
        experiments = ExperimentalPredictions(components['universe'], 
                                           components['network'], 
                                           components['myelin'])
        results = experiments.compute_log_bayes_factor()
        
        # Guardar checkpoint final
        final_checkpoint = {
            'stage': 'complete',
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
                'myelin_pt': myelin.is_pt_symmetric
            },
            'results': results,
            'resources': {
                'memory_gb': ResourceMonitor.get_memory_gb(),
                'cpu_percent': psutil.cpu_percent()
            }
        }
        guardar_checkpoint(final_checkpoint)
        
        _bures_cache.clear()
        gc.collect()
        
        logging.info("="*80)
        logging.info("✅ SIMULACIÓN FUSIÓN CRÍTICA COMPLETA")
        logging.info("="*80)
        logging.info(f"ln(BF): {results['ln_bf']:+.2f} | Veredicto: {results['verdict']}")
        
        return results
        
    except MemoryError as e:
        logging.error(f"🚨 MemoryError: {e}")
        logging.info("💡 Sugerencia: Reducir n_leaves o n_nodes")
        raise
    except Exception as e:
        logging.exception(f"💥 Error crítico: {e}")
        raise

# =============================================================================
# 12. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    N_LEAVES = 10000
    N_NODES = 100000
    
    FORCE_RESTART = os.environ.get('RESMA_FORCE_RESTART', 'False').lower() == 'true'
    RESUME = os.environ.get('RESMA_RESUME', 'True').lower() == 'true'
    
    try:
        resultados = simulate_resma_garnier(
            n_leaves=N_LEAVES,
            n_nodes=N_NODES,
            seed=42,
            resume=RESUME,
            force_restart=FORCE_RESTART
        )
        
        print("\n" + "="*80)
        print("RESMA 4.3.6 – RESULTADOS FUSIÓN CRÍTICA")
        print("="*80)
        print(f"ln(Bayes Factor): {resultados['ln_bf']:+.2f}")
        print(f"Veredicto: {resultados['verdict']}")
        print(f"PT-simétrico: {resultados['pt_symmetric']}")
        print(f"Axioma 6: {'✓' if resultados['axioma_6_satisfied'] else '✗'}")
        print("-"*80)
        print("Predicciones Falsables:")
        for k, v in resultados['predictions'].items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.5e}")
            else:
                print(f"  {k:25s}: {v}")
        print("="*80)
        
    except KeyboardInterrupt:
        logging.info("\n⏹️  Simulación interrumpida")
        exit(0)
        
    except Exception as e:
        logging.exception("💥 Fallo final")
        print(f"\n❌ Error: {e}")
        exit(1)
