# orphans

*Community 3 | 25 files | cohesion 0.00*

## Definition

This community groups 25 file(s) rooted at `root` with dominant language py (cohesion 0.00). Central symbols: `Bayes`, `BranchingOperator`, `CNNMNIST`, `EmpiricalValidationProtocol`, `EmunaOperator`, `ExperimentalPredictions`, `ExperimentoCompleto`, `FreedomInvariant`. Core file: `main5.py` (81 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 22/11/2025 Licencia: GPL v3  Descripción:  RESMA 4.3.6.

## Files

### `.` (25 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 0 | yes |
| `demo_mini_resma.py` | py | utility | 4 | no |
| `difract.py` | py | utility | 1 | no |
| `install.sh` | sh | utility | 0 | no |
| `main.py` | py | utility | 63 | yes |
| `main2.py` | py | utility | 63 | yes |
| `main3.py` | py | utility | 30 | yes |
| `main4.1.py` | py | utility | 31 | yes |
| `main4.py.py` | py | utility | 30 | yes |
| `main5.py` | py | utility | 81 | yes |
| `monitor_extremo.py` | py | utility | 12 | yes |
| `quick_monitor.py` | py | utility | 12 | yes |
| `resma4.10.py` | py | utility | 57 | yes |
| `resma4.13.py` | py | utility | 57 | yes |
| `resma4.2.py` | py | utility | 45 | yes |
| `resma4.3.py` | py | utility | 41 | yes |
| `resma4.4.py` | py | utility | 36 | yes |
| `resma4.5.py` | py | utility | 57 | yes |
| `resma4.6.py` | py | utility | 59 | yes |
| `resma4.7.py` | py | utility | 39 | yes |

*... and 5 more files in this community.*


## Key Symbols

- `GarnierLayer` (class, `demo_mini_resma.py:8`) `class GarnierLayer(Module)` - Capa neuronal con temporalidad Garnier T³ (simplificada para demo)
- `__init__` (method, `demo_mini_resma.py:10`) `def __init__(self, in_features, out_features, device)`
- `forward` (method, `demo_mini_resma.py:27`) `def forward(self, x)` - Forward simplificado para demostración
- `demo_resma` (method, `demo_mini_resma.py:49`) `def demo_resma()` - Demostración rápida de la arquitectura RESMA-Garnier
- `visualize_uased_geometry` (function, `difract.py:4`) `def visualize_uased_geometry()`
- `RESMAConstants` (class, `main.py:32`) `class RESMAConstants` - Constantes físicas y parámetros de la teoría RESMA
- `PhysicalValidator` (class, `main.py:58`) `class PhysicalValidator` - Validación de rangos físicos para todas las constantes
- `validate_dimension` (method, `main.py:62`) `def validate_dimension(alpha)` - α ∈ (0,1) por definición de dimensión fractal
- `validate_pt_symmetry` (method, `main.py:68`) `def validate_pt_symmetry(kappa, Omega, chi)` - Verificar κ/Ω < χ/Ω < 1 para PT-simetría
- `validate_connectome_size` (method, `main.py:79`) `def validate_connectome_size(n_nodes)` - Límite inferior para conectoma biológico
- `QuantumLeaf` (class, `main.py:90`) `class QuantumLeaf` - Hoja L_i de la Resma como estado KMS mean-field.
- `__post_init__` (method, `main.py:100`) `def __post_init__(self)` - Validaciones post-construcción (Pilar 3)
- `spectral_density` (method, `main.py:106`) `def spectral_density(self, omega)` - Densidad espectral continua ρ(ω) para álgebra tipo III₁.
- `modular_entropy` (method, `main.py:114`) `def modular_entropy(self)` - Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)
- `bures_distance` (method, `main.py:121`) `def bures_distance(self, other)`
- `_spectral_moments` (method, `main.py:132`) `def _spectral_moments(self, n)` - Momentos espectrales Tr(ρ^k) para k=1..n
- `RESMAUniverse` (class, `main.py:144`) `class RESMAUniverse` - Multiverso como foliación medible sin matrices densas.
- `__init__` (method, `main.py:150`) `def __init__(self, n_leaves, seed)` - Args:
- `_initialize_leaves` (method, `main.py:168`) `def _initialize_leaves(self)` - Genera hojas con gaps espectrales distribuidos
- `_generate_gibbs_measure` (method, `main.py:181`) `def _generate_gibbs_measure(self)` - Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))
- `_construct_global_state` (method, `main.py:203`) `def _construct_global_state(self)` - Estado global: mapa de pesos por hoja (no matriz)
- `BranchingOperator` (class, `main.py:220`) `class BranchingOperator` - Operador Ĥ que abre la Resma cuando β_i es no trivial.
- `__init__` (method, `main.py:226`) `def __init__(self, leaf, threshold)`
- `_construct_cptp_map` (method, `main.py:231`) `def _construct_cptp_map(self)` - Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)
- `_local_jump_operator` (method, `main.py:240`) `def _local_jump_operator(self, power)` - K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
- `apply_branching` (method, `main.py:255`) `def apply_branching(self, state_vector)` - Aplicar canal CPTP a vector de estado local (dim=2)
- `EmunaOperator` (class, `main.py:270`) `class EmunaOperator` - Operador P̂_E: proyección teleológica no lineal.
- `__init__` (method, `main.py:276`) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (method, `main.py:282`) `def _construct_hardy_state(self)` - E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior
- `_szego_projector` (method, `main.py:286`) `def _szego_projector(self)` - Proyector P_E en base de Fourier positiva (dim reducida)

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (resma2: resma_core) and community 3 (orphans).
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (resma2: monitor) and community 3 (orphans).
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 4 file(s) lack file-level docs (e.g. `demo_mini_resma.py`)? What purpose do they serve?
- What would break if the most connected file in orphans changed?
- Should orphans be split, given cohesion 0.00?

## Sources

- `app.py`
- `demo_mini_resma.py`
- `difract.py`
- `install.sh`
- `main.py`
- `main2.py`
- `main3.py`
- `main4.1.py`
- `main4.py.py`
- `main5.py`
- `monitor_extremo.py`
- `quick_monitor.py`
- `resma4.10.py`
- `resma4.13.py`
- `resma4.2.py`
- `resma4.3.py`
- `resma4.4.py`
- `resma4.5.py`
- `resma4.6.py`
- `resma4.7.py`
- *... and 5 more*
