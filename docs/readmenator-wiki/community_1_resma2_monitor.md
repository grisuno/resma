# resma2: monitor

*Community 1 | 5 files | cohesion 0.62*

## Definition

This community groups 5 file(s) rooted at `resma2` with dominant language py (cohesion 0.62). Central symbols: `EpochSnapshot`, `LayerDiagnostics`, `QuantumState`, `RESMAObserver`, `Regime`, `SovereigntyMonitor`, `__init__`, `_calculate_svd_metrics`. Core file: `resma2/monitor.py` (9 symbols). Documented purpose: resma-exp/main.py.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `resma2/main_experiment.py` | py | utility | 2 | yes |
| `resma2/main_experiments.py` | py | utility | 3 | no |
| `resma2/monitor.py` | py | utility | 9 | yes |
| `resma2/resma_app_mnist.py` | py | utility | 3 | yes |
| `resma2/resma_observer.py` | py | utility | 9 | yes |

## Key Symbols

- `set_seed` (function, `resma2/main_experiment.py:26`) `def set_seed(seed)`
- `run_experiment` (function, `resma2/main_experiment.py:31`) `def run_experiment()`
- `inject_noise` (function, `resma2/main_experiments.py:24`) `def inject_noise(x, sigma)`
- `train_epoch` (function, `resma2/main_experiments.py:27`) `def train_epoch(model, loader, optim, obs, epoch)`
- `run` (function, `resma2/main_experiments.py:61`) `def run()`
- `Regime` (class, `resma2/monitor.py:12`) `class Regime(Enum)`
- `LayerDiagnostics` (class, `resma2/monitor.py:18`) `class LayerDiagnostics`
- `EpochSnapshot` (class, `resma2/monitor.py:27`) `class EpochSnapshot`
- `SovereigntyMonitor` (class, `resma2/monitor.py:35`) `class SovereigntyMonitor`
- `__init__` (method, `resma2/monitor.py:36`) `def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_l`
- `_extract_weights` (method, `resma2/monitor.py:50`) `def _extract_weights(self, model)`
- `_calculate_svd_metrics` (method, `resma2/monitor.py:58`) `def _calculate_svd_metrics(self, weight_matrix)`
- `calcular_libertad` (method, `resma2/monitor.py:84`) `def calcular_libertad(self, weights)`
- `calculate` (method, `resma2/monitor.py:92`) `def calculate(self, model)`
- `add_quantum_noise` (function, `resma2/resma_app_mnist.py:25`) `def add_quantum_noise(tensor, noise_factor)` - Inyecta ruido gaussiano simulando fluctuaciones de vacío
- `train` (function, `resma2/resma_app_mnist.py:30`) `def train(model, device, train_loader, optimizer, epoch, observer)`
- `main` (function, `resma2/resma_app_mnist.py:65`) `def main()`
- `QuantumState` (class, `resma2/resma_observer.py:27`) `class QuantumState` - Snapshot del estado físico-estructural de la red
- `to_dict` (method, `resma2/resma_observer.py:36`) `def to_dict(self)`
- `RESMAObserver` (class, `resma2/resma_observer.py:39`) `class RESMAObserver`
- `__init__` (method, `resma2/resma_observer.py:40`) `def __init__(self, model, epsilon_c)`
- `_register_hooks` (method, `resma2/resma_observer.py:51`) `def _register_hooks(self)` - Inyecta sondas en las capas PT para leer telemetría en tiempo real
- `hook_fn` (method, `resma2/resma_observer.py:53`) `def hook_fn(module, input, output)`
- `step` (method, `resma2/resma_observer.py:68`) `def step(self, epoch)` - Ejecutar al final de cada época de entrenamiento/validación.
- `report` (method, `resma2/resma_observer.py:106`) `def report(self, state)` - Imprime reporte formateado a consola
- `plot_phase_space` (method, `resma2/resma_observer.py:118`) `def plot_phase_space(self, save_path)` - Genera el diagrama de fase: Estructura vs Dinámica

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 5
- Cross-boundary resolved imports (EXTRACTED): 3

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: resma2/main_experiment.py imports resma2/resma_core.py.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_breakpoint.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_combat_test.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_noise_phase_test.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_overload.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_train.py in 3 hops.
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (resma2: monitor) and community 2 (root).
- [INFERRED] shares_context community 1 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (resma2: monitor) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `resma2/main_experiments.py`)? What purpose do they serve?
- What would break if the most connected file in resma2: monitor changed?
- Should resma2: monitor be split, given cohesion 0.62?

## Sources

- `resma2/main_experiment.py`
- `resma2/main_experiments.py`
- `resma2/monitor.py`
- `resma2/resma_app_mnist.py`
- `resma2/resma_observer.py`
