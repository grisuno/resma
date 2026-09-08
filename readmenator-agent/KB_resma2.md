# Subsystem: resma2

## resma2/main_experiment.py
- Layer: utility
- Language: py
- Symbols:
  - `set_seed` (function, line 26) `def set_seed(seed)`
  - `run_experiment` (function, line 31) `def run_experiment()`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/main_experiments.py
- Layer: utility
- Language: py
- Symbols:
  - `inject_noise` (function, line 24) `def inject_noise(x, sigma)`
  - `train_epoch` (function, line 27) `def train_epoch(model, loader, optim, obs, epoch)`
  - `run` (function, line 61) `def run()`
- Depends on: `resma2/monitor.py`, `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/monitor.py
- Layer: utility
- Language: py
- Symbols:
  - `Regime` (class, line 12) `class Regime(Enum)`
  - `LayerDiagnostics` (class, line 18) `class LayerDiagnostics`
  - `EpochSnapshot` (class, line 27) `class EpochSnapshot`
  - `SovereigntyMonitor` (class, line 35) `class SovereigntyMonitor`
  - `__init__` (method, line 36) `def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_layers, verbose)`
  - `_extract_weights` (method, line 50) `def _extract_weights(self, model)`
  - `_calculate_svd_metrics` (method, line 58) `def _calculate_svd_metrics(self, weight_matrix)`
  - `calcular_libertad` (method, line 84) `def calcular_libertad(self, weights)`
  - `calculate` (method, line 92) `def calculate(self, model)`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

## resma2/resma_app_mnist.py
- Layer: utility
- Language: py
- Symbols:
  - `add_quantum_noise` (function, line 25) `def add_quantum_noise(tensor, noise_factor)`
  - `train` (function, line 30) `def train(model, device, train_loader, optimizer, epoch, observer)`
  - `main` (function, line 65) `def main()`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/resma_breakpoint.py
- Layer: utility
- Language: py
- Symbols:
  - `find_break_point` (function, line 6) `def find_break_point()`
- Depends on: `resma2/resma_core.py`

## resma2/resma_combat_test.py
- Layer: testing
- Language: py
- Symbols:
  - `combat_test` (function, line 11) `def combat_test()`
- Depends on: `resma2/resma_core.py`

## resma2/resma_core.py
- Layer: utility
- Language: py
- Symbols:
  - `PTSymmetricActivation` (class, line 14) `class PTSymmetricActivation(Module)`
  - `E8LatticeLayer` (class, line 35) `class E8LatticeLayer(Module)`
  - `RESMABrain` (class, line 65) `class RESMABrain(Module)`
  - `__init__` (method, line 15) `def __init__(self, omega, chi, kappa_init)`
  - `forward` (method, line 26) `def forward(self, x)`
  - `__init__` (method, line 37) `def __init__(self, in_features, out_features, q_order)`
  - `_generate_ramsey_mask` (method, line 47) `def _generate_ramsey_mask(self)`
  - `forward` (method, line 59) `def forward(self, x)`
  - `__init__` (method, line 66) `def __init__(self, input_dim, hidden_dim, output_dim)`
  - `forward` (method, line 74) `def forward(self, x)`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

## resma2/resma_noise_phase_test.py
- Layer: testing
- Language: py
- Symbols:
  - `add_noise` (function, line 34) `def add_noise(x, sigma)`
  - `measure_entropy` (function, line 37) `def measure_entropy(gate_tensor)`
- Depends on: `resma2/resma_core.py`

## resma2/resma_observer.py
- Layer: utility
- Language: py
- Symbols:
  - `QuantumState` (class, line 27) `class QuantumState`
  - `RESMAObserver` (class, line 39) `class RESMAObserver`
  - `to_dict` (method, line 36) `def to_dict(self)`
  - `__init__` (method, line 40) `def __init__(self, model, epsilon_c)`
  - `_register_hooks` (method, line 51) `def _register_hooks(self)`
  - `step` (method, line 68) `def step(self, epoch)`
  - `report` (method, line 106) `def report(self, state)`
  - `plot_phase_space` (method, line 118) `def plot_phase_space(self, save_path)`
  - `hook_fn` (method, line 53) `def hook_fn(module, input, output)`
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

## resma2/resma_overload.py
- Layer: utility
- Language: py
- Symbols:
  - `overload_test` (function, line 6) `def overload_test()`
- Depends on: `resma2/resma_core.py`

## resma2/resma_train.py
- Layer: utility
- Language: py
- Depends on: `resma2/resma_core.py`

## resma2/resma_vision.py
- Layer: utility
- Language: py
- Symbols:
  - `add_noise` (function, line 11) `def add_noise(tensor, factor)`
  - `visualize_resma_perception` (function, line 14) `def visualize_resma_perception()`
- Depends on: `resma2/resma_core.py`

## resma2/resma_vision_trained.py
- Layer: utility
- Language: py
- Symbols:
  - `add_noise` (function, line 11) `def add_noise(tensor, factor)`
  - `visualize_trained_perception` (function, line 14) `def visualize_trained_perception()`
- Depends on: `resma2/resma_core.py`
