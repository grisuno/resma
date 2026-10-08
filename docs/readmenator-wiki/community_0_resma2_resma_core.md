# resma2: resma_core

*Community 0 | 8 files | cohesion 0.70*

## Definition

This community groups 8 file(s) rooted at `resma2` with dominant language py (cohesion 0.70). Central symbols: `E8LatticeLayer`, `PTSymmetricActivation`, `RESMABrain`, `__init__`, `_generate_ramsey_mask`, `add_noise`, `combat_test`, `find_break_point`. Core file: `resma2/resma_core.py` (10 symbols). Documented purpose: resma-core/physics.py v5.2.0 (High Flow).

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `resma2/resma_breakpoint.py` | py | utility | 1 | no |
| `resma2/resma_combat_test.py` | py | testing | 1 | no |
| `resma2/resma_core.py` | py | utility | 10 | yes |
| `resma2/resma_noise_phase_test.py` | py | testing | 2 | no |
| `resma2/resma_overload.py` | py | utility | 1 | no |
| `resma2/resma_train.py` | py | utility | 0 | no |
| `resma2/resma_vision.py` | py | utility | 2 | no |
| `resma2/resma_vision_trained.py` | py | utility | 2 | no |

## Key Symbols

- `find_break_point` (function, `resma2/resma_breakpoint.py:6`) `def find_break_point()`
- `combat_test` (function, `resma2/resma_combat_test.py:11`) `def combat_test()`
- `PTSymmetricActivation` (class, `resma2/resma_core.py:14`) `class PTSymmetricActivation(Module)`
- `__init__` (method, `resma2/resma_core.py:15`) `def __init__(self, omega, chi, kappa_init)`
- `forward` (method, `resma2/resma_core.py:26`) `def forward(self, x)`
- `E8LatticeLayer` (class, `resma2/resma_core.py:35`) `class E8LatticeLayer(Module)`
- `__init__` (method, `resma2/resma_core.py:37`) `def __init__(self, in_features, out_features, q_order)`
- `_generate_ramsey_mask` (method, `resma2/resma_core.py:47`) `def _generate_ramsey_mask(self)`
- `forward` (method, `resma2/resma_core.py:59`) `def forward(self, x)`
- `RESMABrain` (class, `resma2/resma_core.py:65`) `class RESMABrain(Module)`
- `__init__` (method, `resma2/resma_core.py:66`) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (method, `resma2/resma_core.py:74`) `def forward(self, x)`
- `add_noise` (function, `resma2/resma_noise_phase_test.py:34`) `def add_noise(x, sigma)`
- `measure_entropy` (function, `resma2/resma_noise_phase_test.py:37`) `def measure_entropy(gate_tensor)`
- `overload_test` (function, `resma2/resma_overload.py:6`) `def overload_test()`
- `add_noise` (function, `resma2/resma_vision.py:11`) `def add_noise(tensor, factor)`
- `visualize_resma_perception` (function, `resma2/resma_vision.py:14`) `def visualize_resma_perception()`
- `add_noise` (function, `resma2/resma_vision_trained.py:11`) `def add_noise(tensor, factor)`
- `visualize_trained_perception` (function, `resma2/resma_vision_trained.py:14`) `def visualize_trained_perception()`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 7
- Cross-boundary resolved imports (EXTRACTED): 3

## Connections

- [EXTRACTED] depends_on community 1 <-> 0 (strength 0.9): Extracted import edge crosses communities: resma2/main_experiment.py imports resma2/resma_core.py.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_breakpoint.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_combat_test.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_noise_phase_test.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_overload.py in 3 hops.
- [INFERRED] bridges community 1 <-> 0 (strength 0.7): Inferred cross-community bridge: resma2/monitor.py reaches resma2/resma_train.py in 3 hops.
- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (resma2: resma_core) and community 2 (root).
- [INFERRED] shares_context community 0 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (resma2: resma_core) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 7 file(s) lack file-level docs (e.g. `resma2/resma_breakpoint.py`)? What purpose do they serve?
- What would break if the most connected file in resma2: resma_core changed?
- Should resma2: resma_core be split, given cohesion 0.70?

## Sources

- `resma2/resma_breakpoint.py`
- `resma2/resma_combat_test.py`
- `resma2/resma_core.py`
- `resma2/resma_noise_phase_test.py`
- `resma2/resma_overload.py`
- `resma2/resma_train.py`
- `resma2/resma_vision.py`
- `resma2/resma_vision_trained.py`
