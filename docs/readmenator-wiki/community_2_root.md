# root

*Community 2 | 4 files | cohesion 1.00*

## Definition

This community groups 4 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `GarnierLayer`, `SilencioActivoNetwork`, `__init__`, `_build_garnier_topology`, `activar_perfilado`, `entrenar`, `entrenar_con_perfilado`, `forward`. Core file: `garnier_nn.py` (11 symbols).

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `garnier_nn.py` | py | utility | 11 | no |
| `test_ultra_simple.py` | py | testing | 0 | no |
| `train_mini_resma.py` | py | utility | 1 | no |
| `train_profile.py` | py | utility | 1 | no |

## Key Symbols

- `GarnierLayer` (class, `garnier_nn.py:9`) `class GarnierLayer(Module)` - Capa neuronal con temporalidad Garnier T³
- `__init__` (method, `garnier_nn.py:11`) `def __init__(self, in_features, out_features, device)`
- `forward` (method, `garnier_nn.py:33`) `def forward(self, x)` - Forward con no-linealidad Garnier
- `SilencioActivoNetwork` (class, `garnier_nn.py:67`) `class SilencioActivoNetwork(Module)` - Red neuronal completa con arquitectura RESMA-Garnier
- `__init__` (method, `garnier_nn.py:69`) `def __init__(self, layer_sizes, scale, device)`
- `_build_garnier_topology` (method, `garnier_nn.py:111`) `def _build_garnier_topology(self)` - Construcción BA+WS modular miniaturizada
- `forward` (method, `garnier_nn.py:130`) `def forward(self, x)` - Forward completo con tracking de métricas de consciencia
- `activar_perfilado` (method, `garnier_nn.py:168`) `def activar_perfilado(self)` - Activar perfilado de tiempo en toda la red
- `mostrar_estadisticas_perfilado` (method, `garnier_nn.py:183`) `def mostrar_estadisticas_perfilado(self)` - Mostrar estadísticas de perfilado
- `entrenar_con_perfilado` (method, `garnier_nn.py:201`) `def entrenar_con_perfilado(self, train_loader, epochs, lr)` - Entrenamiento con perfilado detallado
- `entrenar` (method, `garnier_nn.py:240`) `def entrenar(self, train_loader, epochs, lr)` - Entrenamiento incorporado con regularización Garnier
- `main` (function, `train_mini_resma.py:8`) `def main()`
- `main` (function, `train_profile.py:9`) `def main()`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 3
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 0 (resma2: resma_core) and community 2 (root).
- [INFERRED] shares_context community 1 <-> 2 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 1 (resma2: monitor) and community 2 (root).
- [INFERRED] shares_context community 2 <-> 3 (strength 0.5): Inferred shared context (language py and layer utility) with no import path between community 2 (root) and community 3 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 4 file(s) lack file-level docs (e.g. `garnier_nn.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `garnier_nn.py`
- `test_ultra_simple.py`
- `train_mini_resma.py`
- `train_profile.py`
