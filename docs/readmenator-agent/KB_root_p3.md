# Subsystem: root (page 3 of 3)
Previous: [KB_root_p2.md](KB_root_p2.md)

## sovereignty_monitor.py
- Doc: 🔥 EXPERIMENTO COMPLETO SOVEREIGNTY MONITOR 🔥 Demostración completa: ¿Puede L predecir el colapso...
- Layer: utility
- Language: py
- Symbols:
  - `setup_matplotlib_for_plotting` (function, line 21) `def setup_matplotlib_for_plotting()`
  - `SovereigntyMonitor` (class, line 29) `class SovereigntyMonitor`
  - `CNNMNIST` (class, line 94) `class CNNMNIST(Module)`
  - `cargar_datos` (method, line 126) `def cargar_datos()`
  - `ExperimentoCompleto` (class, line 146) `class ExperimentoCompleto`
  - `main` (method, line 433) `def main()`
  - `__init__` (method, line 35) `def __init__(self, epsilon_c)`
  - `calcular_libertad` (method, line 38) `def calcular_libertad(self, weights)`
  - `evaluar_regimen` (method, line 85) `def evaluar_regimen(self, L)`
  - `__init__` (method, line 96) `def __init__(self)`
  - `forward` (method, line 110) `def forward(self, x)`
  - `get_linear_layers` (method, line 122) `def get_linear_layers(self)`
  - `__init__` (method, line 149) `def __init__(self, num_epochs)`
  - `calcular_metricas_sovereignty` (method, line 189) `def calcular_metricas_sovereignty(self)`
  - `entrenar_epoca` (method, line 210) `def entrenar_epoca(self, epoca)`
  - `evaluar_epoca` (method, line 233) `def evaluar_epoca(self)`
  - `ejecutar_experimento` (method, line 253) `def ejecutar_experimento(self)`
  - `generar_graficos` (method, line 362) `def generar_graficos(self)`

## test_simple.py
- Doc: Test ultra-simple de la implementación RESMA-Garnier
- Layer: testing
- Language: py
- Symbols:
  - `test_basic_math` (function, line 8) `def test_basic_math()`

## test_ultra_simple.py
- Layer: testing
- Language: py
- Depends on: `garnier_nn.py`

## train_mini_resma.py
- Layer: utility
- Language: py
- Symbols:
  - `main` (function, line 8) `def main()`
- Depends on: `garnier_nn.py`

## train_profile.py
- Layer: utility
- Language: py
- Symbols:
  - `main` (function, line 9) `def main()`
- Depends on: `garnier_nn.py`

## visualize_resma.py
- Doc: setup_matplotlib_for_plotting: Setup matplotlib and seaborn for plotting with proper configuration.
- Layer: utility
- Language: py
- Symbols:
  - `setup_matplotlib_for_plotting` (function, line 6) `def setup_matplotlib_for_plotting()`
  - `diagnosticar_modelo` (function, line 30) `def diagnosticar_modelo(checkpoint_path)`

