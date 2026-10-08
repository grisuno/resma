# API (page 2 of 2)
Previous: [API.md](API.md)

## sovereignty_monitor.py
- `setup_matplotlib_for_plotting` (function) `sovereignty_monitor.py:21` `def setup_matplotlib_for_plotting()` -- Setup matplotlib para visualización
- `SovereigntyMonitor.__init__` (method) `sovereignty_monitor.py:35` `def __init__(self, epsilon_c)`
- `SovereigntyMonitor.calcular_libertad` (method) `sovereignty_monitor.py:38` `def calcular_libertad(self, weights)` -- Calcula la métrica L (libertad) de una matriz de pesos
- `SovereigntyMonitor.evaluar_regimen` (method) `sovereignty_monitor.py:85` `def evaluar_regimen(self, L)` -- Evalúa el régimen del modelo
- `CNNMNIST.__init__` (method) `sovereignty_monitor.py:96` `def __init__(self)`
- `CNNMNIST.forward` (method) `sovereignty_monitor.py:110` `def forward(self, x)`
- `CNNMNIST.get_linear_layers` (method) `sovereignty_monitor.py:122` `def get_linear_layers(self)` -- Retorna todas las capas lineales para monitoreo
- `CNNMNIST.cargar_datos` (method) `sovereignty_monitor.py:126` `def cargar_datos()` -- Carga y prepara el dataset MNIST
- `ExperimentoCompleto.__init__` (method) `sovereignty_monitor.py:149` `def __init__(self, num_epochs)`
- `ExperimentoCompleto.calcular_metricas_sovereignty` (method) `sovereignty_monitor.py:189` `def calcular_metricas_sovereignty(self)` -- Calcula métricas L para todas las capas lineales
- `ExperimentoCompleto.entrenar_epoca` (method) `sovereignty_monitor.py:210` `def entrenar_epoca(self, epoca)` -- Entrena una época completa
- `ExperimentoCompleto.evaluar_epoca` (method) `sovereignty_monitor.py:233` `def evaluar_epoca(self)` -- Evalúa el modelo en el conjunto de validación
- `ExperimentoCompleto.ejecutar_experimento` (method) `sovereignty_monitor.py:253` `def ejecutar_experimento(self)` -- Ejecuta el experimento completo
- `ExperimentoCompleto.generar_graficos` (method) `sovereignty_monitor.py:362` `def generar_graficos(self)` -- Genera gráficos comprehensivos de resultados
- `ExperimentoCompleto.main` (method) `sovereignty_monitor.py:433` `def main()` -- Función principal

## train_mini_resma.py
Depends on: `garnier_nn.py`
- `main` (function) `train_mini_resma.py:8` `def main()`

## train_profile.py
Depends on: `garnier_nn.py`
- `main` (function) `train_profile.py:9` `def main()`

## visualize_resma.py
- `setup_matplotlib_for_plotting` (function) `visualize_resma.py:6` `def setup_matplotlib_for_plotting()` -- Setup matplotlib and seaborn for plotting with proper configuration.
- `diagnosticar_modelo` (function) `visualize_resma.py:30` `def diagnosticar_modelo(checkpoint_path)` -- Cargar y visualizar estado de red entrenada

