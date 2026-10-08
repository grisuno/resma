# API (page 1 of 2)
Pages: [API.md](API.md), [API_p2.md](API_p2.md)

## demo_mini_resma.py
- `GarnierLayer.__init__` (method) `demo_mini_resma.py:10` `def __init__(self, in_features, out_features, device)`
- `GarnierLayer.forward` (method) `demo_mini_resma.py:27` `def forward(self, x)` -- Forward simplificado para demostración
- `GarnierLayer.demo_resma` (method) `demo_mini_resma.py:49` `def demo_resma()` -- Demostración rápida de la arquitectura RESMA-Garnier

## difract.py
- `visualize_uased_geometry` (function) `difract.py:4` `def visualize_uased_geometry()`

## garnier_nn.py
Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`
- `GarnierLayer.__init__` (method) `garnier_nn.py:11` `def __init__(self, in_features, out_features, device)`
- `GarnierLayer.forward` (method) `garnier_nn.py:33` `def forward(self, x)` -- Forward con no-linealidad Garnier Returns: (output, delta_s_loop)
- `SilencioActivoNetwork.__init__` (method) `garnier_nn.py:69` `def __init__(self, layer_sizes, scale, device)`
- `SilencioActivoNetwork.forward` (method) `garnier_nn.py:130` `def forward(self, x)` -- Forward completo con tracking de métricas de consciencia Returns: (logits, metrics)
- `SilencioActivoNetwork.activar_perfilado` (method) `garnier_nn.py:168` `def activar_perfilado(self)` -- Activar perfilado de tiempo en toda la red
- `SilencioActivoNetwork.mostrar_estadisticas_perfilado` (method) `garnier_nn.py:183` `def mostrar_estadisticas_perfilado(self)` -- Mostrar estadísticas de perfilado
- `SilencioActivoNetwork.entrenar_con_perfilado` (method) `garnier_nn.py:201` `def entrenar_con_perfilado(self, train_loader, epochs, lr)` -- Entrenamiento con perfilado detallado
- `SilencioActivoNetwork.entrenar` (method) `garnier_nn.py:240` `def entrenar(self, train_loader, epochs, lr)` -- Entrenamiento incorporado con regularización Garnier

## main.py
- `PhysicalValidator.validate_dimension` (method) `main.py:62` `def validate_dimension(alpha)` -- α ∈ (0,1) por definición de dimensión fractal
- `PhysicalValidator.validate_pt_symmetry` (method) `main.py:68` `def validate_pt_symmetry(kappa, Omega, chi)` -- Verificar κ/Ω < χ/Ω < 1 para PT-simetría
- `PhysicalValidator.validate_connectome_size` (method) `main.py:79` `def validate_connectome_size(n_nodes)` -- Límite inferior para conectoma biológico
- `QuantumLeaf.spectral_density` (method) `main.py:106` `def spectral_density(self, omega)` -- Densidad espectral continua ρ(ω) para álgebra tipo III₁.
- `QuantumLeaf.modular_entropy` (method) `main.py:114` `def modular_entropy(self)` -- Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)
- `QuantumLeaf.bures_distance` (method) `main.py:121` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `main.py:150` `def __init__(self, n_leaves, seed)` -- Args: n_leaves: Número de hojas (target: 1e5 en Colab con mean-field) seed: Reproducibilidad (Pilar 4)
- `BranchingOperator.__init__` (method) `main.py:226` `def __init__(self, leaf, threshold)`
- `BranchingOperator.apply_branching` (method) `main.py:255` `def apply_branching(self, state_vector)` -- Aplicar canal CPTP a vector de estado local (dim=2)
- `EmunaOperator.__init__` (method) `main.py:276` `def __init__(self, universe, n_samples)`
- `EmunaOperator.project` (method) `main.py:310` `def project(self, state_vector)` -- P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO
- `LindbladFractalDynamics.__init__` (method) `main.py:356` `def __init__(self, universe, emuna)`
- `LindbladFractalDynamics.evolve` (method) `main.py:389` `def evolve(self, rho0, t_span, n_steps)` -- Integración SDE con Euler-Maruyama.
- `MyelinCavity.coherence_quantum` (method) `main.py:462` `def coherence_quantum(self)` -- Discordia cuántica aproximada (ejemplo: estado separable → 0)
- `NeuralNetworkRESMA.__init__` (method) `main.py:487` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `main.py:562` `def critical_percolation_time(self)` -- t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25 Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)
- `NeuralNetworkRESMA.is_coherent_subgraph` (method) `main.py:574` `def is_coherent_subgraph(self, subgraph_nodes)` -- Verificar coherencia: subgrafo > 70% del total
- `FreedomInvariant.__init__` (method) `main.py:588` `def __init__(self, network, universe)`
- `FreedomInvariant.compute_entropy_gap` (method) `main.py:592` `def compute_entropy_gap(self)` -- Δ_S* = ε_c en punto excepcional
- `FreedomInvariant.compute_pontryagin_number` (method) `main.py:596` `def compute_pontryagin_number(self)` -- S_top[G] = χ(G)/|V| (número de Euler normalizado)
- `FreedomInvariant.compute_freedom` (method) `main.py:607` `def compute_freedom(self)` -- L[G] = Δ_S* / S_top[G]
- `FreedomInvariant.is_gauge_invariant` (method) `main.py:618` `def is_gauge_invariant(self)` -- |L[G] - 1| < 0.05 en estado crítico
- `NullModels.ising_quantum` (method) `main.py:635` `def ising_quantum(network)` -- Modelo de Ising cuántico transversal en red fractal.
- `NullModels.syk4` (method) `main.py:654` `def syk4(network)` -- SYK₄ estándar (sin R-simetría Spin(7)).
- `NullModels.random_network` (method) `main.py:670` `def random_network(network)` -- Red aleatoria Erdős-Rényi sin percolación cuántica.
- `ExperimentalPredictions.__init__` (method) `main.py:692` `def __init__(self, resma, myelin, network)`
- `ExperimentalPredictions.predict_all` (method) `main.py:699` `def predict_all(self)` -- Predicciones RESMA 3.0
- `ExperimentalPredictions.compute_bayes_factor` (method) `main.py:715` `def compute_bayes_factor(self)` -- BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L) k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)
- `ExperimentalPredictions.simulate_resma_multiverse` (method) `main.py:764` `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` -- Pipeline completo RESMA 3.0 con verificaciones de integridad.

## main2.py
- `PhysicalValidator.validate_dimension` (method) `main2.py:63` `def validate_dimension(alpha)` -- α ∈ (0,1) por definición de dimensión fractal
- `PhysicalValidator.validate_pt_symmetry` (method) `main2.py:69` `def validate_pt_symmetry(kappa, Omega, chi)` -- Verificar κ/Ω < χ/Ω < 1 para PT-simetría
- `PhysicalValidator.validate_connectome_size` (method) `main2.py:80` `def validate_connectome_size(n_nodes)` -- Límite inferior para conectoma biológico
- `QuantumLeaf.spectral_density` (method) `main2.py:106` `def spectral_density(self, omega)` -- Densidad espectral continua ρ(ω) para álgebra tipo III₁.
- `QuantumLeaf.modular_entropy` (method) `main2.py:114` `def modular_entropy(self)` -- Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)
- `QuantumLeaf.bures_distance` (method) `main2.py:121` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `main2.py:150` `def __init__(self, n_leaves, seed)` -- Args: n_leaves: Número de hojas (target: 1e5 en Colab con mean-field) seed: Reproducibilidad (Pilar 4)
- `BranchingOperator.__init__` (method) `main2.py:225` `def __init__(self, leaf, threshold)`
- `BranchingOperator.apply_branching` (method) `main2.py:254` `def apply_branching(self, state_vector)` -- Aplicar canal CPTP a vector de estado local (dim=2)
- `EmunaOperator.__init__` (method) `main2.py:275` `def __init__(self, universe, n_samples)`
- `EmunaOperator.project` (method) `main2.py:309` `def project(self, state_vector)` -- P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO
- `LindbladFractalDynamics.__init__` (method) `main2.py:355` `def __init__(self, universe, emuna)`
- `LindbladFractalDynamics.evolve` (method) `main2.py:388` `def evolve(self, rho0, t_span, n_steps)` -- Integración SDE con Euler-Maruyama.
- `MyelinCavity.coherence_quantum` (method) `main2.py:461` `def coherence_quantum(self)` -- Discordia cuántica aproximada (ejemplo: estado separable → 0)
- `NeuralNetworkRESMA.__init__` (method) `main2.py:486` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `main2.py:561` `def critical_percolation_time(self)` -- t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25 Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)
- `NeuralNetworkRESMA.is_coherent_subgraph` (method) `main2.py:573` `def is_coherent_subgraph(self, subgraph_nodes)` -- Verificar coherencia: subgrafo > 70% del total
- `FreedomInvariant.__init__` (method) `main2.py:587` `def __init__(self, network, universe)`
- `FreedomInvariant.compute_entropy_gap` (method) `main2.py:591` `def compute_entropy_gap(self)` -- Δ_S* = ε_c en punto excepcional
- `FreedomInvariant.compute_pontryagin_number` (method) `main2.py:595` `def compute_pontryagin_number(self)` -- S_top[G] = χ(G)/|V| (número de Euler normalizado)
- `FreedomInvariant.compute_freedom` (method) `main2.py:606` `def compute_freedom(self)` -- L[G] = Δ_S* / S_top[G]
- `FreedomInvariant.is_gauge_invariant` (method) `main2.py:617` `def is_gauge_invariant(self)` -- |L[G] - 1| < 0.05 en estado crítico
- `NullModels.ising_quantum` (method) `main2.py:634` `def ising_quantum(network)` -- Modelo de Ising cuántico transversal en red fractal.
- `NullModels.syk4` (method) `main2.py:653` `def syk4(network)` -- SYK₄ estándar (sin R-simetría Spin(7)).
- `NullModels.random_network` (method) `main2.py:669` `def random_network(network)` -- Red aleatoria Erdős-Rényi sin percolación cuántica.
- `ExperimentalPredictions.__init__` (method) `main2.py:691` `def __init__(self, resma, myelin, network)`
- `ExperimentalPredictions.predict_all` (method) `main2.py:698` `def predict_all(self)` -- Predicciones RESMA 3.0
- `ExperimentalPredictions.compute_bayes_factor` (method) `main2.py:713` `def compute_bayes_factor(self)` -- BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L) k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)
- `ExperimentalPredictions.simulate_resma_multiverse` (method) `main2.py:763` `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` -- Pipeline completo RESMA 3.0 con verificaciones de integridad.

## main3.py
- `Validator.dim` (method) `main3.py:52` `def dim(a)`
- `Validator.pt` (method) `main3.py:56` `def pt(k, o, c)`
- `Validator.size` (method) `main3.py:59` `def size(n)`
- `QuantumLeaf.spectral_density` (method) `main3.py:78` `def spectral_density(self, w)`
- `QuantumLeaf.modular_entropy` (method) `main3.py:81` `def modular_entropy(self)`
- `QuantumLeaf.bures_distance` (method) `main3.py:87` `def bures_distance(self, other)`
- `Universe.__init__` (method) `main3.py:102` `def __init__(self, n_leaves, seed)`
- `Network.__init__` (method) `main3.py:130` `def __init__(self, n_nodes, seed)`
- `Network.t_c` (method) `main3.py:163` `def t_c(self)`
- `MyelinCavity.__init__` (method) `main3.py:172` `def __init__(self, n_modes)`
- `MyelinCavity.coherence_quantum` (method) `main3.py:192` `def coherence_quantum(self)`
- `Bayes.__init__` (method) `main3.py:206` `def __init__(self, pred_resma, nulls)`
- `Bayes.log_lik` (method) `main3.py:210` `def log_lik(self, model_pred)`
- `Bayes.bf` (method) `main3.py:217` `def bf(self)`
- `Bayes.simulate` (method) `main3.py:233` `def simulate(n_leaves, n_nodes, seed)`

## main4.1.py
- `RC.verify_pt_condition` (method) `main4.1.py:50` `def verify_pt_condition(cls)` -- Verifica que kappa < chi*Omega para simetría PT
- `Validator.dim` (method) `main4.1.py:63` `def dim(a)`
- `Validator.pt` (method) `main4.1.py:68` `def pt(k, o, c)` -- Condición PT: kappa < chi*Omega
- `Validator.size` (method) `main4.1.py:73` `def size(n)`
- `QuantumLeaf.spectral_density` (method) `main4.1.py:92` `def spectral_density(self, w)`
- `QuantumLeaf.modular_entropy` (method) `main4.1.py:95` `def modular_entropy(self)`
- `QuantumLeaf.bures_distance` (method) `main4.1.py:104` `def bures_distance(self, other)`
- `Universe.__init__` (method) `main4.1.py:124` `def __init__(self, n_leaves, seed)`
- `Network.__init__` (method) `main4.1.py:153` `def __init__(self, n_nodes, seed)`
- `Network.t_c` (method) `main4.1.py:218` `def t_c(self)` -- Tiempo crítico de percolación
- `MyelinCavity.__init__` (method) `main4.1.py:230` `def __init__(self, n_modes)`
- `MyelinCavity.coherence_quantum` (method) `main4.1.py:251` `def coherence_quantum(self)`
- `Bayes.__init__` (method) `main4.1.py:269` `def __init__(self, pred_resma, nulls)`
- `Bayes.log_lik` (method) `main4.1.py:273` `def log_lik(self, model_pred)` -- Verosimilitud con escalas físicas realistas
- `Bayes.ln_bf` (method) `main4.1.py:288` `def ln_bf(self)` -- Factor de Bayes con penalización de complejidad
- `Bayes.simulate` (method) `main4.1.py:307` `def simulate(n_leaves, n_nodes, seed)`

## main4.py.py
- `Validator.dim` (method) `main4.py.py:52` `def dim(a)`
- `Validator.pt` (method) `main4.py.py:56` `def pt(k, o, c)`
- `Validator.size` (method) `main4.py.py:60` `def size(n)`
- `QuantumLeaf.spectral_density` (method) `main4.py.py:79` `def spectral_density(self, w)`
- `QuantumLeaf.modular_entropy` (method) `main4.py.py:82` `def modular_entropy(self)`
- `QuantumLeaf.bures_distance` (method) `main4.py.py:88` `def bures_distance(self, other)`
- `Universe.__init__` (method) `main4.py.py:103` `def __init__(self, n_leaves, seed)`
- `Network.__init__` (method) `main4.py.py:131` `def __init__(self, n_nodes, seed)`
- `Network.t_c` (method) `main4.py.py:190` `def t_c(self)`
- `MyelinCavity.__init__` (method) `main4.py.py:199` `def __init__(self, n_modes)`
- `MyelinCavity.coherence_quantum` (method) `main4.py.py:219` `def coherence_quantum(self)`
- `Bayes.__init__` (method) `main4.py.py:233` `def __init__(self, pred_resma, nulls)`
- `Bayes.log_lik` (method) `main4.py.py:237` `def log_lik(self, model_pred)`
- `Bayes.ln_bf` (method) `main4.py.py:244` `def ln_bf(self)`
- `Bayes.simulate` (method) `main4.py.py:260` `def simulate(n_leaves, n_nodes, seed)`

## main5.py
- `PhysicalValidator.validate_dimension` (method) `main5.py:74` `def validate_dimension(alpha, tolerance)` -- α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental
- `PhysicalValidator.validate_pt_symmetry` (method) `main5.py:84` `def validate_pt_symmetry(kappa, Omega, chi)` -- Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)
- `PhysicalValidator.validate_connectome_size` (method) `main5.py:95` `def validate_connectome_size(n_nodes)` -- Límite inferior para conectoma biológico realista
- `PhysicalValidator.validate_spectral_dimension` (method) `main5.py:101` `def validate_spectral_dimension(dim)` -- Validar rango físico para dimensión espectral
- `PhysicalValidator.validate_percolation_time` (method) `main5.py:106` `def validate_percolation_time(t_c, expected, tolerance)` -- Validar tiempo de percolación contra predicción empírica
- `QuantumLeaf.spectral_density` (method) `main5.py:133` `def spectral_density(self, omega)` -- Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.
- `QuantumLeaf.modular_entropy` (method) `main5.py:143` `def modular_entropy(self)` -- Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización
- `QuantumLeaf.bures_distance` (method) `main5.py:151` `def bures_distance(self, other)`
- `QuantumLeaf.haagerup_weight` (method) `main5.py:170` `def haagerup_weight(self)` -- Peso de Haagerup para regularización del operador modular
- `RESMAUniverse.__init__` (method) `main5.py:185` `def __init__(self, n_leaves, seed)` -- Args: n_leaves: Número de hojas (target: 1e5 en Colab con mean-field) seed: Reproducibilidad (Pilar 4)
- `RESMAUniverse.compute_gibbs_free_energy` (method) `main5.py:251` `def compute_gibbs_free_energy(self)` -- Energía libre de Gibbs para validación termodinámica
- `BranchingOperator.__init__` (method) `main5.py:266` `def __init__(self, leaf, threshold)`
- `BranchingOperator.apply_branching` (method) `main5.py:300` `def apply_branching(self, state_vector)` -- Aplicar canal CPTP a vector de estado local (dim=2) con normalización
- `EmunaOperator.__init__` (method) `main5.py:324` `def __init__(self, universe, n_samples)`
- `EmunaOperator.project` (method) `main5.py:361` `def project(self, state_vector)` -- P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO
- `EmunaOperator.compute_teleological_overlap` (method) `main5.py:396` `def compute_teleological_overlap(self)` -- Calcular overlap teleológico con estado objetivo
- `LindbladFractalDynamics.__init__` (method) `main5.py:412` `def __init__(self, universe, emuna)`
- `LindbladFractalDynamics.evolve` (method) `main5.py:459` `def evolve(self, rho0, t_span, n_steps)` -- Integración SDE con Euler-Maruyama y control de paso adaptativo.
- `MyelinCavity.coherence_quantum` (method) `main5.py:579` `def coherence_quantum(self)` -- Discordia cuántica aproximada con corrección PT
- `NeuralNetworkRESMA.__init__` (method) `main5.py:610` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `main5.py:735` `def critical_percolation_time(self)` -- t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25 Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)
- `NeuralNetworkRESMA.is_coherent_subgraph` (method) `main5.py:747` `def is_coherent_subgraph(self, subgraph_nodes)` -- Verificar coherencia: subgrafo > 70% del total
- `NeuralNetworkRESMA.compute_network_entropy` (method) `main5.py:751` `def compute_network_entropy(self)` -- Entropía de la red basada en distribución de grados
- `FreedomInvariant.__init__` (method) `main5.py:768` `def __init__(self, network, universe)`
- `FreedomInvariant.compute_entropy_gap` (method) `main5.py:772` `def compute_entropy_gap(self)` -- Δ_S* = ε_c en punto excepcional con corrección de regularización
- `FreedomInvariant.compute_pontryagin_number` (method) `main5.py:776` `def compute_pontryagin_number(self)` -- S_top[G] = χ(G)/|V| (número de Euler normalizado)
- `FreedomInvariant.compute_freedom` (method) `main5.py:792` `def compute_freedom(self)` -- L[G] = Δ_S* / S_top[G] con protección de división por cero
- `FreedomInvariant.is_gauge_invariant` (method) `main5.py:803` `def is_gauge_invariant(self)` -- |L[G] - 1| < 0.05 en estado crítico (invariante de libertad)
- `NullModels.ising_quantum` (method) `main5.py:823` `def ising_quantum(network)` -- Modelo de Ising cuántico transversal en red fractal.
- `NullModels.syk4` (method) `main5.py:843` `def syk4(network)` -- SYK₄ estándar (sin R-simetría Spin(7) ni E₈).
- `NullModels.random_network` (method) `main5.py:860` `def random_network(network)` -- Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.
- `ExperimentalPredictions.__init__` (method) `main5.py:883` `def __init__(self, resma, myelin, network, freedom)`
- `ExperimentalPredictions.predict_all` (method) `main5.py:891` `def predict_all(self)` -- Predicciones RESMA 4.0 con valores empíricos objetivo
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `main5.py:912` `def compute_log_bayes_factor(self)` -- log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L) FIX RESMA 4.0: Usar espacio logarítmico para evitar desbordamiento.
- `EmpiricalValidationProtocol.__init__` (method) `main5.py:981` `def __init__(self, predictions)`
- `EmpiricalValidationProtocol.evaluate_feasibility` (method) `main5.py:1014` `def evaluate_feasibility(self, budget, time_limit)` -- Evaluar viabilidad del protocolo completo
- `EmpiricalValidationProtocol.simulate_experimental_outcome` (method) `main5.py:1027` `def simulate_experimental_outcome(self, protocol_name)` -- Simular resultado experimental con ruido realista
- `EmpiricalValidationProtocol.simulate_resma_multiverse` (method) `main5.py:1052` `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)` -- Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.

## monitor_extremo.py
- `setup_matplotlib_for_plotting` (function) `monitor_extremo.py:19` `def setup_matplotlib_for_plotting()`
- `SovereigntyMonitor.__init__` (method) `monitor_extremo.py:28` `def __init__(self, epsilon_c)`
- `SovereigntyMonitor.calcular_libertad` (method) `monitor_extremo.py:31` `def calcular_libertad(self, weights)` -- Calcula la métrica L (libertad) de una matriz de pesos
- `SovereigntyMonitor.evaluar_regimen` (method) `monitor_extremo.py:63` `def evaluar_regimen(self, L)` -- Evalúa el régimen del modelo
- `ModeloGrande.__init__` (method) `monitor_extremo.py:74` `def __init__(self)`
- `ModeloGrande.forward` (method) `monitor_extremo.py:88` `def forward(self, x)`
- `ModeloGrande.get_linear_layers` (method) `monitor_extremo.py:100` `def get_linear_layers(self)`
- `ModeloGrande.generar_datos_toxico` (method) `monitor_extremo.py:103` `def generar_datos_toxico()` -- Genera datos diseñados específicamente para causar colapso
- `ModeloGrande.experimento_colapso_forzado` (method) `monitor_extremo.py:126` `def experimento_colapso_forzado()` -- Experimento diseñado para forzar el colapso del modelo
- `ModeloGrande.generar_graficos_extremos` (method) `monitor_extremo.py:339` `def generar_graficos_extremos(historial)` -- Genera gráficos del experimento extremo

## quick_monitor.py
- `setup_matplotlib_for_plotting` (function) `quick_monitor.py:19` `def setup_matplotlib_for_plotting()`
- `SovereigntyMonitor.__init__` (method) `quick_monitor.py:28` `def __init__(self, epsilon_c)`
- `SovereigntyMonitor.calcular_libertad` (method) `quick_monitor.py:31` `def calcular_libertad(self, weights)` -- Calcula la métrica L (libertad) de una matriz de pesos
- `SovereigntyMonitor.evaluar_regimen` (method) `quick_monitor.py:63` `def evaluar_regimen(self, L)` -- Evalúa el régimen del modelo
- `ModeloMNISTPequeno.__init__` (method) `quick_monitor.py:74` `def __init__(self)`
- `ModeloMNISTPequeno.forward` (method) `quick_monitor.py:83` `def forward(self, x)`
- `ModeloMNISTPequeno.get_linear_layers` (method) `quick_monitor.py:92` `def get_linear_layers(self)`
- `ModeloMNISTPequeno.generar_datos_mnist_rapido` (method) `quick_monitor.py:95` `def generar_datos_mnist_rapido()` -- Genera datos sintéticos tipo MNIST para experimento rápido
- `ModeloMNISTPequeno.entrenar_modelo_rapido` (method) `quick_monitor.py:116` `def entrenar_modelo_rapido()` -- Entrena modelo con monitoreo L en tiempo real
- `ModeloMNISTPequeno.generar_graficos_rapido` (method) `quick_monitor.py:295` `def generar_graficos_rapido(historial)` -- Genera gráficos de resultados del experimento rápido

## resma2/main_experiment.py
Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`
- `set_seed` (function) `resma2/main_experiment.py:26` `def set_seed(seed)`
- `run_experiment` (function) `resma2/main_experiment.py:31` `def run_experiment()`

## resma2/main_experiments.py
Depends on: `resma2/monitor.py`, `resma2/resma_core.py`, `resma2/resma_observer.py`
- `inject_noise` (function) `resma2/main_experiments.py:24` `def inject_noise(x, sigma)`
- `train_epoch` (function) `resma2/main_experiments.py:27` `def train_epoch(model, loader, optim, obs, epoch)`
- `run` (function) `resma2/main_experiments.py:61` `def run()`

## resma2/monitor.py
Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`
- `SovereigntyMonitor.__init__` (method) `resma2/monitor.py:36` `def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_layers, verbose)`
- `SovereigntyMonitor.calcular_libertad` (method) `resma2/monitor.py:84` `def calcular_libertad(self, weights)`
- `SovereigntyMonitor.calculate` (method) `resma2/monitor.py:92` `def calculate(self, model)`

## resma2/resma_app_mnist.py
Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`
- `add_quantum_noise` (function) `resma2/resma_app_mnist.py:25` `def add_quantum_noise(tensor, noise_factor)` -- Inyecta ruido gaussiano simulando fluctuaciones de vacío
- `train` (function) `resma2/resma_app_mnist.py:30` `def train(model, device, train_loader, optimizer, epoch, observer)`
- `main` (function) `resma2/resma_app_mnist.py:65` `def main()`

## resma2/resma_breakpoint.py
Depends on: `resma2/resma_core.py`
- `find_break_point` (function) `resma2/resma_breakpoint.py:6` `def find_break_point()`

## resma2/resma_core.py
Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`
- `PTSymmetricActivation.__init__` (method) `resma2/resma_core.py:15` `def __init__(self, omega, chi, kappa_init)`
- `PTSymmetricActivation.forward` (method) `resma2/resma_core.py:26` `def forward(self, x)`
- `E8LatticeLayer.__init__` (method) `resma2/resma_core.py:37` `def __init__(self, in_features, out_features, q_order)`
- `E8LatticeLayer.forward` (method) `resma2/resma_core.py:59` `def forward(self, x)`
- `RESMABrain.__init__` (method) `resma2/resma_core.py:66` `def __init__(self, input_dim, hidden_dim, output_dim)`
- `RESMABrain.forward` (method) `resma2/resma_core.py:74` `def forward(self, x)`

## resma2/resma_observer.py
Depends on: `resma2/monitor.py`
Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`
- `QuantumState.to_dict` (method) `resma2/resma_observer.py:36` `def to_dict(self)`
- `RESMAObserver.__init__` (method) `resma2/resma_observer.py:40` `def __init__(self, model, epsilon_c)`
- `RESMAObserver.hook_fn` (method) `resma2/resma_observer.py:53` `def hook_fn(module, input, output)`
- `RESMAObserver.step` (method) `resma2/resma_observer.py:68` `def step(self, epoch)` -- Ejecutar al final de cada época de entrenamiento/validación.
- `RESMAObserver.report` (method) `resma2/resma_observer.py:106` `def report(self, state)` -- Imprime reporte formateado a consola
- `RESMAObserver.plot_phase_space` (method) `resma2/resma_observer.py:118` `def plot_phase_space(self, save_path)` -- Genera el diagrama de fase: Estructura vs Dinámica

## resma2/resma_overload.py
Depends on: `resma2/resma_core.py`
- `overload_test` (function) `resma2/resma_overload.py:6` `def overload_test()`

## resma2/resma_vision.py
Depends on: `resma2/resma_core.py`
- `add_noise` (function) `resma2/resma_vision.py:11` `def add_noise(tensor, factor)`
- `visualize_resma_perception` (function) `resma2/resma_vision.py:14` `def visualize_resma_perception()`

## resma2/resma_vision_trained.py
Depends on: `resma2/resma_core.py`
- `add_noise` (function) `resma2/resma_vision_trained.py:11` `def add_noise(tensor, factor)`
- `visualize_trained_perception` (function) `resma2/resma_vision_trained.py:14` `def visualize_trained_perception()`

## resma4.10.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.10.py:51` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.10.py:87` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.10.py:90` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.10.py:93` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.10.py:103` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.10.py:118` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.10.py:408` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.10.py:423` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.10.py:432` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.10.py:436` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.10.py:443` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.10.py:470` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.10.py:476` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.10.py:507` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.10.py:615` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.10.py:783` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.10.py:819` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.10.py:824` `def compute_log_bayes_factor(self)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.10.py:867` `def get_memory_gb()`
- `ResourceMonitor.log_resources` (method) `resma4.10.py:872` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.10.py:877` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.10.py:935` `def cargar_checkpoint(filename)`
- `ResourceMonitor.simulate_resma_garnier` (method) `resma4.10.py:1067` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resma4.13.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.13.py:51` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.13.py:87` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.13.py:90` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.13.py:93` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.13.py:103` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.13.py:118` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.13.py:408` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.13.py:423` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.13.py:432` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.13.py:436` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.13.py:443` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.13.py:470` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.13.py:476` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.13.py:507` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.13.py:640` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.13.py:808` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.13.py:844` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.13.py:849` `def compute_log_bayes_factor(self)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.13.py:892` `def get_memory_gb()`
- `ResourceMonitor.log_resources` (method) `resma4.13.py:897` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.13.py:902` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.13.py:960` `def cargar_checkpoint(filename)`
- `ResourceMonitor.simulate_resma_garnier` (method) `resma4.13.py:1092` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resma4.2.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.2.py:67` `def verify_pt_condition(cls)` -- Verificar condición PT: κ < χΩ
- `PhysicalValidator.validate_dimension` (method) `resma4.2.py:80` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.2.py:88` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.2.py:96` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.2.py:101` `def validate_spectral_dimension(dim)`
- `QuantumLeaf.spectral_density` (method) `resma4.2.py:121` `def spectral_density(self, omega)` -- ρ(ω) con regularización UV
- `QuantumLeaf.modular_entropy` (method) `resma4.2.py:126` `def modular_entropy(self)` -- S = -∫ ρ log ρ dω
- `QuantumLeaf.bures_distance` (method) `resma4.2.py:136` `def bures_distance(self, other)` -- Distancia de Bures W₂(ρ₁, ρ₂)
- `QuantumLeaf.haagerup_weight` (method) `resma4.2.py:154` `def haagerup_weight(self)` -- Peso de Haagerup para regularización
- `RESMAUniverse.__init__` (method) `resma4.2.py:165` `def __init__(self, n_leaves, seed)`
- `RESMAUniverse.compute_gibbs_free_energy` (method) `resma4.2.py:216` `def compute_gibbs_free_energy(self)`
- `EmunaOperator.__init__` (method) `resma4.2.py:227` `def __init__(self, universe, n_samples)`
- `EmunaOperator.project` (method) `resma4.2.py:258` `def project(self, state_vector)` -- P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)
- `MyelinCavity.coherence_quantum` (method) `resma4.2.py:327` `def coherence_quantum(self)` -- Coherencia cuántica con verificación espectral
- `NeuralNetworkRESMA.__init__` (method) `resma4.2.py:354` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `resma4.2.py:461` `def critical_percolation_time(self)` -- t_c = 21 · (N/N₀)^0.25 / log R_Q
- `ExperimentalPredictions.__init__` (method) `resma4.2.py:477` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.predict_all` (method) `resma4.2.py:483` `def predict_all(self)` -- Predicciones RESMA 4.2
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.2.py:496` `def compute_log_bayes_factor(self)` -- ln(BF) con AIC
- `ExperimentalPredictions.simulate_resma_complete` (method) `resma4.2.py:556` `def simulate_resma_complete(n_leaves, n_nodes, seed)` -- Pipeline RESMA 4.2 completo

## resma4.3.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.3.py:35` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.3.py:40` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.3.py:49` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.3.py:54` `def guardar_checkpoint(data, filename)` -- Guardado atómico con backup
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.3.py:84` `def cargar_checkpoint(filename)` -- Cargar checkpoint con fallback
- `RESMAConstants.verify_pt_condition` (method) `resma4.3.py:127` `def verify_pt_condition(cls)`
- `QuantumLeaf.spectral_density` (method) `resma4.3.py:154` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.3.py:158` `def bures_distance(self, other)` -- Distancia Bures con caché EXTERNO (no en instancia)
- `RESMAUniverse.__init__` (method) `resma4.3.py:193` `def __init__(self, n_leaves, seed)`
- `PhysicalValidator.validate_dimension` (method) `resma4.3.py:271` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.3.py:279` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.3.py:287` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.3.py:292` `def validate_spectral_dimension(dim)`
- `MyelinCavity.coherence_quantum` (method) `resma4.3.py:331` `def coherence_quantum(self)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.3.py:353` `def __init__(self, n_nodes, seed)`
- `NeuralNetworkRESMA.critical_percolation_time` (method) `resma4.3.py:476` `def critical_percolation_time(self)` -- Tiempo crítico de percolación
- `ExperimentalPredictions.__init__` (method) `resma4.3.py:486` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.3.py:492` `def compute_log_bayes_factor(self)`
- `ExperimentalPredictions.simulate_resma_with_checkpointing` (method) `resma4.3.py:545` `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` -- Pipeline con reanudación inteligente desde checkpoints

## resma4.4.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.4.py:36` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.4.py:41` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.4.py:50` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.4.py:59` `def guardar_checkpoint(data, filename)` -- Guarda el estado COMPLETO de los objetos, no solo metadatos
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.4.py:93` `def cargar_checkpoint(filename)` -- Carga el estado COMPLETO desde disco
- `RESMAConstants.verify_pt_condition` (method) `resma4.4.py:146` `def verify_pt_condition(cls)`
- `QuantumLeaf.spectral_density` (method) `resma4.4.py:172` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.4.py:176` `def bures_distance(self, other)` -- Distancia Bures con caché externo
- `RESMAUniverse.__init__` (method) `resma4.4.py:206` `def __init__(self, n_leaves, seed, leaves, measure, global_state)` -- Constructor que puede recibir estado serializado
- `PhysicalValidator.validate_dimension` (method) `resma4.4.py:318` `def validate_dimension(alpha, tolerance)`
- `PhysicalValidator.validate_pt_symmetry` (method) `resma4.4.py:326` `def validate_pt_symmetry(kappa, Omega, chi)`
- `PhysicalValidator.validate_connectome_size` (method) `resma4.4.py:334` `def validate_connectome_size(n_nodes)`
- `PhysicalValidator.validate_spectral_dimension` (method) `resma4.4.py:339` `def validate_spectral_dimension(dim)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.4.py:378` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)` -- Constructor que puede recibir grafo ya construido
- `NeuralNetworkRESMA.simulate_resma_with_checkpointing` (method) `resma4.4.py:554` `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` -- Pipeline con reanudación que realmente carga objetos

## resma4.5.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.5.py:36` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.5.py:41` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.5.py:50` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.5.py:59` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.5.py:88` `def cargar_checkpoint(filename)`
- `RESMAConstants.verify_pt_condition` (method) `resma4.5.py:152` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.factor_escala` (method) `resma4.5.py:181` `def factor_escala(self, tiempo_idx)` -- Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.5.py:185` `def epsilon_critico(self)` -- Entropía crítica de percolación (ADIMENSIONAL). log(2) es la entropía de un bit cuántico crítico.
- `GarnierTresTiempos.to_dict` (method) `resma4.5.py:192` `def to_dict(self)` -- Para serialización
- `GarnierTresTiempos.from_dict` (method) `resma4.5.py:197` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.5.py:206` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.5.py:235` `def operator(self)` -- Construye D̂_G(ϕ) dimensionalmente consistente
- `OperadorDesdoblamiento.aplicar_a_estado` (method) `resma4.5.py:254` `def aplicar_a_estado(self, estado)` -- Aplica desdoblamiento a un estado cuántico |Ψ⟩
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.5.py:260` `def calcular_alpha_modificado(self, alpha_base)` -- α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃)) Garantiza α' ∈ [0, α]
- `SilencioActivoMonitor.__init__` (method) `resma4.5.py:273` `def __init__(self, garnier, network)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.5.py:278` `def calcular_delta_s_loop(self, rho_red)` -- ΔS_loop = S_vN(ρ_red) - log(b₁ + 1) rho_red: matriz densidad reducida (si es None, se calcula)
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.5.py:307` `def es_silencio_activo(self, rho_red)` -- Verifica Silencio-Activo y calcula Libertad L.
- `SilencioActivoMonitor.umbral_percolacion` (method) `resma4.5.py:324` `def umbral_percolacion(self)` -- Umbral de percolación para soberanía: 70% (Axioma 6)
- `QuantumLeaf.spectral_density` (method) `resma4.5.py:349` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.5.py:353` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.5.py:383` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` -- Constructor que puede recibir estado serializado
- `MyelinCavity.__init__` (method) `resma4.5.py:488` `def __init__(self, axon_length, radius, n_modes)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.5.py:520` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` -- Constructor que puede recibir grafo ya construido
- `NeuralNetworkRESMA.validar_axioma_6` (method) `resma4.5.py:644` `def validar_axioma_6(self)` -- Verifica: conectividad > 70% para soberanía
- `ExperimentalPredictions.__init__` (method) `resma4.5.py:661` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.5.py:666` `def compute_log_bayes_factor(self)` -- Calcula Factor de Bayes integrando Garnier
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.5.py:695` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` -- Pipeline único con Garnier integrado

## resma4.6.py
- `ResourceMonitor.get_memory_gb` (method) `resma4.6.py:33` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.6.py:38` `def check_memory_limit(threshold)`
- `ResourceMonitor.log_resources` (method) `resma4.6.py:47` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.6.py:56` `def guardar_checkpoint(data, filename)` -- Guarda estado completo con manejo robusto de errores
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.6.py:86` `def cargar_checkpoint(filename)` -- Carga checkpoint con fallback automático
- `RESMAConstants.verify_pt_condition` (method) `resma4.6.py:158` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.factor_escala` (method) `resma4.6.py:194` `def factor_escala(self, tiempo_idx)` -- Factor de escala con supresión ZPE
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.6.py:200` `def epsilon_critico(self)` -- **UMBRAL CRÍTICO CON ZPE**: Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)
- `GarnierTresTiempos.to_dict` (method) `resma4.6.py:208` `def to_dict(self)` -- Serialización completa
- `GarnierTresTiempos.from_dict` (method) `resma4.6.py:220` `def from_dict(cls, data)` -- Deserialización
- `OperadorDesdoblamiento.__init__` (method) `resma4.6.py:238` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.6.py:284` `def operator(self)` -- Construye D̂_G(ϕ) con cancelación ZPE
- `OperadorDesdoblamiento.alpha_modificado` (method) `resma4.6.py:304` `def alpha_modificado(self, alpha_base)` -- **α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**
- `SilencioActivoMonitor.__init__` (method) `resma4.6.py:318` `def __init__(self, garnier, network)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.6.py:327` `def calcular_delta_s_loop(self, rho_red)` -- **ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE** **NUEVO**: La entropía ZPE se SUMA a la entropía total
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.6.py:383` `def es_silencio_activo(self, rho_red)` -- **DETECCIÓN DE ANTAGONISMO**: Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)
- `SilencioActivoMonitor.umbral_percolacion` (method) `resma4.6.py:411` `def umbral_percolacion(self)` -- Umbral para soberanía: 70%
- `SilencioActivoMonitor.modo_goldstone` (method) `resma4.6.py:415` `def modo_goldstone(self)` -- **MODO GOLDSTONE DEL DOBLE CUÁNTICO**: Excitación colectiva que anuncia ruptura de simetría ZPE
- `QuantumLeaf.spectral_density` (method) `resma4.6.py:453` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.6.py:457` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.6.py:487` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `MyelinCavity.__init__` (method) `resma4.6.py:611` `def __init__(self, axon_length, radius, n_modes)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.6.py:662` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `NeuralNetworkRESMA.validar_axioma_6` (method) `resma4.6.py:836` `def validar_axioma_6(self)` -- **AXIOMA 6**: Conectividad > 70% para soberanía
- `ExperimentalPredictions.__init__` (method) `resma4.6.py:856` `def __init__(self, universe, myelin, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.6.py:861` `def compute_log_bayes_factor(self)` -- Calcula Factor de Bayes con antagonismo ZPE-Silencio
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.6.py:906` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)` -- Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio

## resma4.7.py
- `GarnierTresTiempos.factor_escala` (method) `resma4.7.py:72` `def factor_escala(self, tiempo_idx)` -- Factor de escala temporal
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.7.py:77` `def epsilon_critico(self)` -- Entropía crítica con corrección de acoplamiento: ε_c = log(2) · (C0/C3)² · (1 + ξ)
- `GarnierTresTiempos.modulation_factor` (method) `resma4.7.py:85` `def modulation_factor(self)` -- Factor de modulación para la medida cuántica: M = exp(-|φ₃ - π|/C3) Máximo cuando φ₃ ≈ π (apertura temporal óptima)
- `OperadorDesdoblamiento.__init__` (method) `resma4.7.py:98` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.7.py:115` `def operator(self)` -- Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)
- `OperadorDesdoblamiento.aplicar_modulacion` (method) `resma4.7.py:120` `def aplicar_modulacion(self, state_vector)` -- Aplica desdoblamiento a vector de estado
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.7.py:126` `def calcular_alpha_modificado(self, alpha_base)` -- α'(φ) = α · |cos(φ₃)|^(C0/C3) Garantiza α' ∈ [0, α]
- `SilencioActivoMonitor.__init__` (method) `resma4.7.py:143` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.7.py:147` `def calcular_delta_s_loop(self, rho_red, b1)` -- ΔS_loop = S_vN(ρ) - log(b₁ + 1)
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.7.py:166` `def es_silencio_activo(self, rho_red, b1)` -- Verifica condición y calcula libertad L = 1/(ΔS + ε_c)
- `QuantumLeaf.spectral_density` (method) `resma4.7.py:197` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.7.py:203` `def bures_distance(self, other)` -- Distancia de Bures simplificada
- `RESMAUniverse.__init__` (method) `resma4.7.py:226` `def __init__(self, n_leaves, seed, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.7.py:341` `def __init__(self, n_nodes, seed, garnier)`
- `ExperimentalPredictions.__init__` (method) `resma4.7.py:487` `def __init__(self, universe, network)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.7.py:491` `def compute_log_bayes_factor(self)` -- ln(BF) ∝ log(L_red · L_univ) Veredicto basado en libertad total
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.7.py:537` `def simulate_resma_garnier(n_leaves, n_nodes, seed)` -- Pipeline completo RESMA-Garnier con correcciones

## resma4.8.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.8.py:54` `def verify_pt_condition(cls)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.8.py:72` `def get_memory_gb()`
- `ResourceMonitor.check_memory_limit` (method) `resma4.8.py:77` `def check_memory_limit()`
- `ResourceMonitor.log_resources` (method) `resma4.8.py:86` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.8.py:91` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.8.py:117` `def cargar_checkpoint(filename)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.8.py:182` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.8.py:186` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.8.py:189` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.8.py:199` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.8.py:210` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.8.py:233` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.8.py:248` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.8.py:257` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.8.py:261` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.8.py:268` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.8.py:295` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.8.py:301` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.8.py:332` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.8.py:434` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.8.py:585` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.8.py:621` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.8.py:626` `def compute_log_bayes_factor(self)`
- `ExperimentalPredictions.simulate_resma_garnier` (method) `resma4.8.py:667` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`

## resma4.9.py
- `RESMAConstants.verify_pt_condition` (method) `resma4.9.py:50` `def verify_pt_condition(cls)`
- `GarnierTresTiempos.epsilon_critico` (method) `resma4.9.py:86` `def epsilon_critico(self)`
- `GarnierTresTiempos.modulation_factor` (method) `resma4.9.py:89` `def modulation_factor(self)`
- `GarnierTresTiempos.to_dict` (method) `resma4.9.py:92` `def to_dict(self)`
- `GarnierTresTiempos.from_dict` (method) `resma4.9.py:102` `def from_dict(cls, data)`
- `OperadorDesdoblamiento.__init__` (method) `resma4.9.py:112` `def __init__(self, garnier, dimension)`
- `OperadorDesdoblamiento.operator` (method) `resma4.9.py:135` `def operator(self)`
- `OperadorDesdoblamiento.calcular_alpha_modificado` (method) `resma4.9.py:150` `def calcular_alpha_modificado(self, alpha_base)`
- `SilencioActivoMonitor.__init__` (method) `resma4.9.py:159` `def __init__(self, garnier)`
- `SilencioActivoMonitor.calcular_delta_s_loop` (method) `resma4.9.py:163` `def calcular_delta_s_loop(self, rho_red, b1)`
- `SilencioActivoMonitor.es_silencio_activo` (method) `resma4.9.py:170` `def es_silencio_activo(self, rho_red, b1)`
- `QuantumLeaf.spectral_density` (method) `resma4.9.py:197` `def spectral_density(self, omega)`
- `QuantumLeaf.bures_distance` (method) `resma4.9.py:203` `def bures_distance(self, other)`
- `RESMAUniverse.__init__` (method) `resma4.9.py:234` `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `NeuralNetworkRESMA.__init__` (method) `resma4.9.py:342` `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `MyelinCavity.__init__` (method) `resma4.9.py:493` `def __init__(self, axon_length, radius, n_modes)`
- `ExperimentalPredictions.__init__` (method) `resma4.9.py:529` `def __init__(self, universe, network, myelin)`
- `ExperimentalPredictions.compute_log_bayes_factor` (method) `resma4.9.py:534` `def compute_log_bayes_factor(self)`
- `ResourceMonitor.get_memory_gb` (method) `resma4.9.py:577` `def get_memory_gb()`
- `ResourceMonitor.log_resources` (method) `resma4.9.py:582` `def log_resources()`
- `ResourceMonitor.guardar_checkpoint` (method) `resma4.9.py:587` `def guardar_checkpoint(data, filename)`
- `ResourceMonitor.cargar_checkpoint` (method) `resma4.9.py:615` `def cargar_checkpoint(filename)`
- `ResourceMonitor.simulate_resma_garnier` (method) `resma4.9.py:655` `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`


Next: [API_p2.md](API_p2.md)
