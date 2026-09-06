# API

## demo_mini_resma.py

### demo_resma `def demo_resma()`
- Defined: `demo_mini_resma.py:49`
- Doc: Demostración rápida de la arquitectura RESMA-Garnier

### __init__ `def __init__(self, in_features, out_features, device)`
- Defined: `demo_mini_resma.py:10`

### forward `def forward(self, x)`
- Defined: `demo_mini_resma.py:27`
- Doc: Forward simplificado para demostración

## difract.py

### visualize_uased_geometry `def visualize_uased_geometry()`
- Defined: `difract.py:4`

## garnier_nn.py

### __init__ `def __init__(self, in_features, out_features, device)`
- Defined: `garnier_nn.py:11`
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### forward `def forward(self, x)`
- Defined: `garnier_nn.py:33`
- Doc: Forward con no-linealidad Garnier
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### __init__ `def __init__(self, layer_sizes, scale, device)`
- Defined: `garnier_nn.py:69`
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### _build_garnier_topology `def _build_garnier_topology(self)`
- Defined: `garnier_nn.py:111`
- Doc: Construcción BA+WS modular miniaturizada
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### forward `def forward(self, x)`
- Defined: `garnier_nn.py:130`
- Doc: Forward completo con tracking de métricas de consciencia
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### activar_perfilado `def activar_perfilado(self)`
- Defined: `garnier_nn.py:168`
- Doc: Activar perfilado de tiempo en toda la red
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### mostrar_estadisticas_perfilado `def mostrar_estadisticas_perfilado(self)`
- Defined: `garnier_nn.py:183`
- Doc: Mostrar estadísticas de perfilado
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### entrenar_con_perfilado `def entrenar_con_perfilado(self, train_loader, epochs, lr)`
- Defined: `garnier_nn.py:201`
- Doc: Entrenamiento con perfilado detallado
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

### entrenar `def entrenar(self, train_loader, epochs, lr)`
- Defined: `garnier_nn.py:240`
- Doc: Entrenamiento incorporado con regularización Garnier
- Imported by: `test_ultra_simple.py`, `train_mini_resma.py`, `train_profile.py`

## main.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed)`
- Defined: `main.py:764`
- Doc: Pipeline completo RESMA 3.0 con verificaciones de integridad.

### validate_dimension `def validate_dimension(alpha)`
- Defined: `main.py:62`
- Doc: α ∈ (0,1) por definición de dimensión fractal

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main.py:68`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main.py:79`
- Doc: Límite inferior para conectoma biológico

### __post_init__ `def __post_init__(self)`
- Defined: `main.py:100`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main.py:106`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁.

### modular_entropy `def modular_entropy(self)`
- Defined: `main.py:114`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)

### bures_distance `def bures_distance(self, other)`
- Defined: `main.py:121`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main.py:132`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main.py:150`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main.py:168`
- Doc: Genera hojas con gaps espectrales distribuidos

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main.py:181`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main.py:203`
- Doc: Estado global: mapa de pesos por hoja (no matriz)

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main.py:226`

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main.py:231`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main.py:240`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main.py:255`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2)

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main.py:276`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main.py:282`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main.py:286`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main.py:294`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main.py:310`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main.py:356`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main.py:362`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main.py:372`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main.py:382`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞]

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main.py:389`
- Doc: Integración SDE con Euler-Maruyama.

### __post_init__ `def __post_init__(self)`
- Defined: `main.py:437`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main.py:442`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main.py:448`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.7

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main.py:456`
- Doc: Verificar κ/Ω < χ/Ω < 1

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main.py:462`
- Doc: Discordia cuántica aproximada (ejemplo: estado separable → 0)

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main.py:487`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main.py:500`
- Doc: Grafo dirigido con distribución de grados power-law.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main.py:510`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main.py:527`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main.py:548`
- Doc: Matriz de distancias shortest-path (sparse CSR)

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main.py:562`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main.py:574`
- Doc: Verificar coherencia: subgrafo > 70% del total

### __init__ `def __init__(self, network, universe)`
- Defined: `main.py:588`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main.py:592`
- Doc: Δ_S* = ε_c en punto excepcional

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main.py:596`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main.py:607`
- Doc: L[G] = Δ_S* / S_top[G]

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main.py:618`
- Doc: |L[G] - 1| < 0.05 en estado crítico

### ising_quantum `def ising_quantum(network)`
- Defined: `main.py:635`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main.py:654`
- Doc: SYK₄ estándar (sin R-simetría Spin(7)).

### random_network `def random_network(network)`
- Defined: `main.py:670`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica.

### __init__ `def __init__(self, resma, myelin, network)`
- Defined: `main.py:692`

### predict_all `def predict_all(self)`
- Defined: `main.py:699`
- Doc: Predicciones RESMA 3.0

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main.py:710`
- Doc: q₀ = 2π/L_E8 (sin ajuste)

### compute_bayes_factor `def compute_bayes_factor(self)`
- Defined: `main.py:715`
- Doc: BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)

## main2.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed)`
- Defined: `main2.py:763`
- Doc: Pipeline completo RESMA 3.0 con verificaciones de integridad.

### validate_dimension `def validate_dimension(alpha)`
- Defined: `main2.py:63`
- Doc: α ∈ (0,1) por definición de dimensión fractal

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main2.py:69`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main2.py:80`
- Doc: Límite inferior para conectoma biológico

### __post_init__ `def __post_init__(self)`
- Defined: `main2.py:101`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main2.py:106`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁.

### modular_entropy `def modular_entropy(self)`
- Defined: `main2.py:114`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)

### bures_distance `def bures_distance(self, other)`
- Defined: `main2.py:121`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main2.py:132`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main2.py:150`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main2.py:167`
- Doc: Genera hojas con gaps espectrales distribuidos

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main2.py:180`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main2.py:202`
- Doc: Estado global: mapa de pesos por hoja (no matriz)

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main2.py:225`

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main2.py:230`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main2.py:239`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main2.py:254`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2)

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main2.py:275`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main2.py:281`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main2.py:285`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main2.py:293`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main2.py:309`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main2.py:355`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main2.py:361`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main2.py:371`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main2.py:381`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞]

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main2.py:388`
- Doc: Integración SDE con Euler-Maruyama.

### __post_init__ `def __post_init__(self)`
- Defined: `main2.py:436`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main2.py:441`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main2.py:447`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.7

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main2.py:455`
- Doc: Verificar κ/Ω < χ/Ω < 1

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main2.py:461`
- Doc: Discordia cuántica aproximada (ejemplo: estado separable → 0)

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main2.py:486`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main2.py:499`
- Doc: Grafo dirigido con distribución de grados power-law.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main2.py:509`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main2.py:526`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main2.py:547`
- Doc: Matriz de distancias shortest-path (sparse CSR)

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main2.py:561`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main2.py:573`
- Doc: Verificar coherencia: subgrafo > 70% del total

### __init__ `def __init__(self, network, universe)`
- Defined: `main2.py:587`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main2.py:591`
- Doc: Δ_S* = ε_c en punto excepcional

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main2.py:595`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main2.py:606`
- Doc: L[G] = Δ_S* / S_top[G]

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main2.py:617`
- Doc: |L[G] - 1| < 0.05 en estado crítico

### ising_quantum `def ising_quantum(network)`
- Defined: `main2.py:634`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main2.py:653`
- Doc: SYK₄ estándar (sin R-simetría Spin(7)).

### random_network `def random_network(network)`
- Defined: `main2.py:669`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica.

### __init__ `def __init__(self, resma, myelin, network)`
- Defined: `main2.py:691`

### predict_all `def predict_all(self)`
- Defined: `main2.py:698`
- Doc: Predicciones RESMA 3.0

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main2.py:708`
- Doc: q₀ = 2π/L_E8 (sin ajuste)

### compute_bayes_factor `def compute_bayes_factor(self)`
- Defined: `main2.py:713`
- Doc: BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)

## main3.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main3.py:233`

### dim `def dim(a)`
- Defined: `main3.py:52`

### pt `def pt(k, o, c)`
- Defined: `main3.py:56`

### size `def size(n)`
- Defined: `main3.py:59`

### __post_init__ `def __post_init__(self)`
- Defined: `main3.py:74`

### spectral_density `def spectral_density(self, w)`
- Defined: `main3.py:78`

### modular_entropy `def modular_entropy(self)`
- Defined: `main3.py:81`

### bures_distance `def bures_distance(self, other)`
- Defined: `main3.py:87`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main3.py:102`

### _gibbs `def _gibbs(self)`
- Defined: `main3.py:110`

### _global `def _global(self)`
- Defined: `main3.py:119`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main3.py:130`

### _spectral_dim `def _spectral_dim(self, k)`
- Defined: `main3.py:139`

### _ramsey `def _ramsey(self)`
- Defined: `main3.py:149`

### t_c `def t_c(self)`
- Defined: `main3.py:163`

### __init__ `def __init__(self, n_modes)`
- Defined: `main3.py:172`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main3.py:178`

### _loss_potential `def _loss_potential(self)`
- Defined: `main3.py:183`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main3.py:189`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main3.py:192`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main3.py:206`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main3.py:210`

### bf `def bf(self)`
- Defined: `main3.py:217`

## main4.1.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main4.1.py:307`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `main4.1.py:50`
- Doc: Verifica que kappa < chi*Omega para simetría PT

### dim `def dim(a)`
- Defined: `main4.1.py:63`

### pt `def pt(k, o, c)`
- Defined: `main4.1.py:68`
- Doc: Condición PT: kappa < chi*Omega

### size `def size(n)`
- Defined: `main4.1.py:73`

### __post_init__ `def __post_init__(self)`
- Defined: `main4.1.py:88`

### spectral_density `def spectral_density(self, w)`
- Defined: `main4.1.py:92`

### modular_entropy `def modular_entropy(self)`
- Defined: `main4.1.py:95`

### bures_distance `def bures_distance(self, other)`
- Defined: `main4.1.py:104`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main4.1.py:124`

### _gibbs `def _gibbs(self)`
- Defined: `main4.1.py:133`

### _global `def _global(self)`
- Defined: `main4.1.py:142`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main4.1.py:153`

### _spectral_dim `def _spectral_dim(self, k, n_fit)`
- Defined: `main4.1.py:163`
- Doc: Dimensión espectral corregida

### _ramsey `def _ramsey(self)`
- Defined: `main4.1.py:199`
- Doc: Número de Ramsey topológico

### t_c `def t_c(self)`
- Defined: `main4.1.py:218`
- Doc: Tiempo crítico de percolación

### __init__ `def __init__(self, n_modes)`
- Defined: `main4.1.py:230`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main4.1.py:237`

### _loss_potential `def _loss_potential(self)`
- Defined: `main4.1.py:242`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main4.1.py:248`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main4.1.py:251`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main4.1.py:269`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main4.1.py:273`
- Doc: Verosimilitud con escalas físicas realistas

### ln_bf `def ln_bf(self)`
- Defined: `main4.1.py:288`
- Doc: Factor de Bayes con penalización de complejidad

## main4.py.py

### simulate `def simulate(n_leaves, n_nodes, seed)`
- Defined: `main4.py.py:260`

### dim `def dim(a)`
- Defined: `main4.py.py:52`

### pt `def pt(k, o, c)`
- Defined: `main4.py.py:56`

### size `def size(n)`
- Defined: `main4.py.py:60`

### __post_init__ `def __post_init__(self)`
- Defined: `main4.py.py:75`

### spectral_density `def spectral_density(self, w)`
- Defined: `main4.py.py:79`

### modular_entropy `def modular_entropy(self)`
- Defined: `main4.py.py:82`

### bures_distance `def bures_distance(self, other)`
- Defined: `main4.py.py:88`

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main4.py.py:103`

### _gibbs `def _gibbs(self)`
- Defined: `main4.py.py:111`

### _global `def _global(self)`
- Defined: `main4.py.py:120`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main4.py.py:131`

### _spectral_dim `def _spectral_dim(self, k, n_fit)`
- Defined: `main4.py.py:140`

### _ramsey `def _ramsey(self)`
- Defined: `main4.py.py:176`

### t_c `def t_c(self)`
- Defined: `main4.py.py:190`

### __init__ `def __init__(self, n_modes)`
- Defined: `main4.py.py:199`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main4.py.py:205`

### _loss_potential `def _loss_potential(self)`
- Defined: `main4.py.py:210`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main4.py.py:216`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main4.py.py:219`

### __init__ `def __init__(self, pred_resma, nulls)`
- Defined: `main4.py.py:233`

### log_lik `def log_lik(self, model_pred)`
- Defined: `main4.py.py:237`

### ln_bf `def ln_bf(self)`
- Defined: `main4.py.py:244`

## main5.py

### simulate_resma_multiverse `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)`
- Defined: `main5.py:1052`
- Doc: Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `main5.py:74`
- Doc: α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `main5.py:84`
- Doc: Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `main5.py:95`
- Doc: Límite inferior para conectoma biológico realista

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `main5.py:101`
- Doc: Validar rango físico para dimensión espectral

### validate_percolation_time `def validate_percolation_time(t_c, expected, tolerance)`
- Defined: `main5.py:106`
- Doc: Validar tiempo de percolación contra predicción empírica

### __post_init__ `def __post_init__(self)`
- Defined: `main5.py:127`
- Doc: Validaciones post-construcción (Pilar 3)

### spectral_density `def spectral_density(self, omega)`
- Defined: `main5.py:133`
- Doc: Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.

### modular_entropy `def modular_entropy(self)`
- Defined: `main5.py:143`
- Doc: Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización

### bures_distance `def bures_distance(self, other)`
- Defined: `main5.py:151`

### _spectral_moments `def _spectral_moments(self, n)`
- Defined: `main5.py:163`
- Doc: Momentos espectrales Tr(ρ^k) para k=1..n con regularización

### haagerup_weight `def haagerup_weight(self)`
- Defined: `main5.py:170`
- Doc: Peso de Haagerup para regularización del operador modular

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `main5.py:185`
- Doc: Args:

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `main5.py:203`
- Doc: Genera hojas con gaps espectrales distribuidos exponencialmente

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `main5.py:217`
- Doc: Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta

### _construct_global_state `def _construct_global_state(self)`
- Defined: `main5.py:239`
- Doc: Estado global: mapa de pesos por hoja (no matriz) con regularización

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self)`
- Defined: `main5.py:251`
- Doc: Energía libre de Gibbs para validación termodinámica

### __init__ `def __init__(self, leaf, threshold)`
- Defined: `main5.py:266`

### _compute_holonomy `def _compute_holonomy(self)`
- Defined: `main5.py:272`
- Doc: Defecto de holonomía como variación del gap espectral

### _construct_cptp_map `def _construct_cptp_map(self)`
- Defined: `main5.py:276`
- Doc: Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)

### _local_jump_operator `def _local_jump_operator(self, power)`
- Defined: `main5.py:284`
- Doc: K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.

### apply_branching `def apply_branching(self, state_vector)`
- Defined: `main5.py:300`
- Doc: Aplicar canal CPTP a vector de estado local (dim=2) con normalización

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `main5.py:324`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `main5.py:331`
- Doc: E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior

### _szego_projector `def _szego_projector(self)`
- Defined: `main5.py:335`
- Doc: Proyector P_E en base de Fourier positiva (dim reducida)

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `main5.py:345`
- Doc: Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))

### project `def project(self, state_vector)`
- Defined: `main5.py:361`
- Doc: P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO

### compute_teleological_overlap `def compute_teleological_overlap(self)`
- Defined: `main5.py:396`
- Doc: Calcular overlap teleológico con estado objetivo

### __init__ `def __init__(self, universe, emuna)`
- Defined: `main5.py:412`

### _effective_hamiltonian `def _effective_hamiltonian(self)`
- Defined: `main5.py:419`
- Doc: H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)

### _modular_dissipator `def _modular_dissipator(self, state)`
- Defined: `main5.py:432`
- Doc: L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización

### _nonlinear_term `def _nonlinear_term(self, state)`
- Defined: `main5.py:444`
- Doc: G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo

### _stochastic_term `def _stochastic_term(self, dt)`
- Defined: `main5.py:452`
- Doc: Término estocástico ξ(t) con correlaciones cuánticas

### evolve `def evolve(self, rho0, t_span, n_steps)`
- Defined: `main5.py:459`
- Doc: Integración SDE con Euler-Maruyama y control de paso adaptativo.

### _normalize_density_matrix `def _normalize_density_matrix(self, state)`
- Defined: `main5.py:500`
- Doc: Normalizar matriz densidad y forzar hermiticidad

### _is_physical_state `def _is_physical_state(self, state)`
- Defined: `main5.py:509`
- Doc: Verificar si el estado es físico (hermitiano, traza=1, positivo)

### _correct_non_physical_state `def _correct_non_physical_state(self, state)`
- Defined: `main5.py:523`
- Doc: Corregir estado no físico proyectando en el cono de estados válidos

### __post_init__ `def __post_init__(self)`
- Defined: `main5.py:548`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `main5.py:555`
- Doc: H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `main5.py:561`
- Doc: V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `main5.py:569`
- Doc: Campo escalar masivo para estabilización de Spin(7)

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `main5.py:573`
- Doc: Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos

### coherence_quantum `def coherence_quantum(self)`
- Defined: `main5.py:579`
- Doc: Discordia cuántica aproximada con corrección PT

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `main5.py:610`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `main5.py:625`
- Doc: Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `main5.py:649`
- Doc: d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `main5.py:680`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `main5.py:706`
- Doc: Calcular números de Betti para análisis topológico

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `main5.py:722`
- Doc: Matriz de distancias shortest-path (sparse CSR) para homología

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `main5.py:735`
- Doc: t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25

### is_coherent_subgraph `def is_coherent_subgraph(self, subgraph_nodes)`
- Defined: `main5.py:747`
- Doc: Verificar coherencia: subgrafo > 70% del total

### compute_network_entropy `def compute_network_entropy(self)`
- Defined: `main5.py:751`
- Doc: Entropía de la red basada en distribución de grados

### __init__ `def __init__(self, network, universe)`
- Defined: `main5.py:768`

### compute_entropy_gap `def compute_entropy_gap(self)`
- Defined: `main5.py:772`
- Doc: Δ_S* = ε_c en punto excepcional con corrección de regularización

### compute_pontryagin_number `def compute_pontryagin_number(self)`
- Defined: `main5.py:776`
- Doc: S_top[G] = χ(G)/|V| (número de Euler normalizado)

### compute_freedom `def compute_freedom(self)`
- Defined: `main5.py:792`
- Doc: L[G] = Δ_S* / S_top[G] con protección de división por cero

### is_gauge_invariant `def is_gauge_invariant(self)`
- Defined: `main5.py:803`
- Doc: |L[G] - 1| < 0.05 en estado crítico (invariante de libertad)

### ising_quantum `def ising_quantum(network)`
- Defined: `main5.py:823`
- Doc: Modelo de Ising cuántico transversal en red fractal.

### syk4 `def syk4(network)`
- Defined: `main5.py:843`
- Doc: SYK₄ estándar (sin R-simetría Spin(7) ni E₈).

### random_network `def random_network(network)`
- Defined: `main5.py:860`
- Doc: Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.

### __init__ `def __init__(self, resma, myelin, network, freedom)`
- Defined: `main5.py:883`

### predict_all `def predict_all(self)`
- Defined: `main5.py:891`
- Doc: Predicciones RESMA 4.0 con valores empíricos objetivo

### _predict_diffraction_peak `def _predict_diffraction_peak(self)`
- Defined: `main5.py:906`
- Doc: q₀ = 2π/L_E8 (predicción de difracción UASED)

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `main5.py:912`
- Doc: log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)

### __init__ `def __init__(self, predictions)`
- Defined: `main5.py:981`

### _define_protocols `def _define_protocols(self)`
- Defined: `main5.py:985`
- Doc: Definir protocolos experimentales con parámetros técnicos

### evaluate_feasibility `def evaluate_feasibility(self, budget, time_limit)`
- Defined: `main5.py:1014`
- Doc: Evaluar viabilidad del protocolo completo

### simulate_experimental_outcome `def simulate_experimental_outcome(self, protocol_name)`
- Defined: `main5.py:1027`
- Doc: Simular resultado experimental con ruido realista

## monitor_extremo.py

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `monitor_extremo.py:19`

### generar_datos_toxico `def generar_datos_toxico()`
- Defined: `monitor_extremo.py:103`
- Doc: Genera datos diseñados específicamente para causar colapso

### experimento_colapso_forzado `def experimento_colapso_forzado()`
- Defined: `monitor_extremo.py:126`
- Doc: Experimento diseñado para forzar el colapso del modelo

### generar_graficos_extremos `def generar_graficos_extremos(historial)`
- Defined: `monitor_extremo.py:339`
- Doc: Genera gráficos del experimento extremo

### __init__ `def __init__(self, epsilon_c)`
- Defined: `monitor_extremo.py:28`

### calcular_libertad `def calcular_libertad(self, weights)`
- Defined: `monitor_extremo.py:31`
- Doc: Calcula la métrica L (libertad) de una matriz de pesos

### evaluar_regimen `def evaluar_regimen(self, L)`
- Defined: `monitor_extremo.py:63`
- Doc: Evalúa el régimen del modelo

### __init__ `def __init__(self)`
- Defined: `monitor_extremo.py:74`

### forward `def forward(self, x)`
- Defined: `monitor_extremo.py:88`

### get_linear_layers `def get_linear_layers(self)`
- Defined: `monitor_extremo.py:100`

## quick_monitor.py

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `quick_monitor.py:19`

### generar_datos_mnist_rapido `def generar_datos_mnist_rapido()`
- Defined: `quick_monitor.py:95`
- Doc: Genera datos sintéticos tipo MNIST para experimento rápido

### entrenar_modelo_rapido `def entrenar_modelo_rapido()`
- Defined: `quick_monitor.py:116`
- Doc: Entrena modelo con monitoreo L en tiempo real

### generar_graficos_rapido `def generar_graficos_rapido(historial)`
- Defined: `quick_monitor.py:295`
- Doc: Genera gráficos de resultados del experimento rápido

### __init__ `def __init__(self, epsilon_c)`
- Defined: `quick_monitor.py:28`

### calcular_libertad `def calcular_libertad(self, weights)`
- Defined: `quick_monitor.py:31`
- Doc: Calcula la métrica L (libertad) de una matriz de pesos

### evaluar_regimen `def evaluar_regimen(self, L)`
- Defined: `quick_monitor.py:63`
- Doc: Evalúa el régimen del modelo

### __init__ `def __init__(self)`
- Defined: `quick_monitor.py:74`

### forward `def forward(self, x)`
- Defined: `quick_monitor.py:83`

### get_linear_layers `def get_linear_layers(self)`
- Defined: `quick_monitor.py:92`

## resma2/main_experiment.py

### set_seed `def set_seed(seed)`
- Defined: `resma2/main_experiment.py:26`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

### run_experiment `def run_experiment()`
- Defined: `resma2/main_experiment.py:31`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/main_experiments.py

### inject_noise `def inject_noise(x, sigma)`
- Defined: `resma2/main_experiments.py:24`
- Depends on: `resma2/monitor.py`, `resma2/resma_core.py`, `resma2/resma_observer.py`

### train_epoch `def train_epoch(model, loader, optim, obs, epoch)`
- Defined: `resma2/main_experiments.py:27`
- Depends on: `resma2/monitor.py`, `resma2/resma_core.py`, `resma2/resma_observer.py`

### run `def run()`
- Defined: `resma2/main_experiments.py:61`
- Depends on: `resma2/monitor.py`, `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/monitor.py

### __init__ `def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_layers, verbose)`
- Defined: `resma2/monitor.py:36`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

### _extract_weights `def _extract_weights(self, model)`
- Defined: `resma2/monitor.py:50`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

### _calculate_svd_metrics `def _calculate_svd_metrics(self, weight_matrix)`
- Defined: `resma2/monitor.py:58`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

### calcular_libertad `def calcular_libertad(self, weights)`
- Defined: `resma2/monitor.py:84`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

### calculate `def calculate(self, model)`
- Defined: `resma2/monitor.py:92`
- Imported by: `resma2/main_experiments.py`, `resma2/resma_observer.py`

## resma2/resma_app_mnist.py

### add_quantum_noise `def add_quantum_noise(tensor, noise_factor)`
- Defined: `resma2/resma_app_mnist.py:25`
- Doc: Inyecta ruido gaussiano simulando fluctuaciones de vacío
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

### train `def train(model, device, train_loader, optimizer, epoch, observer)`
- Defined: `resma2/resma_app_mnist.py:30`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

### main `def main()`
- Defined: `resma2/resma_app_mnist.py:65`
- Depends on: `resma2/resma_core.py`, `resma2/resma_observer.py`

## resma2/resma_breakpoint.py

### find_break_point `def find_break_point()`
- Defined: `resma2/resma_breakpoint.py:6`
- Depends on: `resma2/resma_core.py`

## resma2/resma_combat_test.py

### combat_test `def combat_test()`
- Defined: `resma2/resma_combat_test.py:11`
- Depends on: `resma2/resma_core.py`

## resma2/resma_core.py

### __init__ `def __init__(self, omega, chi, kappa_init)`
- Defined: `resma2/resma_core.py:15`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### forward `def forward(self, x)`
- Defined: `resma2/resma_core.py:26`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### __init__ `def __init__(self, in_features, out_features, q_order)`
- Defined: `resma2/resma_core.py:37`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### _generate_ramsey_mask `def _generate_ramsey_mask(self)`
- Defined: `resma2/resma_core.py:47`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### forward `def forward(self, x)`
- Defined: `resma2/resma_core.py:59`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### __init__ `def __init__(self, input_dim, hidden_dim, output_dim)`
- Defined: `resma2/resma_core.py:66`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

### forward `def forward(self, x)`
- Defined: `resma2/resma_core.py:74`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`, `resma2/resma_breakpoint.py`, `resma2/resma_combat_test.py`, `resma2/resma_noise_phase_test.py`, `resma2/resma_overload.py`, `resma2/resma_train.py`, `resma2/resma_vision.py`, `resma2/resma_vision_trained.py`

## resma2/resma_noise_phase_test.py

### add_noise `def add_noise(x, sigma)`
- Defined: `resma2/resma_noise_phase_test.py:34`
- Depends on: `resma2/resma_core.py`

### measure_entropy `def measure_entropy(gate_tensor)`
- Defined: `resma2/resma_noise_phase_test.py:37`
- Depends on: `resma2/resma_core.py`

## resma2/resma_observer.py

### to_dict `def to_dict(self)`
- Defined: `resma2/resma_observer.py:36`
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### __init__ `def __init__(self, model, epsilon_c)`
- Defined: `resma2/resma_observer.py:40`
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### _register_hooks `def _register_hooks(self)`
- Defined: `resma2/resma_observer.py:51`
- Doc: Inyecta sondas en las capas PT para leer telemetría en tiempo real
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### step `def step(self, epoch)`
- Defined: `resma2/resma_observer.py:68`
- Doc: Ejecutar al final de cada época de entrenamiento/validación.
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### report `def report(self, state)`
- Defined: `resma2/resma_observer.py:106`
- Doc: Imprime reporte formateado a consola
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### plot_phase_space `def plot_phase_space(self, save_path)`
- Defined: `resma2/resma_observer.py:118`
- Doc: Genera el diagrama de fase: Estructura vs Dinámica
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

### hook_fn `def hook_fn(module, input, output)`
- Defined: `resma2/resma_observer.py:53`
- Depends on: `resma2/monitor.py`
- Imported by: `resma2/main_experiment.py`, `resma2/main_experiments.py`, `resma2/resma_app_mnist.py`

## resma2/resma_overload.py

### overload_test `def overload_test()`
- Defined: `resma2/resma_overload.py:6`
- Depends on: `resma2/resma_core.py`

## resma2/resma_vision.py

### add_noise `def add_noise(tensor, factor)`
- Defined: `resma2/resma_vision.py:11`
- Depends on: `resma2/resma_core.py`

### visualize_resma_perception `def visualize_resma_perception()`
- Defined: `resma2/resma_vision.py:14`
- Depends on: `resma2/resma_core.py`

## resma2/resma_vision_trained.py

### add_noise `def add_noise(tensor, factor)`
- Defined: `resma2/resma_vision_trained.py:11`
- Depends on: `resma2/resma_core.py`

### visualize_trained_perception `def visualize_trained_perception()`
- Defined: `resma2/resma_vision_trained.py:14`
- Depends on: `resma2/resma_core.py`

## resma4.10.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.10.py:877`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.10.py:935`

### _make_serializable `def _make_serializable(obj, depth, max_depth, _visited)`
- Defined: `resma4.10.py:963`
- Doc: Convierte objetos a formato serializable de forma segura.

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.10.py:1067`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.10.py:51`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.10.py:75`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.10.py:87`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.10.py:90`

### to_dict `def to_dict(self)`
- Defined: `resma4.10.py:93`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.10.py:103`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.10.py:118`

### _generate_e8_roots `def _generate_e8_roots()`
- Defined: `resma4.10.py:149`
- Doc: Genera las 240 raíces de E8 en R⁸.

### _idx `def _idx(self, root_vec)`
- Defined: `resma4.10.py:189`
- Doc: Índice global (0..239) de una raíz.

### _compute_structure_constants `def _compute_structure_constants(self)`
- Defined: `resma4.10.py:197`
- Doc: Constantes N_{α,β} para toda raíz α, β con α+β también raíz.

### _adjoint_matrix `def _adjoint_matrix(self, cartan, roots_coeff)`
- Defined: `resma4.10.py:238`
- Doc: Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.

### _construir_generadores_e8 `def _construir_generadores_e8(self)`
- Defined: `resma4.10.py:330`
- Doc: Construye 3 generadores genuinos del álgebra E8 en la adjunta.

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.10.py:399`

### operator `def operator(self)`
- Defined: `resma4.10.py:408`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.10.py:423`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.10.py:432`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.10.py:436`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.10.py:443`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.10.py:466`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.10.py:470`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.10.py:476`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.10.py:507`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.10.py:543`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.10.py:554`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.10.py:583`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.10.py:594`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.10.py:603`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.10.py:606`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.10.py:615`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.10.py:653`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.10.py:727`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.10.py:735`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.10.py:757`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.10.py:761`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.10.py:769`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.10.py:783`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.10.py:800`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.10.py:805`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.10.py:811`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.10.py:819`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.10.py:824`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.10.py:867`

### log_resources `def log_resources()`
- Defined: `resma4.10.py:872`

## resma4.13.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.13.py:902`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.13.py:960`

### _make_serializable `def _make_serializable(obj, depth, max_depth, _visited)`
- Defined: `resma4.13.py:988`
- Doc: Convierte objetos a formato serializable de forma segura.

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.13.py:1092`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.13.py:51`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.13.py:75`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.13.py:87`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.13.py:90`

### to_dict `def to_dict(self)`
- Defined: `resma4.13.py:93`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.13.py:103`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.13.py:118`

### _generate_e8_roots `def _generate_e8_roots()`
- Defined: `resma4.13.py:149`
- Doc: Genera las 240 raíces de E8 en R⁸.

### _idx `def _idx(self, root_vec)`
- Defined: `resma4.13.py:189`
- Doc: Índice global (0..239) de una raíz.

### _compute_structure_constants `def _compute_structure_constants(self)`
- Defined: `resma4.13.py:197`
- Doc: Constantes N_{α,β} para toda raíz α, β con α+β también raíz.

### _adjoint_matrix `def _adjoint_matrix(self, cartan, roots_coeff)`
- Defined: `resma4.13.py:238`
- Doc: Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.

### _construir_generadores_e8 `def _construir_generadores_e8(self)`
- Defined: `resma4.13.py:330`
- Doc: Construye 3 generadores genuinos del álgebra E8 en la adjunta.

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.13.py:399`

### operator `def operator(self)`
- Defined: `resma4.13.py:408`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.13.py:423`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.13.py:432`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.13.py:436`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.13.py:443`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.13.py:466`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.13.py:470`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.13.py:476`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.13.py:507`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.13.py:543`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.13.py:553`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.13.py:608`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.13.py:619`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.13.py:628`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.13.py:631`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.13.py:640`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.13.py:678`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.13.py:752`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.13.py:760`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.13.py:782`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.13.py:786`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.13.py:794`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.13.py:808`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.13.py:825`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.13.py:830`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.13.py:836`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.13.py:844`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.13.py:849`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.13.py:892`

### log_resources `def log_resources()`
- Defined: `resma4.13.py:897`

## resma4.2.py

### simulate_resma_complete `def simulate_resma_complete(n_leaves, n_nodes, seed)`
- Defined: `resma4.2.py:556`
- Doc: Pipeline RESMA 4.2 completo

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.2.py:67`
- Doc: Verificar condición PT: κ < χΩ

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.2.py:80`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.2.py:88`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.2.py:96`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.2.py:101`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.2.py:117`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.2.py:121`
- Doc: ρ(ω) con regularización UV

### modular_entropy `def modular_entropy(self)`
- Defined: `resma4.2.py:126`
- Doc: S = -∫ ρ log ρ dω

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.2.py:136`
- Doc: Distancia de Bures W₂(ρ₁, ρ₂)

### haagerup_weight `def haagerup_weight(self)`
- Defined: `resma4.2.py:154`
- Doc: Peso de Haagerup para regularización

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `resma4.2.py:165`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.2.py:176`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.2.py:189`
- Doc: μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.2.py:205`
- Doc: Estado global: pesos por hoja

### compute_gibbs_free_energy `def compute_gibbs_free_energy(self)`
- Defined: `resma4.2.py:216`
- Doc: F = -ln(Tr(μ)) / β

### __init__ `def __init__(self, universe, n_samples)`
- Defined: `resma4.2.py:227`

### _construct_hardy_state `def _construct_hardy_state(self)`
- Defined: `resma4.2.py:234`
- Doc: E(z) ∈ H²(ℂ⁺)

### _szego_projector `def _szego_projector(self)`
- Defined: `resma4.2.py:238`
- Doc: Proyector en frecuencias positivas

### _evaluation_functional `def _evaluation_functional(self, state_weights)`
- Defined: `resma4.2.py:246`
- Doc: Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)

### project `def project(self, state_vector)`
- Defined: `resma4.2.py:258`
- Doc: P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.2.py:297`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.2.py:304`
- Doc: H₀: dispersión Ω(q) = Ω₀ + q² + χq³

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.2.py:310`
- Doc: V_loss ∝ (r/a₀)^(2α)

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.2.py:317`
- Doc: Campo escalar para estabilización Spin(7)

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.2.py:321`
- Doc: κ < χΩ

### coherence_quantum `def coherence_quantum(self)`
- Defined: `resma4.2.py:327`
- Doc: Coherencia cuántica con verificación espectral

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `resma4.2.py:354`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.2.py:366`
- Doc: Scale-free → NO DIRIGIDO

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.2.py:381`
- Doc: d_s = -2 lim log N(λ)/log λ

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.2.py:410`
- Doc: R_Q(G) = min{n | β_{n-1}(G) > 0}

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.2.py:429`
- Doc: Números de Betti β₀, β₁

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.2.py:445`
- Doc: Matriz de distancias para homología

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `resma4.2.py:461`
- Doc: t_c = 21 · (N/N₀)^0.25 / log R_Q

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.2.py:477`

### predict_all `def predict_all(self)`
- Defined: `resma4.2.py:483`
- Doc: Predicciones RESMA 4.2

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.2.py:496`
- Doc: ln(BF) con AIC

## resma4.3.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.3.py:54`
- Doc: Guardado atómico con backup

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.3.py:84`
- Doc: Cargar checkpoint con fallback

### simulate_resma_with_checkpointing `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)`
- Defined: `resma4.3.py:545`
- Doc: Pipeline con reanudación inteligente desde checkpoints

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.3.py:35`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.3.py:40`

### log_resources `def log_resources()`
- Defined: `resma4.3.py:49`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.3.py:127`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.3.py:150`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.3.py:154`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.3.py:158`
- Doc: Distancia Bures con caché EXTERNO (no en instancia)

### __init__ `def __init__(self, n_leaves, seed)`
- Defined: `resma4.3.py:193`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.3.py:215`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.3.py:227`
- Doc: Matriz de medida con guardado incremental

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.3.py:254`

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.3.py:271`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.3.py:279`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.3.py:287`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.3.py:292`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.3.py:302`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.3.py:312`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.3.py:317`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.3.py:323`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.3.py:326`

### coherence_quantum `def coherence_quantum(self)`
- Defined: `resma4.3.py:331`

### __init__ `def __init__(self, n_nodes, seed)`
- Defined: `resma4.3.py:353`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.3.py:372`
- Doc: Generar grafo por lotes

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.3.py:401`
- Doc: Dimensión espectral con matriz sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.3.py:425`
- Doc: Ramsey topológico

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.3.py:444`
- Doc: Números de Betti

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.3.py:460`
- Doc: Matriz de distancias sparse

### critical_percolation_time `def critical_percolation_time(self)`
- Defined: `resma4.3.py:476`
- Doc: Tiempo crítico de percolación

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.3.py:486`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.3.py:492`
- Doc: ln(BF)

## resma4.4.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.4.py:59`
- Doc: Guarda el estado COMPLETO de los objetos, no solo metadatos

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.4.py:93`
- Doc: Carga el estado COMPLETO desde disco

### simulate_resma_with_checkpointing `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)`
- Defined: `resma4.4.py:554`
- Doc: Pipeline con reanudación que realmente carga objetos

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.4.py:36`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.4.py:41`

### log_resources `def log_resources()`
- Defined: `resma4.4.py:50`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.4.py:146`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.4.py:168`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.4.py:172`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.4.py:176`
- Doc: Distancia Bures con caché externo

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state)`
- Defined: `resma4.4.py:206`
- Doc: Constructor que puede recibir estado serializado

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.4.py:258`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.4.py:269`
- Doc: Matriz de medida

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.4.py:301`

### validate_dimension `def validate_dimension(alpha, tolerance)`
- Defined: `resma4.4.py:318`

### validate_pt_symmetry `def validate_pt_symmetry(kappa, Omega, chi)`
- Defined: `resma4.4.py:326`

### validate_connectome_size `def validate_connectome_size(n_nodes)`
- Defined: `resma4.4.py:334`

### validate_spectral_dimension `def validate_spectral_dimension(dim)`
- Defined: `resma4.4.py:339`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.4.py:348`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.4.py:358`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.4.py:363`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.4.py:369`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.4.py:372`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)`
- Defined: `resma4.4.py:378`
- Doc: Constructor que puede recibir grafo ya construido

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.4.py:446`
- Doc: Generar grafo por lotes

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.4.py:475`
- Doc: Dimensión espectral con eigenvalores sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.4.py:499`
- Doc: Ramsey topológico

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.4.py:518`
- Doc: Números de Betti

### _graph_to_distance_matrix `def _graph_to_distance_matrix(self)`
- Defined: `resma4.4.py:534`
- Doc: Matriz de distancias sparse

## resma4.5.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.5.py:59`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.5.py:88`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.5.py:113`
- Doc: Convierte objetos a formato serializable

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.5.py:695`
- Doc: Pipeline único con Garnier integrado

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.5.py:36`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.5.py:41`

### log_resources `def log_resources()`
- Defined: `resma4.5.py:50`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.5.py:152`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.5.py:170`

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.5.py:181`
- Doc: Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.5.py:185`
- Doc: Entropía crítica de percolación (ADIMENSIONAL).

### to_dict `def to_dict(self)`
- Defined: `resma4.5.py:192`
- Doc: Para serialización

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.5.py:197`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.5.py:206`

### _construir_generadores_E8 `def _construir_generadores_E8(self)`
- Defined: `resma4.5.py:214`
- Doc: Construye 3 generadores temporales (antis-Hermitianos)

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.5.py:226`
- Doc: Operador de Hadamard en dimensión 248 (unitario)

### operator `def operator(self)`
- Defined: `resma4.5.py:235`
- Doc: Construye D̂_G(ϕ) dimensionalmente consistente

### aplicar_a_estado `def aplicar_a_estado(self, estado)`
- Defined: `resma4.5.py:254`
- Doc: Aplica desdoblamiento a un estado cuántico |Ψ⟩

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.5.py:260`
- Doc: α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))

### __init__ `def __init__(self, garnier, network)`
- Defined: `resma4.5.py:273`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red)`
- Defined: `resma4.5.py:278`
- Doc: ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)

### _calcular_rho_reducida_aproximada `def _calcular_rho_reducida_aproximada(self)`
- Defined: `resma4.5.py:299`
- Doc: Aproximación: ρ_red = diag(grados) / sum(grados)

### es_silencio_activo `def es_silencio_activo(self, rho_red)`
- Defined: `resma4.5.py:307`
- Doc: Verifica Silencio-Activo y calcula Libertad L.

### umbral_percolacion `def umbral_percolacion(self)`
- Defined: `resma4.5.py:324`
- Doc: Umbral de percolación para soberanía: 70% (Axioma 6)

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.5.py:345`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.5.py:349`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.5.py:353`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.5.py:383`
- Doc: Constructor que puede recibir estado serializado

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.5.py:420`

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.5.py:431`
- Doc: Matriz de medida sin desdoblamiento

### _aplicar_desdoblamiento_a_medida `def _aplicar_desdoblamiento_a_medida(self, measure)`
- Defined: `resma4.5.py:451`
- Doc: Aplica D̂_G(ϕ) a la medida:

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.5.py:471`

### _calcular_libertad_universo `def _calcular_libertad_universo(self)`
- Defined: `resma4.5.py:481`
- Doc: Libertad del universo: L = 1/ε_c

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.5.py:488`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.5.py:499`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.5.py:504`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.5.py:510`

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.5.py:513`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.5.py:520`
- Doc: Constructor que puede recibir grafo ya construido

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.5.py:559`
- Doc: Generar grafo por lotes con conectividad controlada

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.5.py:591`
- Doc: Dimensión espectral con eigenvalores sparse

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.5.py:615`
- Doc: Ramsey topológico simplificado

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.5.py:627`
- Doc: Números de Betti aproximados por ciclos locales

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.5.py:636`
- Doc: Matriz densidad reducida del conectoma

### validar_axioma_6 `def validar_axioma_6(self)`
- Defined: `resma4.5.py:644`
- Doc: Verifica: conectividad > 70% para soberanía

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.5.py:661`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.5.py:666`
- Doc: Calcula Factor de Bayes integrando Garnier

## resma4.6.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.6.py:56`
- Doc: Guarda estado completo con manejo robusto de errores

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.6.py:86`
- Doc: Carga checkpoint con fallback automático

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.6.py:112`
- Doc: Convierte objetos recursivamente a formato serializable

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)`
- Defined: `resma4.6.py:906`
- Doc: Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.6.py:33`

### check_memory_limit `def check_memory_limit(threshold)`
- Defined: `resma4.6.py:38`

### log_resources `def log_resources()`
- Defined: `resma4.6.py:47`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.6.py:158`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.6.py:181`

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.6.py:194`
- Doc: Factor de escala con supresión ZPE

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.6.py:200`
- Doc: **UMBRAL CRÍTICO CON ZPE**:

### to_dict `def to_dict(self)`
- Defined: `resma4.6.py:208`
- Doc: Serialización completa

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.6.py:220`
- Doc: Deserialización

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.6.py:238`

### _construir_generadores_E8_ZPE `def _construir_generadores_E8_ZPE(self)`
- Defined: `resma4.6.py:246`
- Doc: GENERADORES CON CANCELACIÓN ZPE INTEGRADA

### _hadamard_generalizado_ZPE `def _hadamard_generalizado_ZPE(self)`
- Defined: `resma4.6.py:266`
- Doc: HADAMARD CON ESPACIO NULO ZPE

### operator `def operator(self)`
- Defined: `resma4.6.py:284`
- Doc: Construye D̂_G(ϕ) con cancelación ZPE

### alpha_modificado `def alpha_modificado(self, alpha_base)`
- Defined: `resma4.6.py:304`
- Doc: **α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**

### __init__ `def __init__(self, garnier, network)`
- Defined: `resma4.6.py:318`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red)`
- Defined: `resma4.6.py:327`
- Doc: **ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**

### _calcular_rho_reducida_aproximada `def _calcular_rho_reducida_aproximada(self)`
- Defined: `resma4.6.py:367`
- Doc: Matriz densidad con modulación ZPE

### es_silencio_activo `def es_silencio_activo(self, rho_red)`
- Defined: `resma4.6.py:383`
- Doc: **DETECCIÓN DE ANTAGONISMO**:

### umbral_percolacion `def umbral_percolacion(self)`
- Defined: `resma4.6.py:411`
- Doc: Umbral para soberanía: 70%

### modo_goldstone `def modo_goldstone(self)`
- Defined: `resma4.6.py:415`
- Doc: **MODO GOLDSTONE DEL DOBLE CUÁNTICO**:

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.6.py:449`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.6.py:453`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.6.py:457`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.6.py:487`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.6.py:521`
- Doc: Inicializa hojas con temperatura efectiva afectada por ZPE

### _generate_gibbs_measure `def _generate_gibbs_measure(self)`
- Defined: `resma4.6.py:535`
- Doc: Genera medida de Gibbs

### _aplicar_desdoblamiento_a_medida `def _aplicar_desdoblamiento_a_medida(self, measure)`
- Defined: `resma4.6.py:557`
- Doc: Aplica desdoblamiento con supresión ZPE

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.6.py:584`
- Doc: Construye estado global normalizado

### _calcular_libertad_universo `def _calcular_libertad_universo(self)`
- Defined: `resma4.6.py:603`
- Doc: Libertad intrínseca con supresión ZPE

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.6.py:611`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.6.py:624`
- Doc: Hamiltoniano con energía ZPE incluida

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.6.py:633`
- Doc: Potencial de pérdida PT

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.6.py:640`

### _calcular_zpe `def _calcular_zpe(self)`
- Defined: `resma4.6.py:643`
- Doc: **ENERGÍA DE PUNTO CERO TOTAL**:

### _pt_symmetry_condition `def _pt_symmetry_condition(self)`
- Defined: `resma4.6.py:655`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.6.py:662`

### _generate_fractal_graph `def _generate_fractal_graph(self)`
- Defined: `resma4.6.py:711`
- Doc: Genera grafo con densidad 0.75 (conectoma humano)

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.6.py:753`
- Doc: Dimensión espectral con ZPE

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.6.py:777`
- Doc: Ramsey topológico

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.6.py:788`
- Doc: Números de Betti reales

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.6.py:804`
- Doc: Matriz densidad con supresión ZPE

### _calcular_zpe_conectoma `def _calcular_zpe_conectoma(self)`
- Defined: `resma4.6.py:820`
- Doc: **ENERGÍA ZPE DEL CONECTOMA**:

### validar_axioma_6 `def validar_axioma_6(self)`
- Defined: `resma4.6.py:836`
- Doc: **AXIOMA 6**: Conectividad > 70% para soberanía

### __init__ `def __init__(self, universe, myelin, network)`
- Defined: `resma4.6.py:856`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.6.py:861`
- Doc: Calcula Factor de Bayes con antagonismo ZPE-Silencio

## resma4.7.py

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed)`
- Defined: `resma4.7.py:537`
- Doc: Pipeline completo RESMA-Garnier con correcciones

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.7.py:53`

### _compute_coupling `def _compute_coupling(self)`
- Defined: `resma4.7.py:67`
- Doc: Fuerza de acoplamiento entre tiempos

### factor_escala `def factor_escala(self, tiempo_idx)`
- Defined: `resma4.7.py:72`
- Doc: Factor de escala temporal

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.7.py:77`
- Doc: Entropía crítica con corrección de acoplamiento:

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.7.py:85`
- Doc: Factor de modulación para la medida cuántica:

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.7.py:98`

### _construir_generadores `def _construir_generadores(self)`
- Defined: `resma4.7.py:103`
- Doc: Generadores temporales (anti-Hermitianos normalizados)

### operator `def operator(self)`
- Defined: `resma4.7.py:115`
- Doc: Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)

### aplicar_modulacion `def aplicar_modulacion(self, state_vector)`
- Defined: `resma4.7.py:120`
- Doc: Aplica desdoblamiento a vector de estado

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.7.py:126`
- Doc: α'(φ) = α · |cos(φ₃)|^(C0/C3)

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.7.py:143`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.7.py:147`
- Doc: ΔS_loop = S_vN(ρ) - log(b₁ + 1)

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.7.py:166`
- Doc: Verifica condición y calcula libertad L = 1/(ΔS + ε_c)

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.7.py:197`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.7.py:203`
- Doc: Distancia de Bures simplificada

### __init__ `def __init__(self, n_leaves, seed, garnier)`
- Defined: `resma4.7.py:226`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.7.py:252`
- Doc: Genera hojas con gaps distribuidos exponencialmente

### _generate_modulated_measure `def _generate_modulated_measure(self)`
- Defined: `resma4.7.py:265`
- Doc: Genera medida de transición modulada por Garnier:

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.7.py:312`
- Doc: Estado global como distribución diagonal

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.7.py:322`
- Doc: Libertad del universo: L_U = 1/ε_c

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.7.py:326`
- Doc: Coherencia cuántica: suma de elementos off-diagonal

### __init__ `def __init__(self, n_nodes, seed, garnier)`
- Defined: `resma4.7.py:341`

### _generate_realistic_network `def _generate_realistic_network(self)`
- Defined: `resma4.7.py:379`
- Doc: Genera red con conectividad > 70% usando modelo realista:

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.7.py:424`
- Doc: Números de Betti: b0=componentes, b1=ciclos

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.7.py:433`
- Doc: Dimensión espectral del Laplaciano

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.7.py:455`
- Doc: Número de Ramsey topológico

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.7.py:460`
- Doc: Matriz densidad de la red (normalizada por grados)

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.7.py:470`
- Doc: Verifica conectividad > 70%

### __init__ `def __init__(self, universe, network)`
- Defined: `resma4.7.py:487`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.7.py:491`
- Doc: ln(BF) ∝ log(L_red · L_univ)

## resma4.8.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.8.py:91`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.8.py:117`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.8.py:141`

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.8.py:667`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.8.py:54`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.8.py:72`

### check_memory_limit `def check_memory_limit()`
- Defined: `resma4.8.py:77`

### log_resources `def log_resources()`
- Defined: `resma4.8.py:86`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.8.py:161`

### _compute_coupling `def _compute_coupling(self)`
- Defined: `resma4.8.py:179`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.8.py:182`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.8.py:186`

### to_dict `def to_dict(self)`
- Defined: `resma4.8.py:189`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.8.py:199`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.8.py:210`

### _construir_generadores_aleatorios `def _construir_generadores_aleatorios(self)`
- Defined: `resma4.8.py:219`

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.8.py:228`

### operator `def operator(self)`
- Defined: `resma4.8.py:233`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.8.py:248`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.8.py:257`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.8.py:261`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.8.py:268`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.8.py:291`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.8.py:295`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.8.py:301`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.8.py:332`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.8.py:362`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.8.py:373`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.8.py:402`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.8.py:413`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.8.py:422`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.8.py:425`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.8.py:434`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.8.py:472`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.8.py:529`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.8.py:537`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.8.py:559`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.8.py:563`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.8.py:571`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.8.py:585`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.8.py:602`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.8.py:607`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.8.py:613`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.8.py:621`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.8.py:626`

## resma4.9.py

### guardar_checkpoint `def guardar_checkpoint(data, filename)`
- Defined: `resma4.9.py:587`

### cargar_checkpoint `def cargar_checkpoint(filename)`
- Defined: `resma4.9.py:615`

### _make_serializable `def _make_serializable(obj)`
- Defined: `resma4.9.py:639`

### simulate_resma_garnier `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- Defined: `resma4.9.py:655`

### verify_pt_condition `def verify_pt_condition(cls)`
- Defined: `resma4.9.py:50`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.9.py:74`

### epsilon_critico `def epsilon_critico(self)`
- Defined: `resma4.9.py:86`

### modulation_factor `def modulation_factor(self)`
- Defined: `resma4.9.py:89`

### to_dict `def to_dict(self)`
- Defined: `resma4.9.py:92`

### from_dict `def from_dict(cls, data)`
- Defined: `resma4.9.py:102`

### __init__ `def __init__(self, garnier, dimension)`
- Defined: `resma4.9.py:112`

### _construir_generadores_aleatorios `def _construir_generadores_aleatorios(self)`
- Defined: `resma4.9.py:121`

### _hadamard_generalizado `def _hadamard_generalizado(self)`
- Defined: `resma4.9.py:130`

### operator `def operator(self)`
- Defined: `resma4.9.py:135`

### calcular_alpha_modificado `def calcular_alpha_modificado(self, alpha_base)`
- Defined: `resma4.9.py:150`

### __init__ `def __init__(self, garnier)`
- Defined: `resma4.9.py:159`

### calcular_delta_s_loop `def calcular_delta_s_loop(self, rho_red, b1)`
- Defined: `resma4.9.py:163`

### es_silencio_activo `def es_silencio_activo(self, rho_red, b1)`
- Defined: `resma4.9.py:170`

### __post_init__ `def __post_init__(self)`
- Defined: `resma4.9.py:193`

### spectral_density `def spectral_density(self, omega)`
- Defined: `resma4.9.py:197`

### bures_distance `def bures_distance(self, other)`
- Defined: `resma4.9.py:203`

### __init__ `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- Defined: `resma4.9.py:234`

### _initialize_leaves `def _initialize_leaves(self)`
- Defined: `resma4.9.py:270`

### _generate_complete_measure `def _generate_complete_measure(self)`
- Defined: `resma4.9.py:281`

### _aplicar_modulacion_garnier `def _aplicar_modulacion_garnier(self, measure)`
- Defined: `resma4.9.py:310`

### _construct_global_state `def _construct_global_state(self)`
- Defined: `resma4.9.py:321`

### _calcular_libertad `def _calcular_libertad(self)`
- Defined: `resma4.9.py:330`

### _calcular_coherencia `def _calcular_coherencia(self)`
- Defined: `resma4.9.py:333`

### __init__ `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- Defined: `resma4.9.py:342`

### _generate_realistic_modular_network `def _generate_realistic_modular_network(self)`
- Defined: `resma4.9.py:380`

### _compute_betti_numbers `def _compute_betti_numbers(self)`
- Defined: `resma4.9.py:437`

### _spectral_dimension `def _spectral_dimension(self)`
- Defined: `resma4.9.py:445`

### _topological_ramsey `def _topological_ramsey(self)`
- Defined: `resma4.9.py:467`

### _calcular_rho_reducida `def _calcular_rho_reducida(self)`
- Defined: `resma4.9.py:471`

### _validar_axioma_6 `def _validar_axioma_6(self)`
- Defined: `resma4.9.py:479`

### __init__ `def __init__(self, axon_length, radius, n_modes)`
- Defined: `resma4.9.py:493`

### _free_hamiltonian `def _free_hamiltonian(self)`
- Defined: `resma4.9.py:510`

### _loss_potential `def _loss_potential(self)`
- Defined: `resma4.9.py:515`

### _compute_scalar_mass `def _compute_scalar_mass(self)`
- Defined: `resma4.9.py:521`

### __init__ `def __init__(self, universe, network, myelin)`
- Defined: `resma4.9.py:529`

### compute_log_bayes_factor `def compute_log_bayes_factor(self)`
- Defined: `resma4.9.py:534`

### get_memory_gb `def get_memory_gb()`
- Defined: `resma4.9.py:577`

### log_resources `def log_resources()`
- Defined: `resma4.9.py:582`

## sovereignty_monitor.py

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `sovereignty_monitor.py:21`
- Doc: Setup matplotlib para visualización

### cargar_datos `def cargar_datos()`
- Defined: `sovereignty_monitor.py:126`
- Doc: Carga y prepara el dataset MNIST

### main `def main()`
- Defined: `sovereignty_monitor.py:433`
- Doc: Función principal

### __init__ `def __init__(self, epsilon_c)`
- Defined: `sovereignty_monitor.py:35`

### calcular_libertad `def calcular_libertad(self, weights)`
- Defined: `sovereignty_monitor.py:38`
- Doc: Calcula la métrica L (libertad) de una matriz de pesos

### evaluar_regimen `def evaluar_regimen(self, L)`
- Defined: `sovereignty_monitor.py:85`
- Doc: Evalúa el régimen del modelo

### __init__ `def __init__(self)`
- Defined: `sovereignty_monitor.py:96`

### forward `def forward(self, x)`
- Defined: `sovereignty_monitor.py:110`

### get_linear_layers `def get_linear_layers(self)`
- Defined: `sovereignty_monitor.py:122`
- Doc: Retorna todas las capas lineales para monitoreo

### __init__ `def __init__(self, num_epochs)`
- Defined: `sovereignty_monitor.py:149`

### calcular_metricas_sovereignty `def calcular_metricas_sovereignty(self)`
- Defined: `sovereignty_monitor.py:189`
- Doc: Calcula métricas L para todas las capas lineales

### entrenar_epoca `def entrenar_epoca(self, epoca)`
- Defined: `sovereignty_monitor.py:210`
- Doc: Entrena una época completa

### evaluar_epoca `def evaluar_epoca(self)`
- Defined: `sovereignty_monitor.py:233`
- Doc: Evalúa el modelo en el conjunto de validación

### ejecutar_experimento `def ejecutar_experimento(self)`
- Defined: `sovereignty_monitor.py:253`
- Doc: Ejecuta el experimento completo

### generar_graficos `def generar_graficos(self)`
- Defined: `sovereignty_monitor.py:362`
- Doc: Genera gráficos comprehensivos de resultados

## test_simple.py

### test_basic_math `def test_basic_math()`
- Defined: `test_simple.py:8`
- Doc: Test de las matemáticas básicas RESMA

## train_mini_resma.py

### main `def main()`
- Defined: `train_mini_resma.py:8`
- Depends on: `garnier_nn.py`

## train_profile.py

### main `def main()`
- Defined: `train_profile.py:9`
- Depends on: `garnier_nn.py`

## visualize_resma.py

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `visualize_resma.py:6`
- Doc: Setup matplotlib and seaborn for plotting with proper configuration.

### diagnosticar_modelo `def diagnosticar_modelo(checkpoint_path)`
- Defined: `visualize_resma.py:30`
- Doc: Cargar y visualizar estado de red entrenada
