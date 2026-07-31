# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 42 | **Total Symbols Extracted:** 905 | **Total Imports:** 404
 | **Resolved Imports:** 18

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:e63a2e6 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Community Analysis](#community-analysis)
6. [Suggested Questions](#suggested-questions)
7. [Hotspot Analysis](#hotspot-analysis)
8. [Change Impact Analysis](#change-impact-analysis)
9. [Suggested Linting Rules](#suggested-linting-rules)
10. [Orphans](#orphans)
11. [Query Recipes](#query-recipes)
12. [Structural Knowledge Map](#structural-knowledge-map)
13. [Code Property Graph](#code-property-graph)
14. [Architecture Reference](#architecture-reference)
    - [PY (41 files)](#py-41-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 42 |
| Total Symbols | 905 |
| Total Imports | 404 |
| Call Edges | 5540 |
| Inheritance Edges | 10 |
| Languages | 2 |
| Avg Symbols/File | 21.5 |
| Avg Imports/File | 9.6 |
| Resolved Imports | 18 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `resma4.4.py` | 20 | 36 | py |
| `resma4.3.py` | 19 | 41 | py |
| `resma4.5.py` | 19 | 57 | py |
| `resma4.10.py` | 18 | 57 | py |
| `resma4.13.py` | 18 | 57 | py |
| `resma4.8.py` | 18 | 55 | py |
| `resma4.6.py` | 17 | 59 | py |
| `main5.py` | 16 | 81 | py |
| `resma4.9.py` | 16 | 53 | py |
| `main.py` | 15 | 63 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 36 |
| testing | 4 |
| infrastructure | 1 |
| presentation | 1 |

### utility

- `app.py` (py, 0 symbols)
- `demo_mini_resma.py` (py, 4 symbols)
- `garnier_nn.py` (py, 11 symbols)
- `install.sh` (sh, 0 symbols)
- `main.py` (py, 63 symbols)
- `main2.py` (py, 63 symbols)
- `main3.py` (py, 30 symbols)
- `main4.1.py` (py, 31 symbols)
- `main4.py.py` (py, 30 symbols)
- `main5.py` (py, 81 symbols)
- `monitor_extremo.py` (py, 12 symbols)
- `main_experiment.py` (py, 2 symbols)
- `main_experiments.py` (py, 3 symbols)
- `monitor.py` (py, 9 symbols)
- `resma_app_mnist.py` (py, 3 symbols)
- *... and 21 more*

### infrastructure

- `difract.py` (py, 1 symbols)

### presentation

- `quick_monitor.py` (py, 12 symbols)

### testing

- `resma_combat_test.py` (py, 1 symbols)
- `resma_noise_phase_test.py` (py, 2 symbols)
- `test_simple.py` (py, 1 symbols)
- `test_ultra_simple.py` (py, 0 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `resma_core.py` | 0.1759 | 0.2707 | 0.2707 | 0.00 | 0.00 |
| 2 | `garnier_nn.py` | 0.1591 | 0.1189 | 0.1189 | 0.00 | 0.82 |
| 3 | `resma_observer.py` | 0.1020 | 0.0714 | 0.0714 | 0.00 | 0.56 |
| 4 | `app.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 5 | `test_simple.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 6 | `visualize_resma.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 7 | `main5.py` | 0.0901 | 0.0000 | 0.0000 | 0.00 | 0.90 |
| 8 | `main.py` | 0.0889 | 0.0000 | 0.0000 | 0.00 | 0.89 |
| 9 | `main2.py` | 0.0889 | 0.0000 | 0.0000 | 0.00 | 0.89 |
| 10 | `resma4.7.py` | 0.0795 | 0.0000 | 0.0000 | 0.00 | 0.79 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `resma_core.py` | 21.0 | | 0.2707 |
| `resma_observer.py` | 8.9 | | 0.0714 |
| `main5.py` | 8.1 | | 0.0000 |
| `garnier_nn.py` | 7.1 | | 0.1189 |
| `main.py` | 6.3 | | 0.0000 |
| `main2.py` | 6.3 | | 0.0000 |
| `main_experiments.py` | 6.3 | | 0.0000 |
| `resma4.6.py` | 5.9 | | 0.0000 |
| `resma4.10.py` | 5.7 | | 0.0000 |
| `resma4.13.py` | 5.7 | | 0.0000 |

---

## Community Analysis

Files grouped by import-based community detection. Cohesion measures how tightly connected each community is internally.

### root (Cohesion: 1.00)

**4 files** in this community:

- `garnier_nn.py` (py, 11 symbols)
- `test_ultra_simple.py` (py, 0 symbols)
- `train_mini_resma.py` (py, 1 symbols)
- `train_profile.py` (py, 1 symbols)

### resma2 (Cohesion: 1.00)

**13 files** in this community:

- `main_experiment.py` (py, 2 symbols)
- `main_experiments.py` (py, 3 symbols)
- `monitor.py` (py, 9 symbols)
- `resma_app_mnist.py` (py, 3 symbols)
- `resma_breakpoint.py` (py, 1 symbols)
- `resma_combat_test.py` (py, 1 symbols)
- `resma_core.py` (py, 10 symbols)
- `resma_noise_phase_test.py` (py, 2 symbols)
- `resma_observer.py` (py, 9 symbols)
- `resma_overload.py` (py, 1 symbols)
- `resma_train.py` (py, 0 symbols)
- `resma_vision.py` (py, 2 symbols)
- `resma_vision_trained.py` (py, 2 symbols)

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does resma_core.py depend on, and what depends on it? (10 connections)
- What does resma_observer.py depend on, and what depends on it? (4 connections)
- What does main5.py depend on, and what depends on it? (0 connections)
- How are the 4 files in 'root' related to each other?
- What is GarnierLayer in demo_mini_resma.py and how is it used?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `resma_core.py` | 0.123 | 0.800 | 0.529 | 10 | 16 |
| `garnier_nn.py` | 0.136 | 0.500 | 0.354 | 11 | 10 |
| `resma_observer.py` | 0.111 | 0.550 | 0.374 | 9 | 11 |
| `app.py` | 0.000 | 0.050 | 0.030 | 0 | 1 |
| `test_simple.py` | 0.012 | 0.100 | 0.065 | 1 | 2 |
| `visualize_resma.py` | 0.025 | 0.350 | 0.220 | 2 | 7 |
| `main5.py` | 1.000 | 0.800 | 0.880 | 81 | 16 |
| `main.py` | 0.778 | 0.750 | 0.761 | 63 | 15 |
| `main2.py` | 0.778 | 0.750 | 0.761 | 63 | 15 |
| `resma4.7.py` | 0.481 | 0.500 | 0.493 | 39 | 10 |
| `resma4.5.py` | 0.704 | 0.950 | 0.852 | 57 | 19 |
| `resma4.10.py` | 0.704 | 0.900 | 0.822 | 57 | 18 |
| `resma4.13.py` | 0.704 | 0.900 | 0.822 | 57 | 18 |
| `resma4.8.py` | 0.679 | 0.900 | 0.812 | 55 | 18 |
| `resma4.6.py` | 0.728 | 0.850 | 0.801 | 59 | 17 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `resma_core.py` | 10 | 0 | 10 |
| `monitor.py` | 2 | 2 | 4 |
| `garnier_nn.py` | 3 | 0 | 3 |
| `resma_observer.py` | 3 | 0 | 3 |
| `app.py` | 0 | 0 | 0 |
| `demo_mini_resma.py` | 0 | 0 | 0 |
| `difract.py` | 0 | 0 | 0 |
| `install.sh` | 0 | 0 | 0 |
| `main.py` | 0 | 0 | 0 |
| `main2.py` | 0 | 0 | 0 |
| `main3.py` | 0 | 0 | 0 |
| `main4.1.py` | 0 | 0 | 0 |
| `main4.py.py` | 0 | 0 | 0 |
| `main5.py` | 0 | 0 | 0 |
| `monitor_extremo.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM002` | warning | Bare except clause catches all exceptions including SystemExit | python | 22 |
| `RM001` | info | Large number of functions in py: 737 total | py | 737 |
| `RM003` | info | Print statement found (consider logging instead) | python | 551 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `resma_core.py` (10 symbols, no doc)
- `difract.py` (1 symbols, no doc)
- `install.sh` (0 symbols, no doc)
- `main_experiment.py` (2 symbols, no doc)
- `main_experiments.py` (3 symbols, no doc)
- `monitor.py` (9 symbols, no doc)
- `resma_breakpoint.py` (1 symbols, no doc)
- `resma_combat_test.py` (1 symbols, no doc)
- `resma_noise_phase_test.py` (2 symbols, no doc)
- `resma_overload.py` (1 symbols, no doc)
- `resma_train.py` (0 symbols, no doc)
- `resma_vision.py` (2 symbols, no doc)
- `resma_vision_trained.py` (2 symbols, no doc)
- `test_ultra_simple.py` (0 symbols, no doc)
- `train_mini_resma.py` (1 symbols, no doc)
- `train_profile.py` (1 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

## Structural Knowledge Map

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    resma4_4_py["resma4.4.py (py)"]
    class resma4_4_py mod;
    resma4_4_py_ResourceMonitor["ResourceMonitor"]
    class resma4_4_py_ResourceMonitor cls;
    resma4_4_py --> resma4_4_py_ResourceMonitor
    resma4_4_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_4_py_guardar_checkpoint fn;
    resma4_4_py --> resma4_4_py_guardar_checkpoint
    resma4_4_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_4_py_cargar_checkpoint fn;
    resma4_4_py --> resma4_4_py_cargar_checkpoint
    resma4_4_py_RESMAConstants["RESMAConstants"]
    class resma4_4_py_RESMAConstants cls;
    resma4_4_py --> resma4_4_py_RESMAConstants
    resma4_4_py_QuantumLeaf["QuantumLeaf"]
    class resma4_4_py_QuantumLeaf cls;
    resma4_4_py --> resma4_4_py_QuantumLeaf
    resma4_5_py["resma4.5.py (py)"]
    class resma4_5_py mod;
    resma4_3_py["resma4.3.py (py)"]
    class resma4_3_py mod;
    resma4_10_py["resma4.10.py (py)"]
    class resma4_10_py mod;
    resma4_13_py["resma4.13.py (py)"]
    class resma4_13_py mod;
    resma4_8_py["resma4.8.py (py)"]
    class resma4_8_py mod;
    resma4_6_py["resma4.6.py (py)"]
    class resma4_6_py mod;
    main5_py["main5.py (py)"]
    class main5_py mod;
    resma4_9_py["resma4.9.py (py)"]
    class resma4_9_py mod;
    main_py["main.py (py)"]
    class main_py mod;
    main2_py["main2.py (py)"]
    class main2_py mod;
    resma4_2_py["resma4.2.py (py)"]
    class resma4_2_py mod;
    main3_py["main3.py (py)"]
    class main3_py mod;
    main4_py_py["main4.py.py (py)"]
    class main4_py_py mod;
    sovereignty_monitor_py["sovereignty_monitor.py (py)"]
    class sovereignty_monitor_py mod;
    subgraph community_1 ["resma2"]
    resma2_main_experiments_py["main_experiments.py (py)"]
    class resma2_main_experiments_py mod;
    resma4_7_py["resma4.7.py (py)"]
    class resma4_7_py mod;
    main4_1_py["main4.1.py (py)"]
    class main4_1_py mod;
    monitor_extremo_py["monitor_extremo.py (py)"]
    class monitor_extremo_py mod;
    quick_monitor_py["quick_monitor.py (py)"]
    class quick_monitor_py mod;
    resma2_resma_app_mnist_py["resma_app_mnist.py (py)"]
    class resma2_resma_app_mnist_py mod;
    end
    subgraph community_0 ["root"]
    test_ultra_simple_py["test_ultra_simple.py (py)"]
    class test_ultra_simple_py mod;
    resma2_resma_observer_py["resma_observer.py (py)"]
    class resma2_resma_observer_py mod;
    resma2_main_experiment_py["main_experiment.py (py)"]
    class resma2_main_experiment_py mod;
    train_profile_py["train_profile.py (py)"]
    class train_profile_py mod;
    garnier_nn_py["garnier_nn.py (py)"]
    class garnier_nn_py mod;
    resma2_resma_noise_phase_test_py["resma_noise_phase_test.py (py)"]
    class resma2_resma_noise_phase_test_py mod;
    resma2_resma_vision_py["resma_vision.py (py)"]
    class resma2_resma_vision_py mod;
    visualize_resma_py["visualize_resma.py (py)"]
    class visualize_resma_py mod;
    resma2_resma_combat_test_py["resma_combat_test.py (py)"]
    class resma2_resma_combat_test_py mod;
    train_mini_resma_py["train_mini_resma.py (py)"]
    class train_mini_resma_py mod;
    resma2_resma_core_py["resma_core.py (py)"]
    class resma2_resma_core_py mod;
    resma2_monitor_py["monitor.py (py)"]
    class resma2_monitor_py mod;
    demo_mini_resma_py["demo_mini_resma.py (py)"]
    class demo_mini_resma_py mod;
    resma2_resma_vision_trained_py["resma_vision_trained.py (py)"]
    class resma2_resma_vision_trained_py mod;
    resma2_resma_train_py["resma_train.py (py)"]
    class resma2_resma_train_py mod;
    resma2_resma_breakpoint_py["resma_breakpoint.py (py)"]
    class resma2_resma_breakpoint_py mod;
    resma2_resma_overload_py["resma_overload.py (py)"]
    class resma2_resma_overload_py mod;
    difract_py["difract.py (py)"]
    class difract_py mod;
    test_simple_py["test_simple.py (py)"]
    class test_simple_py mod;
    app_py["app.py (py)"]
    class app_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    end
    resma2_main_experiment_py -- resolved_imports --> resma2_resma_core_py
    resma2_main_experiment_py -- resolved_imports --> resma2_resma_observer_py
    resma2_main_experiments_py -- resolved_imports --> resma2_resma_core_py
    resma2_main_experiments_py -- resolved_imports --> resma2_resma_observer_py
    resma2_main_experiments_py -- resolved_imports --> resma2_monitor_py
    resma2_resma_app_mnist_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_app_mnist_py -- resolved_imports --> resma2_resma_observer_py
    resma2_resma_breakpoint_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_combat_test_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_noise_phase_test_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_observer_py -- resolved_imports --> resma2_monitor_py
    resma2_resma_overload_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_train_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_vision_py -- resolved_imports --> resma2_resma_core_py
    resma2_resma_vision_trained_py -- resolved_imports --> resma2_resma_core_py
    test_ultra_simple_py -- resolved_imports --> garnier_nn_py
    train_mini_resma_py -- resolved_imports --> garnier_nn_py
    train_profile_py -- resolved_imports --> garnier_nn_py
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_torch["torch"]
    class ext_torch ext;
    demo_mini_resma_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    demo_mini_resma_py -.->|imports| ext_torch_nn
    ext_numpy["numpy"]
    class ext_numpy ext;
    demo_mini_resma_py -.->|imports| ext_numpy
    ext_networkx["networkx"]
    class ext_networkx ext;
    demo_mini_resma_py -.->|imports| ext_networkx
    ext_typing["typing"]
    class ext_typing ext;
    demo_mini_resma_py -.->|imports| ext_typing
    ext_logging["logging"]
    class ext_logging ext;
    demo_mini_resma_py -.->|imports| ext_logging
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    difract_py -.->|imports| ext_matplotlib_pyplot
    difract_py -.->|imports| ext_numpy
    garnier_nn_py -.->|imports| ext_torch
    garnier_nn_py -.->|imports| ext_torch_nn
    garnier_nn_py -.->|imports| ext_numpy
    garnier_nn_py -.->|imports| ext_networkx
    garnier_nn_py -.->|imports| ext_typing
    garnier_nn_py -.->|imports| ext_logging
    ext_time["time"]
    class ext_time ext;
    garnier_nn_py -.->|imports| ext_time
    main_py -.->|imports| ext_numpy
    ext_scipy_linalg["scipy.linalg"]
    class ext_scipy_linalg ext;
    main_py -.->|imports| ext_scipy_linalg
    main_py -.->|imports| ext_networkx
    ext_scipy_integrate["scipy.integrate"]
    class ext_scipy_integrate ext;
    main_py -.->|imports| ext_scipy_integrate
    main_py -.->|imports| ext_typing
    main_py -.->|imports| ext_logging
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    main_py -.->|imports| ext_dataclasses
    ext_pint["pint"]
    class ext_pint ext;
    main_py -.->|imports| ext_pint
    main_py -.->|imports| ext_numpy
    ext_scipy_sparse_linalg["scipy.sparse.linalg"]
    class ext_scipy_sparse_linalg ext;
    main_py -.->|imports| ext_scipy_sparse_linalg
    ext_psutil["psutil"]
    class ext_psutil ext;
    main_py -.->|imports| ext_psutil
    main_py -.->|imports| ext_scipy_sparse_linalg
    ext_scipy_sparse["scipy.sparse"]
    class ext_scipy_sparse ext;
    main_py -.->|imports| ext_scipy_sparse
    ext_scipy_interpolate["scipy.interpolate"]
    class ext_scipy_interpolate ext;
    main_py -.->|imports| ext_scipy_interpolate
    ext_ripser["ripser"]
    class ext_ripser ext;
    main_py -.->|imports| ext_ripser
    main2_py -.->|imports| ext_numpy
    main2_py -.->|imports| ext_scipy_linalg
    main2_py -.->|imports| ext_networkx
    main2_py -.->|imports| ext_scipy_integrate
    main2_py -.->|imports| ext_typing
    main2_py -.->|imports| ext_logging
    main2_py -.->|imports| ext_dataclasses
    main2_py -.->|imports| ext_pint
    main2_py -.->|imports| ext_numpy
    main2_py -.->|imports| ext_scipy_sparse_linalg
    main2_py -.->|imports| ext_psutil
    main2_py -.->|imports| ext_scipy_sparse_linalg
    main2_py -.->|imports| ext_scipy_sparse
    main2_py -.->|imports| ext_scipy_interpolate
    main2_py -.->|imports| ext_ripser
    main3_py -.->|imports| ext_numpy
    main3_py -.->|imports| ext_scipy_linalg
    main3_py -.->|imports| ext_networkx
    main3_py -.->|imports| ext_scipy_integrate
    main3_py -.->|imports| ext_scipy_sparse_linalg
    main3_py -.->|imports| ext_typing
    main3_py -.->|imports| ext_dataclasses
    main3_py -.->|imports| ext_logging
    main3_py -.->|imports| ext_pint
    main3_py -.->|imports| ext_psutil
    main3_py -.->|imports| ext_ripser
    main3_py -.->|imports| ext_scipy_sparse
    main4_1_py -.->|imports| ext_numpy
    main4_1_py -.->|imports| ext_scipy_linalg
    main4_1_py -.->|imports| ext_networkx
    main4_1_py -.->|imports| ext_scipy_integrate
    main4_1_py -.->|imports| ext_scipy_sparse_linalg
    main4_1_py -.->|imports| ext_typing
    main4_1_py -.->|imports| ext_dataclasses
    main4_1_py -.->|imports| ext_logging
    ext_warnings["warnings"]
    class ext_warnings ext;
    main4_1_py -.->|imports| ext_warnings
    main4_1_py -.->|imports| ext_ripser
    main4_py_py -.->|imports| ext_numpy
    main4_py_py -.->|imports| ext_scipy_linalg
    main4_py_py -.->|imports| ext_networkx
    main4_py_py -.->|imports| ext_scipy_integrate
    main4_py_py -.->|imports| ext_scipy_sparse_linalg
    main4_py_py -.->|imports| ext_typing
    main4_py_py -.->|imports| ext_dataclasses
    main4_py_py -.->|imports| ext_logging
    main4_py_py -.->|imports| ext_pint
    main4_py_py -.->|imports| ext_psutil
    main4_py_py -.->|imports| ext_ripser
    main4_py_py -.->|imports| ext_scipy_sparse
    main5_py -.->|imports| ext_numpy
    main5_py -.->|imports| ext_scipy_linalg
    main5_py -.->|imports| ext_networkx
    main5_py -.->|imports| ext_scipy_integrate
    main5_py -.->|imports| ext_typing
    main5_py -.->|imports| ext_logging
    main5_py -.->|imports| ext_dataclasses
    main5_py -.->|imports| ext_pint
    main5_py -.->|imports| ext_numpy
    main5_py -.->|imports| ext_scipy_sparse_linalg
    main5_py -.->|imports| ext_psutil
    main5_py -.->|imports| ext_warnings
    main5_py -.->|imports| ext_scipy_interpolate
    main5_py -.->|imports| ext_scipy_sparse
    main5_py -.->|imports| ext_ripser
    main5_py -.->|imports| ext_ripser
    monitor_extremo_py -.->|imports| ext_torch
    monitor_extremo_py -.->|imports| ext_torch_nn
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    monitor_extremo_py -.->|imports| ext_torch_optim
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    monitor_extremo_py -.->|imports| ext_torch_nn_functional
    monitor_extremo_py -.->|imports| ext_numpy
    monitor_extremo_py -.->|imports| ext_matplotlib_pyplot
    monitor_extremo_py -.->|imports| ext_typing
    monitor_extremo_py -.->|imports| ext_warnings
    monitor_extremo_py -.->|imports| ext_time
    ext_traceback["traceback"]
    class ext_traceback ext;
    monitor_extremo_py -.->|imports| ext_traceback
    quick_monitor_py -.->|imports| ext_torch
    quick_monitor_py -.->|imports| ext_torch_nn
    quick_monitor_py -.->|imports| ext_torch_optim
    quick_monitor_py -.->|imports| ext_torch_nn_functional
    quick_monitor_py -.->|imports| ext_numpy
    quick_monitor_py -.->|imports| ext_matplotlib_pyplot
    quick_monitor_py -.->|imports| ext_typing
    quick_monitor_py -.->|imports| ext_warnings
    quick_monitor_py -.->|imports| ext_time
    quick_monitor_py -.->|imports| ext_traceback
    resma2_main_experiment_py -.->|imports| ext_torch
    resma2_main_experiment_py -.->|imports| ext_torch_nn
    resma2_main_experiment_py -.->|imports| ext_numpy
    ext_random["random"]
    class ext_random ext;
    resma2_main_experiment_py -.->|imports| ext_random
    ext_resma_core["resma_core"]
    class ext_resma_core ext;
    resma2_main_experiment_py -.->|imports| ext_resma_core
    ext_resma_observer["resma_observer"]
    class ext_resma_observer ext;
    resma2_main_experiment_py -.->|imports| ext_resma_observer
    resma2_main_experiments_py -.->|imports| ext_torch
    ext_torchvision["torchvision"]
    class ext_torchvision ext;
    resma2_main_experiments_py -.->|imports| ext_torchvision
    ext_torch_utils_data["torch.utils.data"]
    class ext_torch_utils_data ext;
    resma2_main_experiments_py -.->|imports| ext_torch_utils_data
    resma2_main_experiments_py -.->|imports| ext_torch_optim
    resma2_main_experiments_py -.->|imports| ext_torch_nn
    resma2_main_experiments_py -.->|imports| ext_resma_core
    resma2_main_experiments_py -.->|imports| ext_resma_observer
    ext_monitor["monitor"]
    class ext_monitor ext;
    resma2_main_experiments_py -.->|imports| ext_monitor
    resma2_monitor_py -.->|imports| ext_torch
    resma2_monitor_py -.->|imports| ext_numpy
    resma2_monitor_py -.->|imports| ext_typing
    resma2_monitor_py -.->|imports| ext_warnings
    resma2_monitor_py -.->|imports| ext_dataclasses
    ext_enum["enum"]
    class ext_enum ext;
    resma2_monitor_py -.->|imports| ext_enum
    resma2_resma_app_mnist_py -.->|imports| ext_torch
    resma2_resma_app_mnist_py -.->|imports| ext_torch_nn
    resma2_resma_app_mnist_py -.->|imports| ext_torch_optim
    resma2_resma_app_mnist_py -.->|imports| ext_torchvision
    resma2_resma_app_mnist_py -.->|imports| ext_torch_utils_data
    resma2_resma_app_mnist_py -.->|imports| ext_numpy
    resma2_resma_app_mnist_py -.->|imports| ext_resma_core
    resma2_resma_app_mnist_py -.->|imports| ext_resma_observer
    resma2_resma_breakpoint_py -.->|imports| ext_torch
    resma2_resma_breakpoint_py -.->|imports| ext_numpy
    resma2_resma_breakpoint_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_breakpoint_py -.->|imports| ext_resma_core
    resma2_resma_combat_test_py -.->|imports| ext_torch
    resma2_resma_combat_test_py -.->|imports| ext_numpy
    resma2_resma_combat_test_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_combat_test_py -.->|imports| ext_torchvision
    resma2_resma_combat_test_py -.->|imports| ext_torch_utils_data
    resma2_resma_combat_test_py -.->|imports| ext_resma_core
    resma2_resma_core_py -.->|imports| ext_torch
    resma2_resma_core_py -.->|imports| ext_torch_nn
    resma2_resma_core_py -.->|imports| ext_torch_nn_functional
    resma2_resma_core_py -.->|imports| ext_networkx
    resma2_resma_core_py -.->|imports| ext_numpy
    resma2_resma_core_py -.->|imports| ext_typing
    resma2_resma_noise_phase_test_py -.->|imports| ext_torch
    resma2_resma_noise_phase_test_py -.->|imports| ext_torch_nn_functional
    resma2_resma_noise_phase_test_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_noise_phase_test_py -.->|imports| ext_numpy
    resma2_resma_noise_phase_test_py -.->|imports| ext_torchvision
    resma2_resma_noise_phase_test_py -.->|imports| ext_resma_core
    resma2_resma_observer_py -.->|imports| ext_torch
    resma2_resma_observer_py -.->|imports| ext_numpy
    resma2_resma_observer_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_observer_py -.->|imports| ext_dataclasses
    resma2_resma_observer_py -.->|imports| ext_typing
    ext_json["json"]
    class ext_json ext;
    resma2_resma_observer_py -.->|imports| ext_json
    resma2_resma_observer_py -.->|imports| ext_monitor
    resma2_resma_overload_py -.->|imports| ext_torch
    resma2_resma_overload_py -.->|imports| ext_numpy
    resma2_resma_overload_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_overload_py -.->|imports| ext_resma_core
    resma2_resma_train_py -.->|imports| ext_torch
    resma2_resma_train_py -.->|imports| ext_torch_nn
    resma2_resma_train_py -.->|imports| ext_torchvision
    resma2_resma_train_py -.->|imports| ext_torch_utils_data
    resma2_resma_train_py -.->|imports| ext_resma_core
    resma2_resma_vision_py -.->|imports| ext_torch
    resma2_resma_vision_py -.->|imports| ext_torch_nn
    resma2_resma_vision_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_vision_py -.->|imports| ext_torchvision
    resma2_resma_vision_py -.->|imports| ext_numpy
    resma2_resma_vision_py -.->|imports| ext_resma_core
    resma2_resma_vision_trained_py -.->|imports| ext_torch
    resma2_resma_vision_trained_py -.->|imports| ext_matplotlib_pyplot
    resma2_resma_vision_trained_py -.->|imports| ext_torchvision
    resma2_resma_vision_trained_py -.->|imports| ext_numpy
    resma2_resma_vision_trained_py -.->|imports| ext_resma_core
    resma4_10_py -.->|imports| ext_numpy
    resma4_10_py -.->|imports| ext_scipy_linalg
    resma4_10_py -.->|imports| ext_networkx
    resma4_10_py -.->|imports| ext_scipy_integrate
    resma4_10_py -.->|imports| ext_scipy_sparse_linalg
    resma4_10_py -.->|imports| ext_typing
    resma4_10_py -.->|imports| ext_dataclasses
    resma4_10_py -.->|imports| ext_logging
    resma4_10_py -.->|imports| ext_warnings
    ext_pickle["pickle"]
    class ext_pickle ext;
    resma4_10_py -.->|imports| ext_pickle
    ext_gc["gc"]
    class ext_gc ext;
    resma4_10_py -.->|imports| ext_gc
    resma4_10_py -.->|imports| ext_os
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    resma4_10_py -.->|imports| ext_pathlib
    resma4_10_py -.->|imports| ext_psutil
    ext_datetime["datetime"]
    class ext_datetime ext;
    resma4_10_py -.->|imports| ext_datetime
    ext_weakref["weakref"]
    class ext_weakref ext;
    resma4_10_py -.->|imports| ext_weakref
    resma4_10_py -.->|imports| ext_time
    ext_itertools["itertools"]
    class ext_itertools ext;
    resma4_10_py -.->|imports| ext_itertools
    resma4_13_py -.->|imports| ext_numpy
    resma4_13_py -.->|imports| ext_scipy_linalg
    resma4_13_py -.->|imports| ext_networkx
    resma4_13_py -.->|imports| ext_scipy_integrate
    resma4_13_py -.->|imports| ext_scipy_sparse_linalg
    resma4_13_py -.->|imports| ext_typing
    resma4_13_py -.->|imports| ext_dataclasses
    resma4_13_py -.->|imports| ext_logging
    resma4_13_py -.->|imports| ext_warnings
    resma4_13_py -.->|imports| ext_pickle
    resma4_13_py -.->|imports| ext_gc
    resma4_13_py -.->|imports| ext_os
    resma4_13_py -.->|imports| ext_pathlib
    resma4_13_py -.->|imports| ext_psutil
    resma4_13_py -.->|imports| ext_datetime
    resma4_13_py -.->|imports| ext_weakref
    resma4_13_py -.->|imports| ext_time
    resma4_13_py -.->|imports| ext_itertools
    resma4_2_py -.->|imports| ext_numpy
    resma4_2_py -.->|imports| ext_scipy_linalg
    resma4_2_py -.->|imports| ext_networkx
    resma4_2_py -.->|imports| ext_scipy_integrate
    resma4_2_py -.->|imports| ext_scipy_sparse_linalg
    resma4_2_py -.->|imports| ext_scipy_interpolate
    resma4_2_py -.->|imports| ext_scipy_sparse
    resma4_2_py -.->|imports| ext_typing
    resma4_2_py -.->|imports| ext_dataclasses
    resma4_2_py -.->|imports| ext_logging
    resma4_2_py -.->|imports| ext_pint
    resma4_2_py -.->|imports| ext_psutil
    resma4_2_py -.->|imports| ext_warnings
    resma4_2_py -.->|imports| ext_ripser
    resma4_2_py -.->|imports| ext_ripser
    resma4_3_py -.->|imports| ext_numpy
    resma4_3_py -.->|imports| ext_scipy_linalg
    resma4_3_py -.->|imports| ext_networkx
    resma4_3_py -.->|imports| ext_scipy_integrate
    resma4_3_py -.->|imports| ext_scipy_sparse
    resma4_3_py -.->|imports| ext_typing
    resma4_3_py -.->|imports| ext_dataclasses
    resma4_3_py -.->|imports| ext_logging
    resma4_3_py -.->|imports| ext_pint
    resma4_3_py -.->|imports| ext_psutil
    resma4_3_py -.->|imports| ext_warnings
    resma4_3_py -.->|imports| ext_pickle
    resma4_3_py -.->|imports| ext_time
    resma4_3_py -.->|imports| ext_os
    resma4_3_py -.->|imports| ext_gc
    resma4_3_py -.->|imports| ext_datetime
    resma4_3_py -.->|imports| ext_weakref
    resma4_3_py -.->|imports| ext_ripser
    resma4_3_py -.->|imports| ext_ripser
    resma4_4_py -.->|imports| ext_numpy
    resma4_4_py -.->|imports| ext_scipy_linalg
    resma4_4_py -.->|imports| ext_networkx
    resma4_4_py -.->|imports| ext_scipy_integrate
    resma4_4_py -.->|imports| ext_scipy_sparse
    resma4_4_py -.->|imports| ext_scipy_sparse_linalg
    resma4_4_py -.->|imports| ext_typing
    resma4_4_py -.->|imports| ext_dataclasses
    resma4_4_py -.->|imports| ext_logging
    resma4_4_py -.->|imports| ext_pint
    resma4_4_py -.->|imports| ext_psutil
    resma4_4_py -.->|imports| ext_warnings
    resma4_4_py -.->|imports| ext_pickle
    resma4_4_py -.->|imports| ext_time
    resma4_4_py -.->|imports| ext_os
    resma4_4_py -.->|imports| ext_gc
    resma4_4_py -.->|imports| ext_datetime
    resma4_4_py -.->|imports| ext_weakref
    resma4_4_py -.->|imports| ext_ripser
    resma4_4_py -.->|imports| ext_ripser
    resma4_5_py -.->|imports| ext_numpy
    resma4_5_py -.->|imports| ext_scipy_linalg
    resma4_5_py -.->|imports| ext_networkx
    resma4_5_py -.->|imports| ext_scipy_integrate
    resma4_5_py -.->|imports| ext_scipy_sparse
    resma4_5_py -.->|imports| ext_scipy_sparse_linalg
    resma4_5_py -.->|imports| ext_typing
    resma4_5_py -.->|imports| ext_dataclasses
    resma4_5_py -.->|imports| ext_logging
    resma4_5_py -.->|imports| ext_psutil
    resma4_5_py -.->|imports| ext_warnings
    resma4_5_py -.->|imports| ext_pickle
    resma4_5_py -.->|imports| ext_time
    resma4_5_py -.->|imports| ext_os
    resma4_5_py -.->|imports| ext_gc
    resma4_5_py -.->|imports| ext_datetime
    resma4_5_py -.->|imports| ext_weakref
    resma4_5_py -.->|imports| ext_pathlib
    resma4_5_py -.->|imports| ext_pint
    resma4_6_py -.->|imports| ext_numpy
    resma4_6_py -.->|imports| ext_scipy_linalg
    resma4_6_py -.->|imports| ext_networkx
    resma4_6_py -.->|imports| ext_scipy_integrate
    resma4_6_py -.->|imports| ext_typing
    resma4_6_py -.->|imports| ext_dataclasses
    resma4_6_py -.->|imports| ext_logging
    resma4_6_py -.->|imports| ext_psutil
    resma4_6_py -.->|imports| ext_warnings
    resma4_6_py -.->|imports| ext_pickle
    resma4_6_py -.->|imports| ext_time
    resma4_6_py -.->|imports| ext_os
    resma4_6_py -.->|imports| ext_gc
    resma4_6_py -.->|imports| ext_datetime
    resma4_6_py -.->|imports| ext_weakref
    resma4_6_py -.->|imports| ext_pathlib
    resma4_6_py -.->|imports| ext_pint
    resma4_7_py -.->|imports| ext_numpy
    resma4_7_py -.->|imports| ext_scipy_linalg
    resma4_7_py -.->|imports| ext_networkx
    resma4_7_py -.->|imports| ext_scipy_integrate
    resma4_7_py -.->|imports| ext_scipy_sparse_linalg
    resma4_7_py -.->|imports| ext_typing
    resma4_7_py -.->|imports| ext_dataclasses
    resma4_7_py -.->|imports| ext_logging
    resma4_7_py -.->|imports| ext_warnings
    resma4_7_py -.->|imports| ext_gc
    resma4_8_py -.->|imports| ext_numpy
    resma4_8_py -.->|imports| ext_scipy_linalg
    resma4_8_py -.->|imports| ext_networkx
    resma4_8_py -.->|imports| ext_scipy_integrate
    resma4_8_py -.->|imports| ext_scipy_sparse
    resma4_8_py -.->|imports| ext_scipy_sparse_linalg
    resma4_8_py -.->|imports| ext_typing
    resma4_8_py -.->|imports| ext_dataclasses
    resma4_8_py -.->|imports| ext_logging
    resma4_8_py -.->|imports| ext_psutil
    resma4_8_py -.->|imports| ext_warnings
    resma4_8_py -.->|imports| ext_pickle
    resma4_8_py -.->|imports| ext_time
    resma4_8_py -.->|imports| ext_os
    resma4_8_py -.->|imports| ext_gc
    resma4_8_py -.->|imports| ext_datetime
    resma4_8_py -.->|imports| ext_weakref
    resma4_8_py -.->|imports| ext_pathlib
    resma4_9_py -.->|imports| ext_numpy
    resma4_9_py -.->|imports| ext_scipy_linalg
    resma4_9_py -.->|imports| ext_networkx
    resma4_9_py -.->|imports| ext_scipy_integrate
    resma4_9_py -.->|imports| ext_scipy_sparse_linalg
    resma4_9_py -.->|imports| ext_typing
    resma4_9_py -.->|imports| ext_dataclasses
    resma4_9_py -.->|imports| ext_logging
    resma4_9_py -.->|imports| ext_warnings
    resma4_9_py -.->|imports| ext_pickle
    resma4_9_py -.->|imports| ext_gc
    resma4_9_py -.->|imports| ext_os
    resma4_9_py -.->|imports| ext_pathlib
    resma4_9_py -.->|imports| ext_psutil
    resma4_9_py -.->|imports| ext_datetime
    resma4_9_py -.->|imports| ext_weakref
    sovereignty_monitor_py -.->|imports| ext_torch
    sovereignty_monitor_py -.->|imports| ext_torch_nn
    sovereignty_monitor_py -.->|imports| ext_torch_optim
    sovereignty_monitor_py -.->|imports| ext_torch_nn_functional
    sovereignty_monitor_py -.->|imports| ext_torchvision
    sovereignty_monitor_py -.->|imports| ext_numpy
    sovereignty_monitor_py -.->|imports| ext_matplotlib_pyplot
    sovereignty_monitor_py -.->|imports| ext_typing
    sovereignty_monitor_py -.->|imports| ext_warnings
    sovereignty_monitor_py -.->|imports| ext_os
    sovereignty_monitor_py -.->|imports| ext_traceback
    test_simple_py -.->|imports| ext_torch
    test_simple_py -.->|imports| ext_numpy
    test_ultra_simple_py -.->|imports| ext_torch
    test_ultra_simple_py -.->|imports| ext_torch_nn
    test_ultra_simple_py -.->|imports| ext_numpy
    test_ultra_simple_py -.->|imports| ext_networkx
    ext_garnier_nn["garnier_nn"]
    class ext_garnier_nn ext;
    test_ultra_simple_py -.->|imports| ext_garnier_nn
    test_ultra_simple_py -.->|imports| ext_typing
    test_ultra_simple_py -.->|imports| ext_logging
    test_ultra_simple_py -.->|imports| ext_time
    train_mini_resma_py -.->|imports| ext_torch
    train_mini_resma_py -.->|imports| ext_torch_utils_data
    train_mini_resma_py -.->|imports| ext_torchvision
    train_mini_resma_py -.->|imports| ext_garnier_nn
    train_mini_resma_py -.->|imports| ext_logging
    train_mini_resma_py -.->|imports| ext_os
    train_profile_py -.->|imports| ext_torch
    train_profile_py -.->|imports| ext_torch_utils_data
    train_profile_py -.->|imports| ext_torchvision
    train_profile_py -.->|imports| ext_garnier_nn
    train_profile_py -.->|imports| ext_logging
    train_profile_py -.->|imports| ext_time
    train_profile_py -.->|imports| ext_os
    visualize_resma_py -.->|imports| ext_matplotlib_pyplot
    visualize_resma_py -.->|imports| ext_torch
    visualize_resma_py -.->|imports| ext_networkx
    visualize_resma_py -.->|imports| ext_numpy
    visualize_resma_py -.->|imports| ext_warnings
    visualize_resma_py -.->|imports| ext_matplotlib_pyplot
    ext_seaborn["seaborn"]
    class ext_seaborn ext;
    visualize_resma_py -.->|imports| ext_seaborn
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://readmenator.dev/cpg/v1", "analysis": {"communities": [{"cohesion": 1.0, "id": 0, "label": "root", "size": 4}, {"cohesion": 1.0, "id": 1, "label": "resma2", "size": 13}], "god_nodes": [{"node_id": "resma2/resma_core.py", "score": 21.0}, {"node_id": "resma2/resma_observer.py", "score": 8.9}, {"node_id": "main5.py", "score": 8.1}, {"node_id": "garnier_nn.py", "score": 7.1}, {"node_id": "main.py", "score": 6.3}, {"node_id": "main2.py", "score": 6.3}, {"node_id": "resma2/main_experiments.py", "score": 6.3}, {"node_id": "resma4.6.py", "score": 5.9}, {"node_id": "resma4.10.py", "score": 5.7}, {"node_id": "resma4.13.py", "score": 5.7}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "demo_mini_resma.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "difract.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "difract.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "garnier_nn.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scipy.interpolate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "scipy.interpolate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main2.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main3.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.1.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main4.py.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "scipy.interpolate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main5.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "monitor_extremo.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "quick_monitor.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiment.py", "target": "resma_observer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "resma_observer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/main_experiments.py", "target": "monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/monitor.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_app_mnist.py", "target": "resma_observer"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_breakpoint.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_breakpoint.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_breakpoint.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_breakpoint.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_combat_test.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_core.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_noise_phase_test.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_observer.py", "target": "monitor"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_overload.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_overload.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_overload.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_overload.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_train.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_train.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_train.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_train.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_train.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision_trained.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision_trained.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision_trained.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision_trained.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma2/resma_vision_trained.py", "target": "resma_core"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.10.py", "target": "itertools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.13.py", "target": "itertools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "scipy.interpolate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.2.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.3.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.4.py", "target": "ripser"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.5.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.6.py", "target": "pint"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.7.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.8.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "scipy.integrate"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "scipy.sparse.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "psutil"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resma4.9.py", "target": "weakref"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "sovereignty_monitor.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_simple.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_simple.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "garnier_nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "test_ultra_simple.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "garnier_nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_mini_resma.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "garnier_nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_profile.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "networkx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "visualize_resma.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/main_experiment.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/main_experiment.py", "target": "resma2/resma_observer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/main_experiments.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/main_experiments.py", "target": "resma2/resma_observer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/main_experiments.py", "target": "resma2/monitor.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_app_mnist.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_app_mnist.py", "target": "resma2/resma_observer.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_breakpoint.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_combat_test.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_noise_phase_test.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_observer.py", "target": "resma2/monitor.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_overload.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_train.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_vision.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "resma2/resma_vision_trained.py", "target": "resma2/resma_core.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "test_ultra_simple.py", "target": "garnier_nn.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "train_mini_resma.py", "target": "garnier_nn.py"}, {"confidence": "EXTRACTED", "relation": "resolved_imports", "source": "train_profile.py", "target": "garnier_nn.py"}], "generator": "readmenator", "metadata": {"edge_count": 5972, "file_count": 42, "language_count": 2, "symbol_count": 905}, "nodes": [{"doc": "_*_ coding: utf8 _*_", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "57b21bdb023585b8", "symbol_count": 0, "symbols": []}, {"id": "demo_mini_resma.py", "kind": "module", "label": "demo_mini_resma.py", "language": "py", "sha256": "b7340a1dfbdd0519", "symbol_count": 4, "symbols": [{"doc": "Capa neuronal con temporalidad Garnier T³ (simplificada para demo)", "kind": "class", "line": 8, "name": "GarnierLayer", "signature": "class GarnierLayer(Module)"}, {"doc": "Demostración rápida de la arquitectura RESMA-Garnier", "kind": "method", "line": 49, "name": "demo_resma", "signature": "def demo_resma()"}, {"kind": "method", "line": 10, "name": "__init__", "signature": "def __init__(self, in_features, out_features, device)"}, {"doc": "Forward simplificado para demostración", "kind": "method", "line": 27, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "difract.py", "kind": "module", "label": "difract.py", "language": "py", "sha256": "fddf3a03019f769f", "symbol_count": 1, "symbols": [{"kind": "function", "line": 4, "name": "visualize_uased_geometry", "signature": "def visualize_uased_geometry()"}]}, {"id": "garnier_nn.py", "kind": "module", "label": "garnier_nn.py", "language": "py", "sha256": "8315793d81cad9a6", "symbol_count": 11, "symbols": [{"doc": "Capa neuronal con temporalidad Garnier T³", "kind": "class", "line": 9, "name": "GarnierLayer", "signature": "class GarnierLayer(Module)"}, {"doc": "Red neuronal completa con arquitectura RESMA-Garnier", "kind": "class", "line": 67, "name": "SilencioActivoNetwork", "signature": "class SilencioActivoNetwork(Module)"}, {"kind": "method", "line": 11, "name": "__init__", "signature": "def __init__(self, in_features, out_features, device)"}, {"doc": "Forward con no-linealidad Garnier\nReturns: (output, delta_s_loop)", "kind": "method", "line": 33, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 69, "name": "__init__", "signature": "def __init__(self, layer_sizes, scale, device)"}, {"doc": "Construcción BA+WS modular miniaturizada", "kind": "method", "line": 111, "name": "_build_garnier_topology", "signature": "def _build_garnier_topology(self)"}, {"doc": "Forward completo con tracking de métricas de consciencia\nReturns: (logits, metrics)", "kind": "method", "line": 130, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Activar perfilado de tiempo en toda la red", "kind": "method", "line": 168, "name": "activar_perfilado", "signature": "def activar_perfilado(self)"}, {"doc": "Mostrar estadísticas de perfilado", "kind": "method", "line": 183, "name": "mostrar_estadisticas_perfilado", "signature": "def mostrar_estadisticas_perfilado(self)"}, {"doc": "Entrenamiento con perfilado detallado", "kind": "method", "line": 201, "name": "entrenar_con_perfilado", "signature": "def entrenar_con_perfilado(self, train_loader, epochs, lr)"}, {"doc": "Entrenamiento incorporado con regularización Garnier", "kind": "method", "line": 240, "name": "entrenar", "signature": "def entrenar(self, train_loader, epochs, lr)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"doc": "============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================", "id": "main.py", "kind": "module", "label": "main.py", "language": "py", "sha256": "530d8a7301dc1c27", "symbol_count": 63, "symbols": [{"doc": "Constantes físicas y parámetros de la teoría RESMA", "kind": "class", "line": 32, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Validación de rangos físicos para todas las constantes", "kind": "class", "line": 58, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"doc": "Hoja L_i de la Resma como estado KMS mean-field.\nNo almacena matrices densas (Pilar 4).", "kind": "class", "line": 90, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso como foliación medible sin matrices densas.\nMemoria: O(N_leaves) en lugar de O(N_leaves × dim²)", "kind": "class", "line": 144, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Operador Ĥ que abre la Resma cuando β_i es no trivial.\nImplementación local sin matrices globales (Pilar 4).", "kind": "class", "line": 220, "name": "BranchingOperator", "signature": "class BranchingOperator"}, {"doc": "Operador P̂_E: proyección teleológica no lineal.\nImplementación con muestreo Monte Carlo (Pilar 4).", "kind": "class", "line": 270, "name": "EmunaOperator", "signature": "class EmunaOperator"}, {"doc": "SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)\nIntegración por Euler-Maruyama (Pilar 4: estabilidad numérica).", "kind": "class", "line": 350, "name": "LindbladFractalDynamics", "signature": "class LindbladFractalDynamics"}, {"doc": "Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.\nImplementación 1D mean-field para Colab (Pilar 4).", "kind": "class", "line": 428, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Conectoma humano dirigido con homología persistente.\nImplementación sparse para escalado (Pilar 4).", "kind": "class", "line": 481, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.\nCálculo topológico sin densidades matriciales (Pilar 4).", "kind": "class", "line": 582, "name": "FreedomInvariant", "signature": "class FreedomInvariant"}, {"doc": "Modelos nulos para cálculo de Factor de Bayes.\nBasados en teorías establecidas sin postulados RESMA.", "kind": "class", "line": 628, "name": "NullModels", "signature": "class NullModels"}, {"doc": "Predicciones falsables contra modelos nulos teóricos.\nFactor de Bayes calculado con AIC (aproximación).", "kind": "class", "line": 686, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline completo RESMA 3.0 con verificaciones de integridad.\nDiseñado para ejecución en Google Colab (Pilar 4).", "kind": "method", "line": 764, "name": "simulate_resma_multiverse", "signature": "def simulate_resma_multiverse(n_leaves, n_nodes, seed)"}, {"doc": "α ∈ (0,1) por definición de dimensión fractal", "kind": "method", "line": 62, "name": "validate_dimension", "signature": "def validate_dimension(alpha)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1 para PT-simetría", "kind": "method", "line": 68, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"doc": "Límite inferior para conectoma biológico", "kind": "method", "line": 79, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"doc": "Validaciones post-construcción (Pilar 3)", "kind": "method", "line": 100, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Densidad espectral continua ρ(ω) para álgebra tipo III₁.\nEvidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).", "kind": "method", "line": 106, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)", "kind": "method", "line": 114, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 121, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Momentos espectrales Tr(ρ^k) para k=1..n", "kind": "method", "line": 132, "name": "_spectral_moments", "signature": "def _spectral_moments(self, n)"}, {"doc": "Args:\n    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)\n    seed: Reproducibilidad (Pilar 4)", "kind": "method", "line": 150, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"doc": "Genera hojas con gaps espectrales distribuidos", "kind": "method", "line": 168, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))", "kind": "method", "line": 181, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Estado global: mapa de pesos por hoja (no matriz)", "kind": "method", "line": 203, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 226, "name": "__init__", "signature": "def __init__(self, leaf, threshold)"}, {"doc": "Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)", "kind": "method", "line": 231, "name": "_construct_cptp_map", "signature": "def _construct_cptp_map(self)"}, {"doc": "K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.\nAproximación mean-field: operadores de Pauli escalados por gap.", "kind": "method", "line": 240, "name": "_local_jump_operator", "signature": "def _local_jump_operator(self, power)"}, {"doc": "Aplicar canal CPTP a vector de estado local (dim=2)", "kind": "method", "line": 255, "name": "apply_branching", "signature": "def apply_branching(self, state_vector)"}, {"kind": "method", "line": 276, "name": "__init__", "signature": "def __init__(self, universe, n_samples)"}, {"doc": "E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior", "kind": "method", "line": 282, "name": "_construct_hardy_state", "signature": "def _construct_hardy_state(self)"}, {"doc": "Proyector P_E en base de Fourier positiva (dim reducida)", "kind": "method", "line": 286, "name": "_szego_projector", "signature": "def _szego_projector(self)"}, {"doc": "Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))", "kind": "method", "line": 294, "name": "_evaluation_functional", "signature": "def _evaluation_functional(self, state_weights)"}, {"doc": "P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO", "kind": "method", "line": 310, "name": "project", "signature": "def project(self, state_vector)"}, {"kind": "method", "line": 356, "name": "__init__", "signature": "def __init__(self, universe, emuna)"}, {"doc": "H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)", "kind": "method", "line": 362, "name": "_effective_hamiltonian", "signature": "def _effective_hamiltonian(self)"}, {"doc": "L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}", "kind": "method", "line": 372, "name": "_modular_dissipator", "signature": "def _modular_dissipator(self, state)"}, {"doc": "G[ρ, log ρ_∞] = g[ρ, log ρ_∞]", "kind": "method", "line": 382, "name": "_nonlinear_term", "signature": "def _nonlinear_term(self, state)"}, {"doc": "Integración SDE con Euler-Maruyama.\nReturns: trayectoria [n_steps, 2, 2]", "kind": "method", "line": 389, "name": "evolve", "signature": "def evolve(self, rho0, t_span, n_steps)"}, {"kind": "method", "line": 437, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³", "kind": "method", "line": 442, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"doc": "V_loss ∝ (r⊥/a₀)^{2α} con α=0.7", "kind": "method", "line": 448, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1", "kind": "method", "line": 456, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Discordia cuántica aproximada (ejemplo: estado separable → 0)", "kind": "method", "line": 462, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 487, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Grafo dirigido con distribución de grados power-law.\nFuente: Human Connectome Project (Pilar 1).", "kind": "method", "line": 500, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)", "kind": "method", "line": 510, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "R_Q(G) = min{n | β_{n-1}(G) > 0}\nRequiere ripser (instalable en Colab: !pip install ripser)", "kind": "method", "line": 527, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Matriz de distancias shortest-path (sparse CSR)", "kind": "method", "line": 548, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}, {"doc": "t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25\nPilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)", "kind": "method", "line": 562, "name": "critical_percolation_time", "signature": "def critical_percolation_time(self)"}, {"doc": "Verificar coherencia: subgrafo > 70% del total", "kind": "method", "line": 574, "name": "is_coherent_subgraph", "signature": "def is_coherent_subgraph(self, subgraph_nodes)"}, {"kind": "method", "line": 588, "name": "__init__", "signature": "def __init__(self, network, universe)"}, {"doc": "Δ_S* = ε_c en punto excepcional", "kind": "method", "line": 592, "name": "compute_entropy_gap", "signature": "def compute_entropy_gap(self)"}, {"doc": "S_top[G] = χ(G)/|V| (número de Euler normalizado)", "kind": "method", "line": 596, "name": "compute_pontryagin_number", "signature": "def compute_pontryagin_number(self)"}, {"doc": "L[G] = Δ_S* / S_top[G]", "kind": "method", "line": 607, "name": "compute_freedom", "signature": "def compute_freedom(self)"}, {"doc": "|L[G] - 1| < 0.05 en estado crítico", "kind": "method", "line": 618, "name": "is_gauge_invariant", "signature": "def is_gauge_invariant(self)"}, {"doc": "Modelo de Ising cuántico transversal en red fractal.\nPredice t_c sin SYK₈ ni E₈.", "kind": "method", "line": 635, "name": "ising_quantum", "signature": "def ising_quantum(network)"}, {"doc": "SYK₄ estándar (sin R-simetría Spin(7)).\nPredice α sin postulado E₈.", "kind": "method", "line": 654, "name": "syk4", "signature": "def syk4(network)"}, {"doc": "Red aleatoria Erdős-Rényi sin percolación cuántica.", "kind": "method", "line": 670, "name": "random_network", "signature": "def random_network(network)"}, {"kind": "method", "line": 692, "name": "__init__", "signature": "def __init__(self, resma, myelin, network)"}, {"doc": "Predicciones RESMA 3.0", "kind": "method", "line": 699, "name": "predict_all", "signature": "def predict_all(self)"}, {"doc": "q₀ = 2π/L_E8 (sin ajuste)", "kind": "method", "line": 710, "name": "_predict_diffraction_peak", "signature": "def _predict_diffraction_peak(self)"}, {"doc": "BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)\nk = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)", "kind": "method", "line": 715, "name": "compute_bayes_factor", "signature": "def compute_bayes_factor(self)"}]}, {"doc": "============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================", "id": "main2.py", "kind": "module", "label": "main2.py", "language": "py", "sha256": "956cae30b6c6654f", "symbol_count": 63, "symbols": [{"doc": "Constantes físicas y parámetros de la teoría RESMA", "kind": "class", "line": 33, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Validación de rangos físicos para todas las constantes", "kind": "class", "line": 59, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"doc": "Hoja L_i de la Resma como estado KMS mean-field.\nNo almacena matrices densas (Pilar 4).", "kind": "class", "line": 91, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso como foliación medible sin matrices densas.\nMemoria: O(N_leaves) en lugar de O(N_leaves × dim²)", "kind": "class", "line": 144, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Operador Ĥ que abre la Resma cuando β_i es no trivial.\nImplementación local sin matrices globales (Pilar 4).", "kind": "class", "line": 219, "name": "BranchingOperator", "signature": "class BranchingOperator"}, {"doc": "Operador P̂_E: proyección teleológica no lineal.\nImplementación con muestreo Monte Carlo (Pilar 4).", "kind": "class", "line": 269, "name": "EmunaOperator", "signature": "class EmunaOperator"}, {"doc": "SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)\nIntegración por Euler-Maruyama (Pilar 4: estabilidad numérica).", "kind": "class", "line": 349, "name": "LindbladFractalDynamics", "signature": "class LindbladFractalDynamics"}, {"doc": "Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.\nImplementación 1D mean-field para Colab (Pilar 4).", "kind": "class", "line": 427, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Conectoma humano dirigido con homología persistente.\nImplementación sparse para escalado (Pilar 4).", "kind": "class", "line": 480, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.\nCálculo topológico sin densidades matriciales (Pilar 4).", "kind": "class", "line": 581, "name": "FreedomInvariant", "signature": "class FreedomInvariant"}, {"doc": "Modelos nulos para cálculo de Factor de Bayes.\nBasados en teorías establecidas sin postulados RESMA.", "kind": "class", "line": 627, "name": "NullModels", "signature": "class NullModels"}, {"doc": "Predicciones falsables contra modelos nulos teóricos.\nFactor de Bayes calculado con AIC (aproximación).", "kind": "class", "line": 685, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline completo RESMA 3.0 con verificaciones de integridad.\nDiseñado para ejecución en Google Colab (Pilar 4).", "kind": "method", "line": 763, "name": "simulate_resma_multiverse", "signature": "def simulate_resma_multiverse(n_leaves, n_nodes, seed)"}, {"doc": "α ∈ (0,1) por definición de dimensión fractal", "kind": "method", "line": 63, "name": "validate_dimension", "signature": "def validate_dimension(alpha)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1 para PT-simetría", "kind": "method", "line": 69, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"doc": "Límite inferior para conectoma biológico", "kind": "method", "line": 80, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"doc": "Validaciones post-construcción (Pilar 3)", "kind": "method", "line": 101, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Densidad espectral continua ρ(ω) para álgebra tipo III₁.\nEvidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).", "kind": "method", "line": 106, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)", "kind": "method", "line": 114, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 121, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Momentos espectrales Tr(ρ^k) para k=1..n", "kind": "method", "line": 132, "name": "_spectral_moments", "signature": "def _spectral_moments(self, n)"}, {"doc": "Args:\n    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)\n    seed: Reproducibilidad (Pilar 4)", "kind": "method", "line": 150, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"doc": "Genera hojas con gaps espectrales distribuidos", "kind": "method", "line": 167, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))", "kind": "method", "line": 180, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Estado global: mapa de pesos por hoja (no matriz)", "kind": "method", "line": 202, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 225, "name": "__init__", "signature": "def __init__(self, leaf, threshold)"}, {"doc": "Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)", "kind": "method", "line": 230, "name": "_construct_cptp_map", "signature": "def _construct_cptp_map(self)"}, {"doc": "K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.\nAproximación mean-field: operadores de Pauli escalados por gap.", "kind": "method", "line": 239, "name": "_local_jump_operator", "signature": "def _local_jump_operator(self, power)"}, {"doc": "Aplicar canal CPTP a vector de estado local (dim=2)", "kind": "method", "line": 254, "name": "apply_branching", "signature": "def apply_branching(self, state_vector)"}, {"kind": "method", "line": 275, "name": "__init__", "signature": "def __init__(self, universe, n_samples)"}, {"doc": "E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior", "kind": "method", "line": 281, "name": "_construct_hardy_state", "signature": "def _construct_hardy_state(self)"}, {"doc": "Proyector P_E en base de Fourier positiva (dim reducida)", "kind": "method", "line": 285, "name": "_szego_projector", "signature": "def _szego_projector(self)"}, {"doc": "Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))", "kind": "method", "line": 293, "name": "_evaluation_functional", "signature": "def _evaluation_functional(self, state_weights)"}, {"doc": "P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO", "kind": "method", "line": 309, "name": "project", "signature": "def project(self, state_vector)"}, {"kind": "method", "line": 355, "name": "__init__", "signature": "def __init__(self, universe, emuna)"}, {"doc": "H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)", "kind": "method", "line": 361, "name": "_effective_hamiltonian", "signature": "def _effective_hamiltonian(self)"}, {"doc": "L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}", "kind": "method", "line": 371, "name": "_modular_dissipator", "signature": "def _modular_dissipator(self, state)"}, {"doc": "G[ρ, log ρ_∞] = g[ρ, log ρ_∞]", "kind": "method", "line": 381, "name": "_nonlinear_term", "signature": "def _nonlinear_term(self, state)"}, {"doc": "Integración SDE con Euler-Maruyama.\nReturns: trayectoria [n_steps, 2, 2]", "kind": "method", "line": 388, "name": "evolve", "signature": "def evolve(self, rho0, t_span, n_steps)"}, {"kind": "method", "line": 436, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³", "kind": "method", "line": 441, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"doc": "V_loss ∝ (r⊥/a₀)^{2α} con α=0.7", "kind": "method", "line": 447, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1", "kind": "method", "line": 455, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Discordia cuántica aproximada (ejemplo: estado separable → 0)", "kind": "method", "line": 461, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 486, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Grafo dirigido con distribución de grados power-law.\nFuente: Human Connectome Project (Pilar 1).", "kind": "method", "line": 499, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)", "kind": "method", "line": 509, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "R_Q(G) = min{n | β_{n-1}(G) > 0}\nRequiere ripser (instalable en Colab: !pip install ripser)", "kind": "method", "line": 526, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Matriz de distancias shortest-path (sparse CSR)", "kind": "method", "line": 547, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}, {"doc": "t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25\nPilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)", "kind": "method", "line": 561, "name": "critical_percolation_time", "signature": "def critical_percolation_time(self)"}, {"doc": "Verificar coherencia: subgrafo > 70% del total", "kind": "method", "line": 573, "name": "is_coherent_subgraph", "signature": "def is_coherent_subgraph(self, subgraph_nodes)"}, {"kind": "method", "line": 587, "name": "__init__", "signature": "def __init__(self, network, universe)"}, {"doc": "Δ_S* = ε_c en punto excepcional", "kind": "method", "line": 591, "name": "compute_entropy_gap", "signature": "def compute_entropy_gap(self)"}, {"doc": "S_top[G] = χ(G)/|V| (número de Euler normalizado)", "kind": "method", "line": 595, "name": "compute_pontryagin_number", "signature": "def compute_pontryagin_number(self)"}, {"doc": "L[G] = Δ_S* / S_top[G]", "kind": "method", "line": 606, "name": "compute_freedom", "signature": "def compute_freedom(self)"}, {"doc": "|L[G] - 1| < 0.05 en estado crítico", "kind": "method", "line": 617, "name": "is_gauge_invariant", "signature": "def is_gauge_invariant(self)"}, {"doc": "Modelo de Ising cuántico transversal en red fractal.\nPredice t_c sin SYK₈ ni E₈.", "kind": "method", "line": 634, "name": "ising_quantum", "signature": "def ising_quantum(network)"}, {"doc": "SYK₄ estándar (sin R-simetría Spin(7)).\nPredice α sin postulado E₈.", "kind": "method", "line": 653, "name": "syk4", "signature": "def syk4(network)"}, {"doc": "Red aleatoria Erdős-Rényi sin percolación cuántica.", "kind": "method", "line": 669, "name": "random_network", "signature": "def random_network(network)"}, {"kind": "method", "line": 691, "name": "__init__", "signature": "def __init__(self, resma, myelin, network)"}, {"doc": "Predicciones RESMA 3.0", "kind": "method", "line": 698, "name": "predict_all", "signature": "def predict_all(self)"}, {"doc": "q₀ = 2π/L_E8 (sin ajuste)", "kind": "method", "line": 708, "name": "_predict_diffraction_peak", "signature": "def _predict_diffraction_peak(self)"}, {"doc": "BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)\nk = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)", "kind": "method", "line": 713, "name": "compute_bayes_factor", "signature": "def compute_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.0 – CÓDIGO COMPLETO CORREGIDO Autor: Tu nombre Fecha: 2025-11-21 Descripción: Implementación completa sin simplificaciones críticas =============================================================================", "id": "main3.py", "kind": "module", "label": "main3.py", "language": "py", "sha256": "fcebafabb04c3c0c", "symbol_count": 30, "symbols": [{"kind": "class", "line": 33, "name": "RC", "signature": "class RC"}, {"kind": "class", "line": 50, "name": "Validator", "signature": "class Validator"}, {"kind": "class", "line": 68, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 101, "name": "Universe", "signature": "class Universe"}, {"kind": "class", "line": 129, "name": "Network", "signature": "class Network"}, {"kind": "class", "line": 171, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 205, "name": "Bayes", "signature": "class Bayes"}, {"kind": "method", "line": 233, "name": "simulate", "signature": "def simulate(n_leaves, n_nodes, seed)"}, {"kind": "method", "line": 52, "name": "dim", "signature": "def dim(a)"}, {"kind": "method", "line": 56, "name": "pt", "signature": "def pt(k, o, c)"}, {"kind": "method", "line": 59, "name": "size", "signature": "def size(n)"}, {"kind": "method", "line": 74, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 78, "name": "spectral_density", "signature": "def spectral_density(self, w)"}, {"kind": "method", "line": 81, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 87, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 102, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"kind": "method", "line": 110, "name": "_gibbs", "signature": "def _gibbs(self)"}, {"kind": "method", "line": 119, "name": "_global", "signature": "def _global(self)"}, {"kind": "method", "line": 130, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"kind": "method", "line": 139, "name": "_spectral_dim", "signature": "def _spectral_dim(self, k)"}, {"kind": "method", "line": 149, "name": "_ramsey", "signature": "def _ramsey(self)"}, {"kind": "method", "line": 163, "name": "t_c", "signature": "def t_c(self)"}, {"kind": "method", "line": 172, "name": "__init__", "signature": "def __init__(self, n_modes)"}, {"kind": "method", "line": 178, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 183, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 189, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"kind": "method", "line": 192, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, pred_resma, nulls)"}, {"kind": "method", "line": 210, "name": "log_lik", "signature": "def log_lik(self, model_pred)"}, {"kind": "method", "line": 217, "name": "bf", "signature": "def bf(self)"}]}, {"doc": "============================================================================= RESMA 4.1 – VERSIÓN CORREGIDA Y VALIDADA Correcciones críticas: 1. Escala correcta para q0 2. Factor de Bayes con regularización 3. Condiciones PT-simétricas ajustadas 4. Manejo robusto de errores numéricos =============================================================================", "id": "main4.1.py", "kind": "module", "label": "main4.1.py", "language": "py", "sha256": "b3ffa3820a276569", "symbol_count": 31, "symbols": [{"kind": "class", "line": 29, "name": "RC", "signature": "class RC"}, {"kind": "class", "line": 61, "name": "Validator", "signature": "class Validator"}, {"kind": "class", "line": 82, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 123, "name": "Universe", "signature": "class Universe"}, {"kind": "class", "line": 152, "name": "Network", "signature": "class Network"}, {"kind": "class", "line": 229, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 268, "name": "Bayes", "signature": "class Bayes"}, {"kind": "method", "line": 307, "name": "simulate", "signature": "def simulate(n_leaves, n_nodes, seed)"}, {"doc": "Verifica que kappa < chi*Omega para simetría PT", "kind": "method", "line": 50, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 63, "name": "dim", "signature": "def dim(a)"}, {"doc": "Condición PT: kappa < chi*Omega", "kind": "method", "line": 68, "name": "pt", "signature": "def pt(k, o, c)"}, {"kind": "method", "line": 73, "name": "size", "signature": "def size(n)"}, {"kind": "method", "line": 88, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 92, "name": "spectral_density", "signature": "def spectral_density(self, w)"}, {"kind": "method", "line": 95, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 104, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 124, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"kind": "method", "line": 133, "name": "_gibbs", "signature": "def _gibbs(self)"}, {"kind": "method", "line": 142, "name": "_global", "signature": "def _global(self)"}, {"kind": "method", "line": 153, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Dimensión espectral corregida", "kind": "method", "line": 163, "name": "_spectral_dim", "signature": "def _spectral_dim(self, k, n_fit)"}, {"doc": "Número de Ramsey topológico", "kind": "method", "line": 199, "name": "_ramsey", "signature": "def _ramsey(self)"}, {"doc": "Tiempo crítico de percolación", "kind": "method", "line": 218, "name": "t_c", "signature": "def t_c(self)"}, {"kind": "method", "line": 230, "name": "__init__", "signature": "def __init__(self, n_modes)"}, {"kind": "method", "line": 237, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 242, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 248, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"kind": "method", "line": 251, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 269, "name": "__init__", "signature": "def __init__(self, pred_resma, nulls)"}, {"doc": "Verosimilitud con escalas físicas realistas", "kind": "method", "line": 273, "name": "log_lik", "signature": "def log_lik(self, model_pred)"}, {"doc": "Factor de Bayes con penalización de complejidad", "kind": "method", "line": 288, "name": "ln_bf", "signature": "def ln_bf(self)"}]}, {"doc": "============================================================================= RESMA 4.0 – CÓDIGO COMPLETO FINAL (FIX: NameError, UnboundLocalError & Numérico) Autor: Tu nombre Fecha: 2025-11-21 Descripción: Implementación completa, estable y corregida. =============================================================================", "id": "main4.py.py", "kind": "module", "label": "main4.py.py", "language": "py", "sha256": "6fe2e69358dd82a4", "symbol_count": 30, "symbols": [{"kind": "class", "line": 33, "name": "RC", "signature": "class RC"}, {"kind": "class", "line": 50, "name": "Validator", "signature": "class Validator"}, {"kind": "class", "line": 69, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 102, "name": "Universe", "signature": "class Universe"}, {"kind": "class", "line": 130, "name": "Network", "signature": "class Network"}, {"kind": "class", "line": 198, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 232, "name": "Bayes", "signature": "class Bayes"}, {"kind": "method", "line": 260, "name": "simulate", "signature": "def simulate(n_leaves, n_nodes, seed)"}, {"kind": "method", "line": 52, "name": "dim", "signature": "def dim(a)"}, {"kind": "method", "line": 56, "name": "pt", "signature": "def pt(k, o, c)"}, {"kind": "method", "line": 60, "name": "size", "signature": "def size(n)"}, {"kind": "method", "line": 75, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 79, "name": "spectral_density", "signature": "def spectral_density(self, w)"}, {"kind": "method", "line": 82, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 88, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 103, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"kind": "method", "line": 111, "name": "_gibbs", "signature": "def _gibbs(self)"}, {"kind": "method", "line": 120, "name": "_global", "signature": "def _global(self)"}, {"kind": "method", "line": 131, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"kind": "method", "line": 140, "name": "_spectral_dim", "signature": "def _spectral_dim(self, k, n_fit)"}, {"kind": "method", "line": 176, "name": "_ramsey", "signature": "def _ramsey(self)"}, {"kind": "method", "line": 190, "name": "t_c", "signature": "def t_c(self)"}, {"kind": "method", "line": 199, "name": "__init__", "signature": "def __init__(self, n_modes)"}, {"kind": "method", "line": 205, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 210, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 216, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"kind": "method", "line": 219, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 233, "name": "__init__", "signature": "def __init__(self, pred_resma, nulls)"}, {"kind": "method", "line": 237, "name": "log_lik", "signature": "def log_lik(self, model_pred)"}, {"kind": "method", "line": 244, "name": "ln_bf", "signature": "def ln_bf(self)"}]}, {"doc": "============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================", "id": "main5.py", "kind": "module", "label": "main5.py", "language": "py", "sha256": "8026a15d54228a9c", "symbol_count": 81, "symbols": [{"doc": "Constantes físicas y parámetros de la teoría RESMA 4.0", "kind": "class", "line": 37, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Validación de rangos físicos para todas las constantes RESMA 4.0", "kind": "class", "line": 70, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"doc": "Hoja L_i de la Resma como estado KMS mean-field con espacio de Hilbert standard.\nImplementación RESMA 4.0 con regularización Haagerup.", "kind": "class", "line": 116, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso como foliación medible sin matrices densas, con espacio de Hilbert standard.\nMemoria: O(N_leaves) con regularización de transiciones.", "kind": "class", "line": 179, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Operador Ĥ que abre la Resma cuando β_i es no trivial.\nImplementación local con operadores de salto SYK₈ (Pilar 4).", "kind": "class", "line": 260, "name": "BranchingOperator", "signature": "class BranchingOperator"}, {"doc": "Operador P̂_E: proyección teleológica no lineal.\nImplementación con muestreo Monte Carlo y espacio de Hardy H²(ℂ⁺) (Pilar 4).", "kind": "class", "line": 318, "name": "EmunaOperator", "signature": "class EmunaOperator"}, {"doc": "SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ_∞] + ξ(t)\nIntegración por Euler-Maruyama con control de precisión (Pilar 4).", "kind": "class", "line": 406, "name": "LindbladFractalDynamics", "signature": "class LindbladFractalDynamics"}, {"doc": "Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.\nImplementación 1D mean-field con R-simetría Spin(7) (Pilar 4).", "kind": "class", "line": 539, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Conectoma humano NO DIRIGIDO con homología persistente.\nImplementación sparse para escalado con conversión a grafo no dirigido (Pilar 4).", "kind": "class", "line": 604, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.\nCálculo topológico sin densidades matriciales (Pilar 4).", "kind": "class", "line": 762, "name": "FreedomInvariant", "signature": "class FreedomInvariant"}, {"doc": "Modelos nulos para cálculo de Factor de Bayes.\nBasados en teorías establecidas sin postulados holográficos de RESMA.", "kind": "class", "line": 816, "name": "NullModels", "signature": "class NullModels"}, {"doc": "Predicciones falsables contra modelos nulos teóricos.\nFactor de Bayes calculado con AIC y transformaciones logarítmicas (FIX).", "kind": "class", "line": 877, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Protocolo experimental para falsación controlada de RESMA 4.0.\nDefine setups experimentales y criterios de éxito.", "kind": "class", "line": 975, "name": "EmpiricalValidationProtocol", "signature": "class EmpiricalValidationProtocol"}, {"doc": "Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.\nDiseñado para ejecución en Google Colab (Pilar 4).", "kind": "method", "line": 1052, "name": "simulate_resma_multiverse", "signature": "def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)"}, {"doc": "α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental", "kind": "method", "line": 74, "name": "validate_dimension", "signature": "def validate_dimension(alpha, tolerance)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)", "kind": "method", "line": 84, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"doc": "Límite inferior para conectoma biológico realista", "kind": "method", "line": 95, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"doc": "Validar rango físico para dimensión espectral", "kind": "method", "line": 101, "name": "validate_spectral_dimension", "signature": "def validate_spectral_dimension(dim)"}, {"doc": "Validar tiempo de percolación contra predicción empírica", "kind": "method", "line": 106, "name": "validate_percolation_time", "signature": "def validate_percolation_time(t_c, expected, tolerance)"}, {"doc": "Validaciones post-construcción (Pilar 3)", "kind": "method", "line": 127, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.\nEvidencia: SYK₈ con Spin(7) tiene espectro continuo con gap infrarrojo.", "kind": "method", "line": 133, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización", "kind": "method", "line": 143, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"kind": "method", "line": 151, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Momentos espectrales Tr(ρ^k) para k=1..n con regularización", "kind": "method", "line": 163, "name": "_spectral_moments", "signature": "def _spectral_moments(self, n)"}, {"doc": "Peso de Haagerup para regularización del operador modular", "kind": "method", "line": 170, "name": "haagerup_weight", "signature": "def haagerup_weight(self)"}, {"doc": "Args:\n    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)\n    seed: Reproducibilidad (Pilar 4)", "kind": "method", "line": 185, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"doc": "Genera hojas con gaps espectrales distribuidos exponencialmente", "kind": "method", "line": 203, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta", "kind": "method", "line": 217, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Estado global: mapa de pesos por hoja (no matriz) con regularización", "kind": "method", "line": 239, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"doc": "Energía libre de Gibbs para validación termodinámica", "kind": "method", "line": 251, "name": "compute_gibbs_free_energy", "signature": "def compute_gibbs_free_energy(self)"}, {"kind": "method", "line": 266, "name": "__init__", "signature": "def __init__(self, leaf, threshold)"}, {"doc": "Defecto de holonomía como variación del gap espectral", "kind": "method", "line": 272, "name": "_compute_holonomy", "signature": "def _compute_holonomy(self)"}, {"doc": "Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)", "kind": "method", "line": 276, "name": "_construct_cptp_map", "signature": "def _construct_cptp_map(self)"}, {"doc": "K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.\nAproximación mean-field: operadores de Pauli escalados por gap SYK₈.", "kind": "method", "line": 284, "name": "_local_jump_operator", "signature": "def _local_jump_operator(self, power)"}, {"doc": "Aplicar canal CPTP a vector de estado local (dim=2) con normalización", "kind": "method", "line": 300, "name": "apply_branching", "signature": "def apply_branching(self, state_vector)"}, {"kind": "method", "line": 324, "name": "__init__", "signature": "def __init__(self, universe, n_samples)"}, {"doc": "E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior", "kind": "method", "line": 331, "name": "_construct_hardy_state", "signature": "def _construct_hardy_state(self)"}, {"doc": "Proyector P_E en base de Fourier positiva (dim reducida)", "kind": "method", "line": 335, "name": "_szego_projector", "signature": "def _szego_projector(self)"}, {"doc": "Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))", "kind": "method", "line": 345, "name": "_evaluation_functional", "signature": "def _evaluation_functional(self, state_weights)"}, {"doc": "P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO", "kind": "method", "line": 361, "name": "project", "signature": "def project(self, state_vector)"}, {"doc": "Calcular overlap teleológico con estado objetivo", "kind": "method", "line": 396, "name": "compute_teleological_overlap", "signature": "def compute_teleological_overlap(self)"}, {"kind": "method", "line": 412, "name": "__init__", "signature": "def __init__(self, universe, emuna)"}, {"doc": "H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)", "kind": "method", "line": 419, "name": "_effective_hamiltonian", "signature": "def _effective_hamiltonian(self)"}, {"doc": "L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización", "kind": "method", "line": 432, "name": "_modular_dissipator", "signature": "def _modular_dissipator(self, state)"}, {"doc": "G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo", "kind": "method", "line": 444, "name": "_nonlinear_term", "signature": "def _nonlinear_term(self, state)"}, {"doc": "Término estocástico ξ(t) con correlaciones cuánticas", "kind": "method", "line": 452, "name": "_stochastic_term", "signature": "def _stochastic_term(self, dt)"}, {"doc": "Integración SDE con Euler-Maruyama y control de paso adaptativo.\nReturns: trayectoria [n_steps, 2, 2]", "kind": "method", "line": 459, "name": "evolve", "signature": "def evolve(self, rho0, t_span, n_steps)"}, {"doc": "Normalizar matriz densidad y forzar hermiticidad", "kind": "method", "line": 500, "name": "_normalize_density_matrix", "signature": "def _normalize_density_matrix(self, state)"}, {"doc": "Verificar si el estado es físico (hermitiano, traza=1, positivo)", "kind": "method", "line": 509, "name": "_is_physical_state", "signature": "def _is_physical_state(self, state)"}, {"doc": "Corregir estado no físico proyectando en el cono de estados válidos", "kind": "method", "line": 523, "name": "_correct_non_physical_state", "signature": "def _correct_non_physical_state(self, state)"}, {"kind": "method", "line": 548, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³", "kind": "method", "line": 555, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"doc": "V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)", "kind": "method", "line": 561, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"doc": "Campo escalar masivo para estabilización de Spin(7)", "kind": "method", "line": 569, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"doc": "Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos", "kind": "method", "line": 573, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Discordia cuántica aproximada con corrección PT", "kind": "method", "line": 579, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 610, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.\nSOLUCIÓN RESMA 4.0: Conversión explícita con to_undirected().", "kind": "method", "line": 625, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.\nSOLUCIÓN RESMA 4.0: Uso de función especializada de NetworkX.", "kind": "method", "line": 649, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "R_Q(G) = min{n | β_{n-1}(G) > 0}\nRequiere ripser (instalable en Colab: !pip install ripser)", "kind": "method", "line": 680, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Calcular números de Betti para análisis topológico", "kind": "method", "line": 706, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz de distancias shortest-path (sparse CSR) para homología", "kind": "method", "line": 722, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}, {"doc": "t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25\nPilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)", "kind": "method", "line": 735, "name": "critical_percolation_time", "signature": "def critical_percolation_time(self)"}, {"doc": "Verificar coherencia: subgrafo > 70% del total", "kind": "method", "line": 747, "name": "is_coherent_subgraph", "signature": "def is_coherent_subgraph(self, subgraph_nodes)"}, {"doc": "Entropía de la red basada en distribución de grados", "kind": "method", "line": 751, "name": "compute_network_entropy", "signature": "def compute_network_entropy(self)"}, {"kind": "method", "line": 768, "name": "__init__", "signature": "def __init__(self, network, universe)"}, {"doc": "Δ_S* = ε_c en punto excepcional con corrección de regularización", "kind": "method", "line": 772, "name": "compute_entropy_gap", "signature": "def compute_entropy_gap(self)"}, {"doc": "S_top[G] = χ(G)/|V| (número de Euler normalizado)", "kind": "method", "line": 776, "name": "compute_pontryagin_number", "signature": "def compute_pontryagin_number(self)"}, {"doc": "L[G] = Δ_S* / S_top[G] con protección de división por cero", "kind": "method", "line": 792, "name": "compute_freedom", "signature": "def compute_freedom(self)"}, {"doc": "|L[G] - 1| < 0.05 en estado crítico (invariante de libertad)", "kind": "method", "line": 803, "name": "is_gauge_invariant", "signature": "def is_gauge_invariant(self)"}, {"doc": "Modelo de Ising cuántico transversal en red fractal.\nPredice t_c sin SYK₈ ni E₈ (teoría efectiva estándar).", "kind": "method", "line": 823, "name": "ising_quantum", "signature": "def ising_quantum(network)"}, {"doc": "SYK₄ estándar (sin R-simetría Spin(7) ni E₈).\nPredice α sin postulado de retículo.", "kind": "method", "line": 843, "name": "syk4", "signature": "def syk4(network)"}, {"doc": "Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.", "kind": "method", "line": 860, "name": "random_network", "signature": "def random_network(network)"}, {"kind": "method", "line": 883, "name": "__init__", "signature": "def __init__(self, resma, myelin, network, freedom)"}, {"doc": "Predicciones RESMA 4.0 con valores empíricos objetivo", "kind": "method", "line": 891, "name": "predict_all", "signature": "def predict_all(self)"}, {"doc": "q₀ = 2π/L_E8 (predicción de difracción UASED)", "kind": "method", "line": 906, "name": "_predict_diffraction_peak", "signature": "def _predict_diffraction_peak(self)"}, {"doc": "log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)\nFIX RESMA 4.0: Usar espacio logarítmico para evitar desbordamiento.", "kind": "method", "line": 912, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}, {"kind": "method", "line": 981, "name": "__init__", "signature": "def __init__(self, predictions)"}, {"doc": "Definir protocolos experimentales con parámetros técnicos", "kind": "method", "line": 985, "name": "_define_protocols", "signature": "def _define_protocols(self)"}, {"doc": "Evaluar viabilidad del protocolo completo", "kind": "method", "line": 1014, "name": "evaluate_feasibility", "signature": "def evaluate_feasibility(self, budget, time_limit)"}, {"doc": "Simular resultado experimental con ruido realista", "kind": "method", "line": 1027, "name": "simulate_experimental_outcome", "signature": "def simulate_experimental_outcome(self, protocol_name)"}]}, {"id": "monitor_extremo.py", "kind": "module", "label": "monitor_extremo.py", "language": "py", "sha256": "8d5aca37dbd5b558", "symbol_count": 12, "symbols": [{"kind": "function", "line": 19, "name": "setup_matplotlib_for_plotting", "signature": "def setup_matplotlib_for_plotting()"}, {"doc": "Implementación del Sovereignty Monitor basada en RESMA", "kind": "class", "line": 26, "name": "SovereigntyMonitor", "signature": "class SovereigntyMonitor"}, {"doc": "Modelo grande diseñado para colapsar con entrenamiento extremo", "kind": "class", "line": 72, "name": "ModeloGrande", "signature": "class ModeloGrande(Module)"}, {"doc": "Genera datos diseñados específicamente para causar colapso", "kind": "method", "line": 103, "name": "generar_datos_toxico", "signature": "def generar_datos_toxico()"}, {"doc": "Experimento diseñado para forzar el colapso del modelo", "kind": "method", "line": 126, "name": "experimento_colapso_forzado", "signature": "def experimento_colapso_forzado()"}, {"doc": "Genera gráficos del experimento extremo", "kind": "method", "line": 339, "name": "generar_graficos_extremos", "signature": "def generar_graficos_extremos(historial)"}, {"kind": "method", "line": 28, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"doc": "Calcula la métrica L (libertad) de una matriz de pesos", "kind": "method", "line": 31, "name": "calcular_libertad", "signature": "def calcular_libertad(self, weights)"}, {"doc": "Evalúa el régimen del modelo", "kind": "method", "line": 63, "name": "evaluar_regimen", "signature": "def evaluar_regimen(self, L)"}, {"kind": "method", "line": 74, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 88, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 100, "name": "get_linear_layers", "signature": "def get_linear_layers(self)"}]}, {"id": "quick_monitor.py", "kind": "module", "label": "quick_monitor.py", "language": "py", "sha256": "06e69abaa463a1f4", "symbol_count": 12, "symbols": [{"kind": "function", "line": 19, "name": "setup_matplotlib_for_plotting", "signature": "def setup_matplotlib_for_plotting()"}, {"doc": "Implementación del Sovereignty Monitor basada en RESMA", "kind": "class", "line": 26, "name": "SovereigntyMonitor", "signature": "class SovereigntyMonitor"}, {"doc": "Modelo CNN pequeño optimizado para entrenamiento rápido", "kind": "class", "line": 72, "name": "ModeloMNISTPequeno", "signature": "class ModeloMNISTPequeno(Module)"}, {"doc": "Genera datos sintéticos tipo MNIST para experimento rápido", "kind": "method", "line": 95, "name": "generar_datos_mnist_rapido", "signature": "def generar_datos_mnist_rapido()"}, {"doc": "Entrena modelo con monitoreo L en tiempo real", "kind": "method", "line": 116, "name": "entrenar_modelo_rapido", "signature": "def entrenar_modelo_rapido()"}, {"doc": "Genera gráficos de resultados del experimento rápido", "kind": "method", "line": 295, "name": "generar_graficos_rapido", "signature": "def generar_graficos_rapido(historial)"}, {"kind": "method", "line": 28, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"doc": "Calcula la métrica L (libertad) de una matriz de pesos", "kind": "method", "line": 31, "name": "calcular_libertad", "signature": "def calcular_libertad(self, weights)"}, {"doc": "Evalúa el régimen del modelo", "kind": "method", "line": 63, "name": "evaluar_regimen", "signature": "def evaluar_regimen(self, L)"}, {"kind": "method", "line": 74, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 83, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 92, "name": "get_linear_layers", "signature": "def get_linear_layers(self)"}]}, {"id": "resma2/main_experiment.py", "kind": "module", "label": "main_experiment.py", "language": "py", "sha256": "e12fb28dd61a5772", "symbol_count": 2, "symbols": [{"kind": "function", "line": 26, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "function", "line": 31, "name": "run_experiment", "signature": "def run_experiment()"}]}, {"id": "resma2/main_experiments.py", "kind": "module", "label": "main_experiments.py", "language": "py", "sha256": "e61b3a8b45fb7533", "symbol_count": 3, "symbols": [{"kind": "function", "line": 24, "name": "inject_noise", "signature": "def inject_noise(x, sigma)"}, {"kind": "function", "line": 27, "name": "train_epoch", "signature": "def train_epoch(model, loader, optim, obs, epoch)"}, {"kind": "function", "line": 61, "name": "run", "signature": "def run()"}]}, {"id": "resma2/monitor.py", "kind": "module", "label": "monitor.py", "language": "py", "sha256": "19e73e49d6f06d45", "symbol_count": 9, "symbols": [{"kind": "class", "line": 12, "name": "Regime", "signature": "class Regime(Enum)"}, {"kind": "class", "line": 18, "name": "LayerDiagnostics", "signature": "class LayerDiagnostics"}, {"kind": "class", "line": 27, "name": "EpochSnapshot", "signature": "class EpochSnapshot"}, {"kind": "class", "line": 35, "name": "SovereigntyMonitor", "signature": "class SovereigntyMonitor"}, {"kind": "method", "line": 36, "name": "__init__", "signature": "def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_layers, verbose)"}, {"kind": "method", "line": 50, "name": "_extract_weights", "signature": "def _extract_weights(self, model)"}, {"kind": "method", "line": 58, "name": "_calculate_svd_metrics", "signature": "def _calculate_svd_metrics(self, weight_matrix)"}, {"kind": "method", "line": 84, "name": "calcular_libertad", "signature": "def calcular_libertad(self, weights)"}, {"kind": "method", "line": 92, "name": "calculate", "signature": "def calculate(self, model)"}]}, {"id": "resma2/resma_app_mnist.py", "kind": "module", "label": "resma_app_mnist.py", "language": "py", "sha256": "52d4cbf0f065871e", "symbol_count": 3, "symbols": [{"doc": "Inyecta ruido gaussiano simulando fluctuaciones de vacío", "kind": "function", "line": 25, "name": "add_quantum_noise", "signature": "def add_quantum_noise(tensor, noise_factor)"}, {"kind": "function", "line": 30, "name": "train", "signature": "def train(model, device, train_loader, optimizer, epoch, observer)"}, {"kind": "function", "line": 65, "name": "main", "signature": "def main()"}]}, {"id": "resma2/resma_breakpoint.py", "kind": "module", "label": "resma_breakpoint.py", "language": "py", "sha256": "32c59a574846b1af", "symbol_count": 1, "symbols": [{"kind": "function", "line": 6, "name": "find_break_point", "signature": "def find_break_point()"}]}, {"id": "resma2/resma_combat_test.py", "kind": "module", "label": "resma_combat_test.py", "language": "py", "sha256": "775efa3252cef226", "symbol_count": 1, "symbols": [{"kind": "function", "line": 11, "name": "combat_test", "signature": "def combat_test()"}]}, {"id": "resma2/resma_core.py", "kind": "module", "label": "resma_core.py", "language": "py", "sha256": "c607524c7edb8688", "symbol_count": 10, "symbols": [{"kind": "class", "line": 14, "name": "PTSymmetricActivation", "signature": "class PTSymmetricActivation(Module)"}, {"kind": "class", "line": 35, "name": "E8LatticeLayer", "signature": "class E8LatticeLayer(Module)"}, {"kind": "class", "line": 65, "name": "RESMABrain", "signature": "class RESMABrain(Module)"}, {"kind": "method", "line": 15, "name": "__init__", "signature": "def __init__(self, omega, chi, kappa_init)"}, {"kind": "method", "line": 26, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 37, "name": "__init__", "signature": "def __init__(self, in_features, out_features, q_order)"}, {"kind": "method", "line": 47, "name": "_generate_ramsey_mask", "signature": "def _generate_ramsey_mask(self)"}, {"kind": "method", "line": 59, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 66, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, output_dim)"}, {"kind": "method", "line": 74, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "resma2/resma_noise_phase_test.py", "kind": "module", "label": "resma_noise_phase_test.py", "language": "py", "sha256": "a2728a2a77e5a754", "symbol_count": 2, "symbols": [{"kind": "function", "line": 34, "name": "add_noise", "signature": "def add_noise(x, sigma)"}, {"kind": "function", "line": 37, "name": "measure_entropy", "signature": "def measure_entropy(gate_tensor)"}]}, {"id": "resma2/resma_observer.py", "kind": "module", "label": "resma_observer.py", "language": "py", "sha256": "ca9fb68ac988d683", "symbol_count": 9, "symbols": [{"doc": "Snapshot del estado físico-estructural de la red", "kind": "class", "line": 27, "name": "QuantumState", "signature": "class QuantumState"}, {"kind": "class", "line": 39, "name": "RESMAObserver", "signature": "class RESMAObserver"}, {"kind": "method", "line": 36, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 40, "name": "__init__", "signature": "def __init__(self, model, epsilon_c)"}, {"doc": "Inyecta sondas en las capas PT para leer telemetría en tiempo real", "kind": "method", "line": 51, "name": "_register_hooks", "signature": "def _register_hooks(self)"}, {"doc": "Ejecutar al final de cada época de entrenamiento/validación.\nFusiona métricas y determina la fase.", "kind": "method", "line": 68, "name": "step", "signature": "def step(self, epoch)"}, {"doc": "Imprime reporte formateado a consola", "kind": "method", "line": 106, "name": "report", "signature": "def report(self, state)"}, {"doc": "Genera el diagrama de fase: Estructura vs Dinámica", "kind": "method", "line": 118, "name": "plot_phase_space", "signature": "def plot_phase_space(self, save_path)"}, {"kind": "method", "line": 53, "name": "hook_fn", "signature": "def hook_fn(module, input, output)"}]}, {"id": "resma2/resma_overload.py", "kind": "module", "label": "resma_overload.py", "language": "py", "sha256": "18326e8ee0ba78a1", "symbol_count": 1, "symbols": [{"kind": "function", "line": 6, "name": "overload_test", "signature": "def overload_test()"}]}, {"id": "resma2/resma_train.py", "kind": "module", "label": "resma_train.py", "language": "py", "sha256": "7902d84183a30cb5", "symbol_count": 0, "symbols": []}, {"id": "resma2/resma_vision.py", "kind": "module", "label": "resma_vision.py", "language": "py", "sha256": "eac548a3d5ac1932", "symbol_count": 2, "symbols": [{"kind": "function", "line": 11, "name": "add_noise", "signature": "def add_noise(tensor, factor)"}, {"kind": "function", "line": 14, "name": "visualize_resma_perception", "signature": "def visualize_resma_perception()"}]}, {"id": "resma2/resma_vision_trained.py", "kind": "module", "label": "resma_vision_trained.py", "language": "py", "sha256": "b566b0f92a05d948", "symbol_count": 2, "symbols": [{"kind": "function", "line": 11, "name": "add_noise", "signature": "def add_noise(tensor, factor)"}, {"kind": "function", "line": 14, "name": "visualize_trained_perception", "signature": "def visualize_trained_perception()"}]}, {"doc": "============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (CÓDIGO DE PRODUCCIÓN COMPLETO) ============================================================================= GUARDAR COMO: resma4.12_fixed.py FIX CRÍTICO: Bug en MyelinCavity._loss_potential (n_nodes → n_modes)", "id": "resma4.10.py", "kind": "module", "label": "resma4.10.py", "language": "py", "sha256": "26828305c069643c", "symbol_count": 57, "symbols": [{"kind": "class", "line": 35, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"kind": "class", "line": 72, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"doc": "Operador de desdoblamiento D̂_G(φ) sobre el álgebra E8 genuina.\nConstrucción: sistema de raíces E8 → base de Chevalley → representación adjunta 248.", "kind": "class", "line": 112, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"kind": "class", "line": 431, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"kind": "class", "line": 459, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 506, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 614, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"kind": "class", "line": 782, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 818, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"kind": "class", "line": 865, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"kind": "method", "line": 877, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"kind": "method", "line": 935, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"doc": "Convierte objetos a formato serializable de forma segura.\n\nArgs:\n    obj: Objeto a serializar\n    depth: Nivel de profundidad actual (auto-incremental)\n    max_depth: Profundidad máxima permitida\n    _visited: Diccionario de objetos ya procesados (para referencias circulares)", "kind": "method", "line": 963, "name": "_make_serializable", "signature": "def _make_serializable(obj, depth, max_depth, _visited)"}, {"kind": "method", "line": 1067, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)"}, {"kind": "method", "line": 51, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 75, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 87, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"kind": "method", "line": 90, "name": "modulation_factor", "signature": "def modulation_factor(self)"}, {"kind": "method", "line": 93, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 103, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 118, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"doc": "Genera las 240 raíces de E8 en R⁸.\nRetorna array (240,8) con forma: [positivas (120) | negativas (120)]\ndonde neg[j] = -pos[j].", "kind": "method", "line": 149, "name": "_generate_e8_roots", "signature": "def _generate_e8_roots()"}, {"doc": "Índice global (0..239) de una raíz.", "kind": "method", "line": 189, "name": "_idx", "signature": "def _idx(self, root_vec)"}, {"doc": "Constantes N_{α,β} para toda raíz α, β con α+β también raíz.\nRetorna dict {(i,j): N_{α_i,α_j}} con ambas orientaciones.\nConvención Chevalley:\n  N_{α,β} = -(p+1) si α < β,  (p+1) si α > β,\n  donde p es el entero max con β - pα raíz.", "kind": "method", "line": 197, "name": "_compute_structure_constants", "signature": "def _compute_structure_constants(self)"}, {"doc": "Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.\n\nBase: |H₀⟩..|H₇⟩ (0-7), |E_α₀⟩..|E_α₁₁₉⟩ (8-127), |E_{-α₀}⟩..|E_{-α₁₁₉}⟩ (128-247)\ncon roots[gi] = α para gi en 0..119, roots[gi+120] = -α.\n\nArgs:\n    cartan:     array[8] coeficientes c_i para H_i.\n    roots_coeff: array[240] coeficientes d_γ para E_γ.", "kind": "method", "line": 238, "name": "_adjoint_matrix", "signature": "def _adjoint_matrix(self, cartan, roots_coeff)"}, {"doc": "Construye 3 generadores genuinos del álgebra E8 en la adjunta.\nCada uno corresponde a una dirección física del formalismo Garnier T³:\n\n  G₀ = H₁         (escala C₀ = 1.0, tiempo físico)\n  G₂ = H₂         (escala C₂ = 2.7, tiempo crítico)\n  G₃ = E_{α₁} + E_{-α₁}  (escala C₃ = 7.3, tiempo teleológico)\n\nLas raíces simples de E8 son:\n  α₁ = (1,-1,0,0,0,0,0,0), α₂ = (0,1,-1,0,0,0,0,0)", "kind": "method", "line": 330, "name": "_construir_generadores_e8", "signature": "def _construir_generadores_e8(self)"}, {"kind": "method", "line": 399, "name": "_hadamard_generalizado", "signature": "def _hadamard_generalizado(self)"}, {"kind": "method", "line": 408, "name": "operator", "signature": "def operator(self)"}, {"kind": "method", "line": 423, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 432, "name": "__init__", "signature": "def __init__(self, garnier)"}, {"kind": "method", "line": 436, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red, b1)"}, {"kind": "method", "line": 443, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red, b1)"}, {"kind": "method", "line": 466, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 470, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 476, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 507, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"kind": "method", "line": 543, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"kind": "method", "line": 554, "name": "_generate_complete_measure", "signature": "def _generate_complete_measure(self)"}, {"kind": "method", "line": 583, "name": "_aplicar_modulacion_garnier", "signature": "def _aplicar_modulacion_garnier(self, measure)"}, {"kind": "method", "line": 594, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 603, "name": "_calcular_libertad", "signature": "def _calcular_libertad(self)"}, {"kind": "method", "line": 606, "name": "_calcular_coherencia", "signature": "def _calcular_coherencia(self)"}, {"kind": "method", "line": 615, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"kind": "method", "line": 653, "name": "_generate_realistic_modular_network", "signature": "def _generate_realistic_modular_network(self)"}, {"kind": "method", "line": 727, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"kind": "method", "line": 735, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"kind": "method", "line": 757, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"kind": "method", "line": 761, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"kind": "method", "line": 769, "name": "_validar_axioma_6", "signature": "def _validar_axioma_6(self)"}, {"kind": "method", "line": 783, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"kind": "method", "line": 800, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 805, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 811, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 819, "name": "__init__", "signature": "def __init__(self, universe, network, myelin)"}, {"kind": "method", "line": 824, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}, {"kind": "method", "line": 867, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 872, "name": "log_resources", "signature": "def log_resources()"}]}, {"doc": "============================================================================= RESMA 4.13 – VECTORIZACIÓN MASIVA (HACK PRO) ============================================================================= OPTIMIZACIÓN: Inicialización y medida cuántica completamente vectorizadas en operaciones NumPy (matrices (N,500) + producto matricial Q@Q.T).", "id": "resma4.13.py", "kind": "module", "label": "resma4.13.py", "language": "py", "sha256": "4d18627ca52e04cc", "symbol_count": 57, "symbols": [{"kind": "class", "line": 35, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"kind": "class", "line": 72, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"doc": "Operador de desdoblamiento D̂_G(φ) sobre el álgebra E8 genuina.\nConstrucción: sistema de raíces E8 → base de Chevalley → representación adjunta 248.", "kind": "class", "line": 112, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"kind": "class", "line": 431, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"kind": "class", "line": 459, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 506, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 639, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"kind": "class", "line": 807, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 843, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"kind": "class", "line": 890, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"kind": "method", "line": 902, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"kind": "method", "line": 960, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"doc": "Convierte objetos a formato serializable de forma segura.\n\nArgs:\n    obj: Objeto a serializar\n    depth: Nivel de profundidad actual (auto-incremental)\n    max_depth: Profundidad máxima permitida\n    _visited: Diccionario de objetos ya procesados (para referencias circulares)", "kind": "method", "line": 988, "name": "_make_serializable", "signature": "def _make_serializable(obj, depth, max_depth, _visited)"}, {"kind": "method", "line": 1092, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)"}, {"kind": "method", "line": 51, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 75, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 87, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"kind": "method", "line": 90, "name": "modulation_factor", "signature": "def modulation_factor(self)"}, {"kind": "method", "line": 93, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 103, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 118, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"doc": "Genera las 240 raíces de E8 en R⁸.\nRetorna array (240,8) con forma: [positivas (120) | negativas (120)]\ndonde neg[j] = -pos[j].", "kind": "method", "line": 149, "name": "_generate_e8_roots", "signature": "def _generate_e8_roots()"}, {"doc": "Índice global (0..239) de una raíz.", "kind": "method", "line": 189, "name": "_idx", "signature": "def _idx(self, root_vec)"}, {"doc": "Constantes N_{α,β} para toda raíz α, β con α+β también raíz.\nRetorna dict {(i,j): N_{α_i,α_j}} con ambas orientaciones.\nConvención Chevalley:\n  N_{α,β} = -(p+1) si α < β,  (p+1) si α > β,\n  donde p es el entero max con β - pα raíz.", "kind": "method", "line": 197, "name": "_compute_structure_constants", "signature": "def _compute_structure_constants(self)"}, {"doc": "Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.\n\nBase: |H₀⟩..|H₇⟩ (0-7), |E_α₀⟩..|E_α₁₁₉⟩ (8-127), |E_{-α₀}⟩..|E_{-α₁₁₉}⟩ (128-247)\ncon roots[gi] = α para gi en 0..119, roots[gi+120] = -α.\n\nArgs:\n    cartan:     array[8] coeficientes c_i para H_i.\n    roots_coeff: array[240] coeficientes d_γ para E_γ.", "kind": "method", "line": 238, "name": "_adjoint_matrix", "signature": "def _adjoint_matrix(self, cartan, roots_coeff)"}, {"doc": "Construye 3 generadores genuinos del álgebra E8 en la adjunta.\nCada uno corresponde a una dirección física del formalismo Garnier T³:\n\n  G₀ = H₁         (escala C₀ = 1.0, tiempo físico)\n  G₂ = H₂         (escala C₂ = 2.7, tiempo crítico)\n  G₃ = E_{α₁} + E_{-α₁}  (escala C₃ = 7.3, tiempo teleológico)\n\nLas raíces simples de E8 son:\n  α₁ = (1,-1,0,0,0,0,0,0), α₂ = (0,1,-1,0,0,0,0,0)", "kind": "method", "line": 330, "name": "_construir_generadores_e8", "signature": "def _construir_generadores_e8(self)"}, {"kind": "method", "line": 399, "name": "_hadamard_generalizado", "signature": "def _hadamard_generalizado(self)"}, {"kind": "method", "line": 408, "name": "operator", "signature": "def operator(self)"}, {"kind": "method", "line": 423, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 432, "name": "__init__", "signature": "def __init__(self, garnier)"}, {"kind": "method", "line": 436, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red, b1)"}, {"kind": "method", "line": 443, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red, b1)"}, {"kind": "method", "line": 466, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 470, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 476, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 507, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"kind": "method", "line": 543, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"kind": "method", "line": 553, "name": "_generate_complete_measure", "signature": "def _generate_complete_measure(self)"}, {"kind": "method", "line": 608, "name": "_aplicar_modulacion_garnier", "signature": "def _aplicar_modulacion_garnier(self, measure)"}, {"kind": "method", "line": 619, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 628, "name": "_calcular_libertad", "signature": "def _calcular_libertad(self)"}, {"kind": "method", "line": 631, "name": "_calcular_coherencia", "signature": "def _calcular_coherencia(self)"}, {"kind": "method", "line": 640, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"kind": "method", "line": 678, "name": "_generate_realistic_modular_network", "signature": "def _generate_realistic_modular_network(self)"}, {"kind": "method", "line": 752, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"kind": "method", "line": 760, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"kind": "method", "line": 782, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"kind": "method", "line": 786, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"kind": "method", "line": 794, "name": "_validar_axioma_6", "signature": "def _validar_axioma_6(self)"}, {"kind": "method", "line": 808, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"kind": "method", "line": 825, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 830, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 836, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 844, "name": "__init__", "signature": "def __init__(self, universe, network, myelin)"}, {"kind": "method", "line": 849, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}, {"kind": "method", "line": 892, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 897, "name": "log_resources", "signature": "def log_resources()"}]}, {"doc": "============================================================================= RESMA 4.2 – IMPLEMENTACIÓN COMPLETA CON CORRECCIONES NUMÉRICAS Integración de RESMA 4.0 (teoría completa) + RESMA 4.1 (fixes numéricos) Autor: Colaboración Claude + Usuario Fecha: 2025-11-21 =============================================================================", "id": "resma4.2.py", "kind": "module", "label": "resma4.2.py", "language": "py", "sha256": "877dce466f87b307", "symbol_count": 45, "symbols": [{"doc": "Constantes físicas RESMA 4.0 con correcciones PT-simétricas", "kind": "class", "line": 38, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"kind": "class", "line": 78, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"doc": "Hoja L_i como estado KMS con espacio de Hilbert standard", "kind": "class", "line": 109, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso como foliación medible, memoria O(N_leaves)", "kind": "class", "line": 162, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "P̂_E: proyección teleológica no lineal en H²(ℂ⁺)", "kind": "class", "line": 224, "name": "EmunaOperator", "signature": "class EmunaOperator"}, {"doc": "Cavidad dieléctrica H = H₀ + iV_loss con Spin(7)", "kind": "class", "line": 291, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Conectoma NO DIRIGIDO con homología persistente", "kind": "class", "line": 351, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "Predicciones con BF logarítmico", "kind": "class", "line": 474, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline RESMA 4.2 completo", "kind": "method", "line": 556, "name": "simulate_resma_complete", "signature": "def simulate_resma_complete(n_leaves, n_nodes, seed)"}, {"doc": "Verificar condición PT: κ < χΩ", "kind": "method", "line": 67, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 80, "name": "validate_dimension", "signature": "def validate_dimension(alpha, tolerance)"}, {"kind": "method", "line": 88, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"kind": "method", "line": 96, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"kind": "method", "line": 101, "name": "validate_spectral_dimension", "signature": "def validate_spectral_dimension(dim)"}, {"kind": "method", "line": 117, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "ρ(ω) con regularización UV", "kind": "method", "line": 121, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "S = -∫ ρ log ρ dω", "kind": "method", "line": 126, "name": "modular_entropy", "signature": "def modular_entropy(self)"}, {"doc": "Distancia de Bures W₂(ρ₁, ρ₂)", "kind": "method", "line": 136, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Peso de Haagerup para regularización", "kind": "method", "line": 154, "name": "haagerup_weight", "signature": "def haagerup_weight(self)"}, {"kind": "method", "line": 165, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"kind": "method", "line": 176, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))", "kind": "method", "line": 189, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Estado global: pesos por hoja", "kind": "method", "line": 205, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"doc": "F = -ln(Tr(μ)) / β", "kind": "method", "line": 216, "name": "compute_gibbs_free_energy", "signature": "def compute_gibbs_free_energy(self)"}, {"kind": "method", "line": 227, "name": "__init__", "signature": "def __init__(self, universe, n_samples)"}, {"doc": "E(z) ∈ H²(ℂ⁺)", "kind": "method", "line": 234, "name": "_construct_hardy_state", "signature": "def _construct_hardy_state(self)"}, {"doc": "Proyector en frecuencias positivas", "kind": "method", "line": 238, "name": "_szego_projector", "signature": "def _szego_projector(self)"}, {"doc": "Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)", "kind": "method", "line": 246, "name": "_evaluation_functional", "signature": "def _evaluation_functional(self, state_weights)"}, {"doc": "P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)", "kind": "method", "line": 258, "name": "project", "signature": "def project(self, state_vector)"}, {"kind": "method", "line": 297, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "H₀: dispersión Ω(q) = Ω₀ + q² + χq³", "kind": "method", "line": 304, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"doc": "V_loss ∝ (r/a₀)^(2α)", "kind": "method", "line": 310, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"doc": "Campo escalar para estabilización Spin(7)", "kind": "method", "line": 317, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"doc": "κ < χΩ", "kind": "method", "line": 321, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Coherencia cuántica con verificación espectral", "kind": "method", "line": 327, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 354, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Scale-free → NO DIRIGIDO", "kind": "method", "line": 366, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "d_s = -2 lim log N(λ)/log λ", "kind": "method", "line": 381, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "R_Q(G) = min{n | β_{n-1}(G) > 0}", "kind": "method", "line": 410, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Números de Betti β₀, β₁", "kind": "method", "line": 429, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz de distancias para homología", "kind": "method", "line": 445, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}, {"doc": "t_c = 21 · (N/N₀)^0.25 / log R_Q", "kind": "method", "line": 461, "name": "critical_percolation_time", "signature": "def critical_percolation_time(self)"}, {"kind": "method", "line": 477, "name": "__init__", "signature": "def __init__(self, universe, myelin, network)"}, {"doc": "Predicciones RESMA 4.2", "kind": "method", "line": 483, "name": "predict_all", "signature": "def predict_all(self)"}, {"doc": "ln(BF) con AIC", "kind": "method", "line": 496, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.1 – FIX: FrozenInstanceError + Reanudación Inteligente =============================================================================", "id": "resma4.3.py", "kind": "module", "label": "resma4.3.py", "language": "py", "sha256": "cece807a62e9f031", "symbol_count": 41, "symbols": [{"kind": "class", "line": 33, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"doc": "Guardado atómico con backup", "kind": "method", "line": 54, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"doc": "Cargar checkpoint con fallback", "kind": "method", "line": 84, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"kind": "class", "line": 110, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Hoja KMS - INMUTABLE pero con caché externo", "kind": "class", "line": 142, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso con construcción lazy", "kind": "class", "line": 190, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 269, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"kind": "class", "line": 297, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 352, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"kind": "class", "line": 485, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline con reanudación inteligente desde checkpoints", "kind": "method", "line": 545, "name": "simulate_resma_with_checkpointing", "signature": "def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)"}, {"kind": "method", "line": 35, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 40, "name": "check_memory_limit", "signature": "def check_memory_limit()"}, {"kind": "method", "line": 49, "name": "log_resources", "signature": "def log_resources()"}, {"kind": "method", "line": 127, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 150, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 154, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Distancia Bures con caché EXTERNO (no en instancia)", "kind": "method", "line": 158, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 193, "name": "__init__", "signature": "def __init__(self, n_leaves, seed)"}, {"kind": "method", "line": 215, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Matriz de medida con guardado incremental", "kind": "method", "line": 227, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"kind": "method", "line": 254, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 271, "name": "validate_dimension", "signature": "def validate_dimension(alpha, tolerance)"}, {"kind": "method", "line": 279, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"kind": "method", "line": 287, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"kind": "method", "line": 292, "name": "validate_spectral_dimension", "signature": "def validate_spectral_dimension(dim)"}, {"kind": "method", "line": 302, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 312, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 317, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 323, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 326, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"kind": "method", "line": 331, "name": "coherence_quantum", "signature": "def coherence_quantum(self)"}, {"kind": "method", "line": 353, "name": "__init__", "signature": "def __init__(self, n_nodes, seed)"}, {"doc": "Generar grafo por lotes", "kind": "method", "line": 372, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "Dimensión espectral con matriz sparse", "kind": "method", "line": 401, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "Ramsey topológico", "kind": "method", "line": 425, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Números de Betti", "kind": "method", "line": 444, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz de distancias sparse", "kind": "method", "line": 460, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}, {"doc": "Tiempo crítico de percolación", "kind": "method", "line": 476, "name": "critical_percolation_time", "signature": "def critical_percolation_time(self)"}, {"kind": "method", "line": 486, "name": "__init__", "signature": "def __init__(self, universe, myelin, network)"}, {"doc": "ln(BF)", "kind": "method", "line": 492, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.2 – REANUDACIÓN REAL + SERIALIZACIÓN DE OBJETOS =============================================================================", "id": "resma4.4.py", "kind": "module", "label": "resma4.4.py", "language": "py", "sha256": "daea9392c9ed71bc", "symbol_count": 36, "symbols": [{"kind": "class", "line": 34, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"doc": "Guarda el estado COMPLETO de los objetos, no solo metadatos", "kind": "method", "line": 59, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"doc": "Carga el estado COMPLETO desde disco", "kind": "method", "line": 93, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"kind": "class", "line": 129, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Hoja KMS - INMUTABLE", "kind": "class", "line": 160, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso con estado serializable", "kind": "class", "line": 203, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 316, "name": "PhysicalValidator", "signature": "class PhysicalValidator"}, {"kind": "class", "line": 343, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 377, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "Pipeline con reanudación que realmente carga objetos", "kind": "method", "line": 554, "name": "simulate_resma_with_checkpointing", "signature": "def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)"}, {"kind": "method", "line": 36, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 41, "name": "check_memory_limit", "signature": "def check_memory_limit()"}, {"kind": "method", "line": 50, "name": "log_resources", "signature": "def log_resources()"}, {"kind": "method", "line": 146, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 168, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 172, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Distancia Bures con caché externo", "kind": "method", "line": 176, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Constructor que puede recibir estado serializado", "kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state)"}, {"kind": "method", "line": 258, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Matriz de medida", "kind": "method", "line": 269, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"kind": "method", "line": 301, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 318, "name": "validate_dimension", "signature": "def validate_dimension(alpha, tolerance)"}, {"kind": "method", "line": 326, "name": "validate_pt_symmetry", "signature": "def validate_pt_symmetry(kappa, Omega, chi)"}, {"kind": "method", "line": 334, "name": "validate_connectome_size", "signature": "def validate_connectome_size(n_nodes)"}, {"kind": "method", "line": 339, "name": "validate_spectral_dimension", "signature": "def validate_spectral_dimension(dim)"}, {"kind": "method", "line": 348, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 358, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 363, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 369, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 372, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Constructor que puede recibir grafo ya construido", "kind": "method", "line": 378, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)"}, {"doc": "Generar grafo por lotes", "kind": "method", "line": 446, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "Dimensión espectral con eigenvalores sparse", "kind": "method", "line": 475, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "Ramsey topológico", "kind": "method", "line": 499, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Números de Betti", "kind": "method", "line": 518, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz de distancias sparse", "kind": "method", "line": 534, "name": "_graph_to_distance_matrix", "signature": "def _graph_to_distance_matrix(self)"}]}, {"doc": "============================================================================= RESMA 4.3.3 – GARNIER INTEGRADO CON CORRECCIONES DIMENSIONALES =============================================================================", "id": "resma4.5.py", "kind": "module", "label": "resma4.5.py", "language": "py", "sha256": "cb1335f1eec350ca", "symbol_count": 57, "symbols": [{"kind": "class", "line": 34, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"kind": "method", "line": 59, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"kind": "method", "line": 88, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"doc": "Convierte objetos a formato serializable", "kind": "method", "line": 113, "name": "_make_serializable", "signature": "def _make_serializable(obj)"}, {"kind": "class", "line": 135, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Toro temporal T³ con parámetros ADIMENSIONALES.\nC0, C2, C3 son ratios de escala, no velocidades.", "kind": "class", "line": 163, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"doc": "D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E\nRepresentación toy de E8 (248x248)", "kind": "class", "line": 201, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"doc": "Monitor de Silencio-Activo: ΔS_loop < ε_c(ϕ)", "kind": "class", "line": 269, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"doc": "Hoja KMS - INMUTABLE (SIN CAMBIOS)", "kind": "class", "line": 337, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso con estado serializable y desdoblamiento Garnier", "kind": "class", "line": 380, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Cavidad PT-simétrica (SIN CAMBIOS)", "kind": "class", "line": 486, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Red neuronal con embedding Garnier", "kind": "class", "line": 517, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "Cálculos experimentales (SIN CAMBIOS)", "kind": "class", "line": 658, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline único con Garnier integrado", "kind": "method", "line": 695, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)"}, {"kind": "method", "line": 36, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 41, "name": "check_memory_limit", "signature": "def check_memory_limit()"}, {"kind": "method", "line": 50, "name": "log_resources", "signature": "def log_resources()"}, {"kind": "method", "line": 152, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 170, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico", "kind": "method", "line": 181, "name": "factor_escala", "signature": "def factor_escala(self, tiempo_idx)"}, {"doc": "Entropía crítica de percolación (ADIMENSIONAL).\nlog(2) es la entropía de un bit cuántico crítico.", "kind": "method", "line": 185, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"doc": "Para serialización", "kind": "method", "line": 192, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 197, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"doc": "Construye 3 generadores temporales (antis-Hermitianos)", "kind": "method", "line": 214, "name": "_construir_generadores_E8", "signature": "def _construir_generadores_E8(self)"}, {"doc": "Operador de Hadamard en dimensión 248 (unitario)", "kind": "method", "line": 226, "name": "_hadamard_generalizado", "signature": "def _hadamard_generalizado(self)"}, {"doc": "Construye D̂_G(ϕ) dimensionalmente consistente", "kind": "method", "line": 235, "name": "operator", "signature": "def operator(self)"}, {"doc": "Aplica desdoblamiento a un estado cuántico |Ψ⟩", "kind": "method", "line": 254, "name": "aplicar_a_estado", "signature": "def aplicar_a_estado(self, estado)"}, {"doc": "α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))\nGarantiza α' ∈ [0, α]", "kind": "method", "line": 260, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 273, "name": "__init__", "signature": "def __init__(self, garnier, network)"}, {"doc": "ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)\nrho_red: matriz densidad reducida (si es None, se calcula)", "kind": "method", "line": 278, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red)"}, {"doc": "Aproximación: ρ_red = diag(grados) / sum(grados)", "kind": "method", "line": 299, "name": "_calcular_rho_reducida_aproximada", "signature": "def _calcular_rho_reducida_aproximada(self)"}, {"doc": "Verifica Silencio-Activo y calcula Libertad L.\nRetorna: (condicion, libertad_L)", "kind": "method", "line": 307, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red)"}, {"doc": "Umbral de percolación para soberanía: 70% (Axioma 6)", "kind": "method", "line": 324, "name": "umbral_percolacion", "signature": "def umbral_percolacion(self)"}, {"kind": "method", "line": 345, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 349, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 353, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"doc": "Constructor que puede recibir estado serializado", "kind": "method", "line": 383, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"kind": "method", "line": 420, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Matriz de medida sin desdoblamiento", "kind": "method", "line": 431, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Aplica D̂_G(ϕ) a la medida:\n- M_ij → M_ij * (C0/C3)^(cos(ϕ₃))\n- Normaliza después", "kind": "method", "line": 451, "name": "_aplicar_desdoblamiento_a_medida", "signature": "def _aplicar_desdoblamiento_a_medida(self, measure)"}, {"kind": "method", "line": 471, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"doc": "Libertad del universo: L = 1/ε_c", "kind": "method", "line": 481, "name": "_calcular_libertad_universo", "signature": "def _calcular_libertad_universo(self)"}, {"kind": "method", "line": 488, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"kind": "method", "line": 499, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 504, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 510, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 513, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"doc": "Constructor que puede recibir grafo ya construido", "kind": "method", "line": 520, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"doc": "Generar grafo por lotes con conectividad controlada", "kind": "method", "line": 559, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "Dimensión espectral con eigenvalores sparse", "kind": "method", "line": 591, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "Ramsey topológico simplificado", "kind": "method", "line": 615, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Números de Betti aproximados por ciclos locales", "kind": "method", "line": 627, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz densidad reducida del conectoma", "kind": "method", "line": 636, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"doc": "Verifica: conectividad > 70% para soberanía", "kind": "method", "line": 644, "name": "validar_axioma_6", "signature": "def validar_axioma_6(self)"}, {"kind": "method", "line": 661, "name": "__init__", "signature": "def __init__(self, universe, myelin, network)"}, {"doc": "Calcula Factor de Bayes integrando Garnier", "kind": "method", "line": 666, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.5 – ZPE-SILENCIO ANTAGONISMO + CONECTOMA HUMANO =============================================================================", "id": "resma4.6.py", "kind": "module", "label": "resma4.6.py", "language": "py", "sha256": "eee9edde3d60288b", "symbol_count": 59, "symbols": [{"kind": "class", "line": 31, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"doc": "Guarda estado completo con manejo robusto de errores", "kind": "method", "line": 56, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"doc": "Carga checkpoint con fallback automático", "kind": "method", "line": 86, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"doc": "Convierte objetos recursivamente a formato serializable", "kind": "method", "line": 112, "name": "_make_serializable", "signature": "def _make_serializable(obj)"}, {"doc": "Constantes físicas fundamentales", "kind": "class", "line": 139, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "**TORO TEMPORAL T³ CON CANCELACIÓN ZPE**\n- phi: Fase de desdoblamiento que controla anulación ZPE\n- zpe_level: Nivel de fluctuaciones de punto cero [0,1]", "kind": "class", "line": 169, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"doc": "**D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E**\n**CONTRA-ZPE**: Opera en subespacio sin fluctuaciones", "kind": "class", "line": 233, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"doc": "**MONITOR DE ANTAGONISMO ZPE-SILENCIO**\n- Detecta cuando fluctuaciones cuánticas son coherentemente anuladas\n- Mide nivel de \"ruido de fondo cuántico\" vs \"silencio ontológico\"", "kind": "class", "line": 312, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"doc": "Hoja KMS - INMUTABLE", "kind": "class", "line": 441, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso con ZPE-Silencio integrado", "kind": "class", "line": 484, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Cavidad PT-simétrica con medición ZPE", "kind": "class", "line": 608, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"doc": "Red neuronal con validación ZPE-Silencio", "kind": "class", "line": 659, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "Cálculos experimentales unificados ZPE-Silencio", "kind": "class", "line": 853, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio", "kind": "method", "line": 906, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)"}, {"kind": "method", "line": 33, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 38, "name": "check_memory_limit", "signature": "def check_memory_limit(threshold)"}, {"kind": "method", "line": 47, "name": "log_resources", "signature": "def log_resources()"}, {"kind": "method", "line": 158, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 181, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Factor de escala con supresión ZPE", "kind": "method", "line": 194, "name": "factor_escala", "signature": "def factor_escala(self, tiempo_idx)"}, {"doc": "**UMBRAL CRÍTICO CON ZPE**:\nCuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)", "kind": "method", "line": 200, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"doc": "Serialización completa", "kind": "method", "line": 208, "name": "to_dict", "signature": "def to_dict(self)"}, {"doc": "Deserialización", "kind": "method", "line": 220, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 238, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"doc": "GENERADORES CON CANCELACIÓN ZPE INTEGRADA", "kind": "method", "line": 246, "name": "_construir_generadores_E8_ZPE", "signature": "def _construir_generadores_E8_ZPE(self)"}, {"doc": "HADAMARD CON ESPACIO NULO ZPE", "kind": "method", "line": 266, "name": "_hadamard_generalizado_ZPE", "signature": "def _hadamard_generalizado_ZPE(self)"}, {"doc": "Construye D̂_G(ϕ) con cancelación ZPE", "kind": "method", "line": 284, "name": "operator", "signature": "def operator(self)"}, {"doc": "**α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))**", "kind": "method", "line": 304, "name": "alpha_modificado", "signature": "def alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 318, "name": "__init__", "signature": "def __init__(self, garnier, network)"}, {"doc": "**ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**\n**NUEVO**: La entropía ZPE se SUMA a la entropía total", "kind": "method", "line": 327, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red)"}, {"doc": "Matriz densidad con modulación ZPE", "kind": "method", "line": 367, "name": "_calcular_rho_reducida_aproximada", "signature": "def _calcular_rho_reducida_aproximada(self)"}, {"doc": "**DETECCIÓN DE ANTAGONISMO**:\nRetorna: (condicion, libertad_L, nivel_ZPE_cancelado)\n\n**CONDICIÓN**: ZPE < 1% AND ΔS_loop < ε_c", "kind": "method", "line": 383, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red)"}, {"doc": "Umbral para soberanía: 70%", "kind": "method", "line": 411, "name": "umbral_percolacion", "signature": "def umbral_percolacion(self)"}, {"doc": "**MODO GOLDSTONE DEL DOBLE CUÁNTICO**:\nExcitación colectiva que anuncia ruptura de simetría ZPE", "kind": "method", "line": 415, "name": "modo_goldstone", "signature": "def modo_goldstone(self)"}, {"kind": "method", "line": 449, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 453, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 457, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 487, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"doc": "Inicializa hojas con temperatura efectiva afectada por ZPE", "kind": "method", "line": 521, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Genera medida de Gibbs", "kind": "method", "line": 535, "name": "_generate_gibbs_measure", "signature": "def _generate_gibbs_measure(self)"}, {"doc": "Aplica desdoblamiento con supresión ZPE", "kind": "method", "line": 557, "name": "_aplicar_desdoblamiento_a_medida", "signature": "def _aplicar_desdoblamiento_a_medida(self, measure)"}, {"doc": "Construye estado global normalizado", "kind": "method", "line": 584, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"doc": "Libertad intrínseca con supresión ZPE", "kind": "method", "line": 603, "name": "_calcular_libertad_universo", "signature": "def _calcular_libertad_universo(self)"}, {"kind": "method", "line": 611, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"doc": "Hamiltoniano con energía ZPE incluida", "kind": "method", "line": 624, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"doc": "Potencial de pérdida PT", "kind": "method", "line": 633, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 640, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"doc": "**ENERGÍA DE PUNTO CERO TOTAL**:\nE_ZPE = Σ_i ½ħω_i", "kind": "method", "line": 643, "name": "_calcular_zpe", "signature": "def _calcular_zpe(self)"}, {"kind": "method", "line": 655, "name": "_pt_symmetry_condition", "signature": "def _pt_symmetry_condition(self)"}, {"kind": "method", "line": 662, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"doc": "Genera grafo con densidad 0.75 (conectoma humano)", "kind": "method", "line": 711, "name": "_generate_fractal_graph", "signature": "def _generate_fractal_graph(self)"}, {"doc": "Dimensión espectral con ZPE", "kind": "method", "line": 753, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "Ramsey topológico", "kind": "method", "line": 777, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Números de Betti reales", "kind": "method", "line": 788, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Matriz densidad con supresión ZPE", "kind": "method", "line": 804, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"doc": "**ENERGÍA ZPE DEL CONECTOMA**:\nE_ZPE = Σ_i ½ħω_i (modos de Laplaciano)", "kind": "method", "line": 820, "name": "_calcular_zpe_conectoma", "signature": "def _calcular_zpe_conectoma(self)"}, {"doc": "**AXIOMA 6**: Conectividad > 70% para soberanía", "kind": "method", "line": 836, "name": "validar_axioma_6", "signature": "def validar_axioma_6(self)"}, {"kind": "method", "line": 856, "name": "__init__", "signature": "def __init__(self, universe, myelin, network)"}, {"doc": "Calcula Factor de Bayes con antagonismo ZPE-Silencio", "kind": "method", "line": 861, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.4 – CORRECCIONES CRÍTICAS GARNIER-MALET =============================================================================", "id": "resma4.7.py", "kind": "module", "label": "resma4.7.py", "language": "py", "sha256": "6a8586c4bf54e9f9", "symbol_count": 39, "symbols": [{"kind": "class", "line": 23, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"doc": "Toro temporal T³ con parámetros físicamente consistentes.\nBasado en la teoría del desdoblamiento del tiempo de Garnier-Malet.", "kind": "class", "line": 46, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"doc": "Operador de desdoblamiento D̂_G(φ) con estructura E8 simplificada", "kind": "class", "line": 94, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"doc": "Monitor de condición de Silencio-Activo: ΔS_loop < ε_c(φ)", "kind": "class", "line": 139, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"kind": "class", "line": 190, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"doc": "Multiverso cuántico con desdoblamiento Garnier-Malet", "kind": "class", "line": 223, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"doc": "Red neuronal con topología realista que satisface Axioma 6", "kind": "class", "line": 336, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"doc": "Cálculo de Factor de Bayes y predicciones", "kind": "class", "line": 484, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"doc": "Pipeline completo RESMA-Garnier con correcciones", "kind": "method", "line": 537, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed)"}, {"kind": "method", "line": 53, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"doc": "Fuerza de acoplamiento entre tiempos", "kind": "method", "line": 67, "name": "_compute_coupling", "signature": "def _compute_coupling(self)"}, {"doc": "Factor de escala temporal", "kind": "method", "line": 72, "name": "factor_escala", "signature": "def factor_escala(self, tiempo_idx)"}, {"doc": "Entropía crítica con corrección de acoplamiento:\nε_c = log(2) · (C0/C3)² · (1 + ξ)", "kind": "method", "line": 77, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"doc": "Factor de modulación para la medida cuántica:\nM = exp(-|φ₃ - π|/C3)\nMáximo cuando φ₃ ≈ π (apertura temporal óptima)", "kind": "method", "line": 85, "name": "modulation_factor", "signature": "def modulation_factor(self)"}, {"kind": "method", "line": 98, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"doc": "Generadores temporales (anti-Hermitianos normalizados)", "kind": "method", "line": 103, "name": "_construir_generadores", "signature": "def _construir_generadores(self)"}, {"doc": "Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)", "kind": "method", "line": 115, "name": "operator", "signature": "def operator(self)"}, {"doc": "Aplica desdoblamiento a vector de estado", "kind": "method", "line": 120, "name": "aplicar_modulacion", "signature": "def aplicar_modulacion(self, state_vector)"}, {"doc": "α'(φ) = α · |cos(φ₃)|^(C0/C3)\nGarantiza α' ∈ [0, α]", "kind": "method", "line": 126, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 143, "name": "__init__", "signature": "def __init__(self, garnier)"}, {"doc": "ΔS_loop = S_vN(ρ) - log(b₁ + 1)\n\nArgs:\n    rho_red: Matriz densidad reducida\n    b1: Primer número de Betti (ciclos independientes)", "kind": "method", "line": 147, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red, b1)"}, {"doc": "Verifica condición y calcula libertad L = 1/(ΔS + ε_c)\n\nReturns:\n    (condicion_satisfecha, libertad)", "kind": "method", "line": 166, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red, b1)"}, {"kind": "method", "line": 197, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"doc": "Distancia de Bures simplificada", "kind": "method", "line": 203, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 226, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, garnier)"}, {"doc": "Genera hojas con gaps distribuidos exponencialmente", "kind": "method", "line": 252, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"doc": "Genera medida de transición modulada por Garnier:\nM_ij = exp(-β d²_ij) · φ(garnier)", "kind": "method", "line": 265, "name": "_generate_modulated_measure", "signature": "def _generate_modulated_measure(self)"}, {"doc": "Estado global como distribución diagonal", "kind": "method", "line": 312, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"doc": "Libertad del universo: L_U = 1/ε_c", "kind": "method", "line": 322, "name": "_calcular_libertad", "signature": "def _calcular_libertad(self)"}, {"doc": "Coherencia cuántica: suma de elementos off-diagonal", "kind": "method", "line": 326, "name": "_calcular_coherencia", "signature": "def _calcular_coherencia(self)"}, {"kind": "method", "line": 341, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, garnier)"}, {"doc": "Genera red con conectividad > 70% usando modelo realista:\n- Watts-Strogatz para mundo pequeño\n- Aumentación para alcanzar umbral", "kind": "method", "line": 379, "name": "_generate_realistic_network", "signature": "def _generate_realistic_network(self)"}, {"doc": "Números de Betti: b0=componentes, b1=ciclos", "kind": "method", "line": 424, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"doc": "Dimensión espectral del Laplaciano", "kind": "method", "line": 433, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"doc": "Número de Ramsey topológico", "kind": "method", "line": 455, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"doc": "Matriz densidad de la red (normalizada por grados)", "kind": "method", "line": 460, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"doc": "Verifica conectividad > 70%", "kind": "method", "line": 470, "name": "_validar_axioma_6", "signature": "def _validar_axioma_6(self)"}, {"kind": "method", "line": 487, "name": "__init__", "signature": "def __init__(self, universe, network)"}, {"doc": "ln(BF) ∝ log(L_red · L_univ)\nVeredicto basado en libertad total", "kind": "method", "line": 491, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (Validada y lista para ejecución) ============================================================================= Código completo y ejecutable sin interrupciones de markdown Guardar como: resma_4_3_6.py", "id": "resma4.8.py", "kind": "module", "label": "resma4.8.py", "language": "py", "sha256": "73e8a15ebee0002e", "symbol_count": 55, "symbols": [{"kind": "class", "line": 37, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"kind": "class", "line": 70, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"kind": "method", "line": 91, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"kind": "method", "line": 117, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"kind": "method", "line": 141, "name": "_make_serializable", "signature": "def _make_serializable(obj)"}, {"kind": "class", "line": 158, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"kind": "class", "line": 209, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"kind": "class", "line": 256, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"kind": "class", "line": 284, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 331, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 433, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"kind": "class", "line": 584, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 620, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"kind": "method", "line": 667, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)"}, {"kind": "method", "line": 54, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 72, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 77, "name": "check_memory_limit", "signature": "def check_memory_limit()"}, {"kind": "method", "line": 86, "name": "log_resources", "signature": "def log_resources()"}, {"kind": "method", "line": 161, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 179, "name": "_compute_coupling", "signature": "def _compute_coupling(self)"}, {"kind": "method", "line": 182, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"kind": "method", "line": 186, "name": "modulation_factor", "signature": "def modulation_factor(self)"}, {"kind": "method", "line": 189, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 199, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 210, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"kind": "method", "line": 219, "name": "_construir_generadores_aleatorios", "signature": "def _construir_generadores_aleatorios(self)"}, {"kind": "method", "line": 228, "name": "_hadamard_generalizado", "signature": "def _hadamard_generalizado(self)"}, {"kind": "method", "line": 233, "name": "operator", "signature": "def operator(self)"}, {"kind": "method", "line": 248, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 257, "name": "__init__", "signature": "def __init__(self, garnier)"}, {"kind": "method", "line": 261, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red, b1)"}, {"kind": "method", "line": 268, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red, b1)"}, {"kind": "method", "line": 291, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 295, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 301, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 332, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"kind": "method", "line": 362, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"kind": "method", "line": 373, "name": "_generate_complete_measure", "signature": "def _generate_complete_measure(self)"}, {"kind": "method", "line": 402, "name": "_aplicar_modulacion_garnier", "signature": "def _aplicar_modulacion_garnier(self, measure)"}, {"kind": "method", "line": 413, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 422, "name": "_calcular_libertad", "signature": "def _calcular_libertad(self)"}, {"kind": "method", "line": 425, "name": "_calcular_coherencia", "signature": "def _calcular_coherencia(self)"}, {"kind": "method", "line": 434, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"kind": "method", "line": 472, "name": "_generate_realistic_modular_network", "signature": "def _generate_realistic_modular_network(self)"}, {"kind": "method", "line": 529, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"kind": "method", "line": 537, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"kind": "method", "line": 559, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"kind": "method", "line": 563, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"kind": "method", "line": 571, "name": "_validar_axioma_6", "signature": "def _validar_axioma_6(self)"}, {"kind": "method", "line": 585, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"kind": "method", "line": 602, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 607, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 613, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 621, "name": "__init__", "signature": "def __init__(self, universe, network, myelin)"}, {"kind": "method", "line": 626, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}]}, {"doc": "============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (CÓDIGO DE PRODUCCIÓN COMPLETO) ============================================================================= GUARDAR COMO: resma4.9_fixed.py COMPATIBLE CON: resma_checkpoint_v4_6.pkl FIX CRÍTICO: Conversión automática de listas a numpy arrays al cargar checkpoint", "id": "resma4.9.py", "kind": "module", "label": "resma4.9.py", "language": "py", "sha256": "b2041c933c1f3b1f", "symbol_count": 53, "symbols": [{"kind": "class", "line": 34, "name": "RESMAConstants", "signature": "class RESMAConstants"}, {"kind": "class", "line": 71, "name": "GarnierTresTiempos", "signature": "class GarnierTresTiempos"}, {"kind": "class", "line": 111, "name": "OperadorDesdoblamiento", "signature": "class OperadorDesdoblamiento"}, {"kind": "class", "line": 158, "name": "SilencioActivoMonitor", "signature": "class SilencioActivoMonitor"}, {"kind": "class", "line": 186, "name": "QuantumLeaf", "signature": "class QuantumLeaf"}, {"kind": "class", "line": 233, "name": "RESMAUniverse", "signature": "class RESMAUniverse"}, {"kind": "class", "line": 341, "name": "NeuralNetworkRESMA", "signature": "class NeuralNetworkRESMA"}, {"kind": "class", "line": 492, "name": "MyelinCavity", "signature": "class MyelinCavity"}, {"kind": "class", "line": 528, "name": "ExperimentalPredictions", "signature": "class ExperimentalPredictions"}, {"kind": "class", "line": 575, "name": "ResourceMonitor", "signature": "class ResourceMonitor"}, {"kind": "method", "line": 587, "name": "guardar_checkpoint", "signature": "def guardar_checkpoint(data, filename)"}, {"kind": "method", "line": 615, "name": "cargar_checkpoint", "signature": "def cargar_checkpoint(filename)"}, {"kind": "method", "line": 639, "name": "_make_serializable", "signature": "def _make_serializable(obj)"}, {"kind": "method", "line": 655, "name": "simulate_resma_garnier", "signature": "def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)"}, {"kind": "method", "line": 50, "name": "verify_pt_condition", "signature": "def verify_pt_condition(cls)"}, {"kind": "method", "line": 74, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 86, "name": "epsilon_critico", "signature": "def epsilon_critico(self)"}, {"kind": "method", "line": 89, "name": "modulation_factor", "signature": "def modulation_factor(self)"}, {"kind": "method", "line": 92, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 102, "name": "from_dict", "signature": "def from_dict(cls, data)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, garnier, dimension)"}, {"kind": "method", "line": 121, "name": "_construir_generadores_aleatorios", "signature": "def _construir_generadores_aleatorios(self)"}, {"kind": "method", "line": 130, "name": "_hadamard_generalizado", "signature": "def _hadamard_generalizado(self)"}, {"kind": "method", "line": 135, "name": "operator", "signature": "def operator(self)"}, {"kind": "method", "line": 150, "name": "calcular_alpha_modificado", "signature": "def calcular_alpha_modificado(self, alpha_base)"}, {"kind": "method", "line": 159, "name": "__init__", "signature": "def __init__(self, garnier)"}, {"kind": "method", "line": 163, "name": "calcular_delta_s_loop", "signature": "def calcular_delta_s_loop(self, rho_red, b1)"}, {"kind": "method", "line": 170, "name": "es_silencio_activo", "signature": "def es_silencio_activo(self, rho_red, b1)"}, {"kind": "method", "line": 193, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 197, "name": "spectral_density", "signature": "def spectral_density(self, omega)"}, {"kind": "method", "line": 203, "name": "bures_distance", "signature": "def bures_distance(self, other)"}, {"kind": "method", "line": 234, "name": "__init__", "signature": "def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)"}, {"kind": "method", "line": 270, "name": "_initialize_leaves", "signature": "def _initialize_leaves(self)"}, {"kind": "method", "line": 281, "name": "_generate_complete_measure", "signature": "def _generate_complete_measure(self)"}, {"kind": "method", "line": 310, "name": "_aplicar_modulacion_garnier", "signature": "def _aplicar_modulacion_garnier(self, measure)"}, {"kind": "method", "line": 321, "name": "_construct_global_state", "signature": "def _construct_global_state(self)"}, {"kind": "method", "line": 330, "name": "_calcular_libertad", "signature": "def _calcular_libertad(self)"}, {"kind": "method", "line": 333, "name": "_calcular_coherencia", "signature": "def _calcular_coherencia(self)"}, {"kind": "method", "line": 342, "name": "__init__", "signature": "def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)"}, {"kind": "method", "line": 380, "name": "_generate_realistic_modular_network", "signature": "def _generate_realistic_modular_network(self)"}, {"kind": "method", "line": 437, "name": "_compute_betti_numbers", "signature": "def _compute_betti_numbers(self)"}, {"kind": "method", "line": 445, "name": "_spectral_dimension", "signature": "def _spectral_dimension(self)"}, {"kind": "method", "line": 467, "name": "_topological_ramsey", "signature": "def _topological_ramsey(self)"}, {"kind": "method", "line": 471, "name": "_calcular_rho_reducida", "signature": "def _calcular_rho_reducida(self)"}, {"kind": "method", "line": 479, "name": "_validar_axioma_6", "signature": "def _validar_axioma_6(self)"}, {"kind": "method", "line": 493, "name": "__init__", "signature": "def __init__(self, axon_length, radius, n_modes)"}, {"kind": "method", "line": 510, "name": "_free_hamiltonian", "signature": "def _free_hamiltonian(self)"}, {"kind": "method", "line": 515, "name": "_loss_potential", "signature": "def _loss_potential(self)"}, {"kind": "method", "line": 521, "name": "_compute_scalar_mass", "signature": "def _compute_scalar_mass(self)"}, {"kind": "method", "line": 529, "name": "__init__", "signature": "def __init__(self, universe, network, myelin)"}, {"kind": "method", "line": 534, "name": "compute_log_bayes_factor", "signature": "def compute_log_bayes_factor(self)"}, {"kind": "method", "line": 577, "name": "get_memory_gb", "signature": "def get_memory_gb()"}, {"kind": "method", "line": 582, "name": "log_resources", "signature": "def log_resources()"}]}, {"id": "sovereignty_monitor.py", "kind": "module", "label": "sovereignty_monitor.py", "language": "py", "sha256": "c60f56308436a20f", "symbol_count": 18, "symbols": [{"doc": "Setup matplotlib para visualización", "kind": "function", "line": 21, "name": "setup_matplotlib_for_plotting", "signature": "def setup_matplotlib_for_plotting()"}, {"doc": "Implementación del Sovereignty Monitor basada en RESMA\n\nCalcula L = 1 / (|S_vN(ρ) − log(rank(W) + 1)| + ε_c)", "kind": "class", "line": 29, "name": "SovereigntyMonitor", "signature": "class SovereigntyMonitor"}, {"doc": "CNN para MNIST con arquitectura diseñada para monitoreo", "kind": "class", "line": 94, "name": "CNNMNIST", "signature": "class CNNMNIST(Module)"}, {"doc": "Carga y prepara el dataset MNIST", "kind": "method", "line": 126, "name": "cargar_datos", "signature": "def cargar_datos()"}, {"doc": "Experimento completo para validar el Sovereignty Monitor", "kind": "class", "line": 146, "name": "ExperimentoCompleto", "signature": "class ExperimentoCompleto"}, {"doc": "Función principal", "kind": "method", "line": 433, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 35, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"doc": "Calcula la métrica L (libertad) de una matriz de pesos\n\nReturns:\n    tuple: (L, S_vn, rank_effective)", "kind": "method", "line": 38, "name": "calcular_libertad", "signature": "def calcular_libertad(self, weights)"}, {"doc": "Evalúa el régimen del modelo", "kind": "method", "line": 85, "name": "evaluar_regimen", "signature": "def evaluar_regimen(self, L)"}, {"kind": "method", "line": 96, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 110, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Retorna todas las capas lineales para monitoreo", "kind": "method", "line": 122, "name": "get_linear_layers", "signature": "def get_linear_layers(self)"}, {"kind": "method", "line": 149, "name": "__init__", "signature": "def __init__(self, num_epochs)"}, {"doc": "Calcula métricas L para todas las capas lineales", "kind": "method", "line": 189, "name": "calcular_metricas_sovereignty", "signature": "def calcular_metricas_sovereignty(self)"}, {"doc": "Entrena una época completa", "kind": "method", "line": 210, "name": "entrenar_epoca", "signature": "def entrenar_epoca(self, epoca)"}, {"doc": "Evalúa el modelo en el conjunto de validación", "kind": "method", "line": 233, "name": "evaluar_epoca", "signature": "def evaluar_epoca(self)"}, {"doc": "Ejecuta el experimento completo", "kind": "method", "line": 253, "name": "ejecutar_experimento", "signature": "def ejecutar_experimento(self)"}, {"doc": "Genera gráficos comprehensivos de resultados", "kind": "method", "line": 362, "name": "generar_graficos", "signature": "def generar_graficos(self)"}]}, {"id": "test_simple.py", "kind": "module", "label": "test_simple.py", "language": "py", "sha256": "c51379778a86aaf9", "symbol_count": 1, "symbols": [{"doc": "Test de las matemáticas básicas RESMA", "kind": "function", "line": 8, "name": "test_basic_math", "signature": "def test_basic_math()"}]}, {"id": "test_ultra_simple.py", "kind": "module", "label": "test_ultra_simple.py", "language": "py", "sha256": "a143091ab21a72da", "symbol_count": 0, "symbols": []}, {"id": "train_mini_resma.py", "kind": "module", "label": "train_mini_resma.py", "language": "py", "sha256": "0b6b1f6e8b008d56", "symbol_count": 1, "symbols": [{"kind": "function", "line": 8, "name": "main", "signature": "def main()"}]}, {"id": "train_profile.py", "kind": "module", "label": "train_profile.py", "language": "py", "sha256": "dbf7af6e0d2aee5f", "symbol_count": 1, "symbols": [{"kind": "function", "line": 9, "name": "main", "signature": "def main()"}]}, {"id": "visualize_resma.py", "kind": "module", "label": "visualize_resma.py", "language": "py", "sha256": "bea9561160840de2", "symbol_count": 2, "symbols": [{"doc": "Setup matplotlib and seaborn for plotting with proper configuration.\nCall this function before creating any plots to ensure proper rendering.", "kind": "function", "line": 6, "name": "setup_matplotlib_for_plotting", "signature": "def setup_matplotlib_for_plotting()"}, {"doc": "Cargar y visualizar estado de red entrenada", "kind": "function", "line": 30, "name": "diagnosticar_modelo", "signature": "def diagnosticar_modelo(checkpoint_path)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (41 files)

#### `app.py`
**Path:** `app.py`
**File Doc:** *_*_ coding: utf8 _*_*

*No symbols extracted*

#### `demo_mini_resma.py`
**Path:** `demo_mini_resma.py`

**Classes:**
- `GarnierLayer` (line 8) `class GarnierLayer(Module)` - *Capa neuronal con temporalidad Garnier T³ (simplificada para demo)*

**Methods:**
- `demo_resma` (line 49) `def demo_resma()` - *Demostración rápida de la arquitectura RESMA-Garnier*
- `__init__` (line 10) `def __init__(self, in_features, out_features, device)`
- `forward` (line 27) `def forward(self, x)` - *Forward simplificado para demostración*

#### `difract.py`
**Path:** `difract.py`

**Functions:**
- `visualize_uased_geometry` (line 4) `def visualize_uased_geometry()`

#### `garnier_nn.py`
**Path:** `garnier_nn.py`

**Classes:**
- `GarnierLayer` (line 9) `class GarnierLayer(Module)` - *Capa neuronal con temporalidad Garnier T³*
- `SilencioActivoNetwork` (line 67) `class SilencioActivoNetwork(Module)` - *Red neuronal completa con arquitectura RESMA-Garnier*

**Methods:**
- `__init__` (line 11) `def __init__(self, in_features, out_features, device)`
- `forward` (line 33) `def forward(self, x)` - *Forward con no-linealidad Garnier
Returns: (output, delta_s_loop)*
- `__init__` (line 69) `def __init__(self, layer_sizes, scale, device)`
- `_build_garnier_topology` (line 111) `def _build_garnier_topology(self)` - *Construcción BA+WS modular miniaturizada*
- `forward` (line 130) `def forward(self, x)` - *Forward completo con tracking de métricas de consciencia
Returns: (logits, metrics)*
- `activar_perfilado` (line 168) `def activar_perfilado(self)` - *Activar perfilado de tiempo en toda la red*
- `mostrar_estadisticas_perfilado` (line 183) `def mostrar_estadisticas_perfilado(self)` - *Mostrar estadísticas de perfilado*
- `entrenar_con_perfilado` (line 201) `def entrenar_con_perfilado(self, train_loader, epochs, lr)` - *Entrenamiento con perfilado detallado*
- `entrenar` (line 240) `def entrenar(self, train_loader, epochs, lr)` - *Entrenamiento incorporado con regularización Garnier*

#### `main.py`
**Path:** `main.py`
**File Doc:** *============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================*

**Classes:**
- `RESMAConstants` (line 32) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 58) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 90) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 220) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 270) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 350) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 428) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 481) `class NeuralNetworkRESMA` - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 582) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 628) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 686) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Methods:**
- `simulate_resma_multiverse` (line 764) `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 62) `def validate_dimension(alpha)` - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 68) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 79) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico*
- `__post_init__` (line 100) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121) `def bures_distance(self, other)`
- `_spectral_moments` (line 132) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 168) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 181) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 203) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 226) `def __init__(self, leaf, threshold)`
- `_construct_cptp_map` (line 231) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 240) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 255) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 276) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 282) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 286) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 294) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 310) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 356) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 362) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 372) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 382) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 389) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 437) `def __post_init__(self)`
- `_free_hamiltonian` (line 442) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 448) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 456) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 462) `def coherence_quantum(self)` - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 487) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 500) `def _generate_fractal_graph(self)` - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 510) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 527) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 548) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 562) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 574) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 588) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 592) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 596) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 607) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 618) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 635) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 654) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 670) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 692) `def __init__(self, resma, myelin, network)`
- `predict_all` (line 699) `def predict_all(self)` - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 710) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 715) `def compute_bayes_factor(self)` - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main2.py`
**Path:** `main2.py`
**File Doc:** *============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================*

**Classes:**
- `RESMAConstants` (line 33) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA*
- `PhysicalValidator` (line 59) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes*
- `QuantumLeaf` (line 91) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field.
No almacena matrices densas (Pilar 4).*
- `RESMAUniverse` (line 144) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas.
Memoria: O(N_leaves) en lugar de O(N_leaves × dim²)*
- `BranchingOperator` (line 219) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local sin matrices globales (Pilar 4).*
- `EmunaOperator` (line 269) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo (Pilar 4).*
- `LindbladFractalDynamics` (line 349) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ] + ξ(t)
Integración por Euler-Maruyama (Pilar 4: estabilidad numérica).*
- `MyelinCavity` (line 427) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field para Colab (Pilar 4).*
- `NeuralNetworkRESMA` (line 480) `class NeuralNetworkRESMA` - *Conectoma humano dirigido con homología persistente.
Implementación sparse para escalado (Pilar 4).*
- `FreedomInvariant` (line 581) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 627) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados RESMA.*
- `ExperimentalPredictions` (line 685) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC (aproximación).*

**Methods:**
- `simulate_resma_multiverse` (line 763) `def simulate_resma_multiverse(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA 3.0 con verificaciones de integridad.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 63) `def validate_dimension(alpha)` - *α ∈ (0,1) por definición de dimensión fractal*
- `validate_pt_symmetry` (line 69) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría*
- `validate_connectome_size` (line 80) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico*
- `__post_init__` (line 101) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 106) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁.
Evidencia: SYK tiene espectro continuo sin gaps (Maldacena, JHEP 2016).*
- `modular_entropy` (line 114) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω (aproximada numéricamente)*
- `bures_distance` (line 121) `def bures_distance(self, other)`
- `_spectral_moments` (line 132) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n*
- `__init__` (line 150) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 167) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos*
- `_generate_gibbs_measure` (line 180) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j))*
- `_construct_global_state` (line 202) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz)*
- `__init__` (line 225) `def __init__(self, leaf, threshold)`
- `_construct_cptp_map` (line 230) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 239) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap.*
- `apply_branching` (line 254) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2)*
- `__init__` (line 275) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 281) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 285) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 293) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 309) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `__init__` (line 355) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 361) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva)*
- `_modular_dissipator` (line 371) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ}*
- `_nonlinear_term` (line 381) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞]*
- `evolve` (line 388) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama.
Returns: trayectoria [n_steps, 2, 2]*
- `__post_init__` (line 436) `def __post_init__(self)`
- `_free_hamiltonian` (line 441) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 447) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.7*
- `_pt_symmetry_condition` (line 455) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1*
- `coherence_quantum` (line 461) `def coherence_quantum(self)` - *Discordia cuántica aproximada (ejemplo: estado separable → 0)*
- `__init__` (line 486) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 499) `def _generate_fractal_graph(self)` - *Grafo dirigido con distribución de grados power-law.
Fuente: Human Connectome Project (Pilar 1).*
- `_spectral_dimension` (line 509) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ (sparse eigenvalue solver)*
- `_topological_ramsey` (line 526) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_graph_to_distance_matrix` (line 547) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR)*
- `critical_percolation_time` (line 561) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 573) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `__init__` (line 587) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 591) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional*
- `compute_pontryagin_number` (line 595) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 606) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G]*
- `is_gauge_invariant` (line 617) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico*
- `ising_quantum` (line 634) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈.*
- `syk4` (line 653) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7)).
Predice α sin postulado E₈.*
- `random_network` (line 669) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica.*
- `__init__` (line 691) `def __init__(self, resma, myelin, network)`
- `predict_all` (line 698) `def predict_all(self)` - *Predicciones RESMA 3.0*
- `_predict_diffraction_peak` (line 708) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (sin ajuste)*
- `compute_bayes_factor` (line 713) `def compute_bayes_factor(self)` - *BF = exp(ΔAIC/2) donde AIC = 2k - 2ln(L)
k = número de parámetros RESMA = 5 (α, β, γ, L_E8, g_coupling)*

#### `main3.py`
**Path:** `main3.py`
**File Doc:** *============================================================================= RESMA 4.0 – CÓDIGO COMPLETO CORREGIDO Autor: Tu nombre Fecha: 2025-11-21 Descripción: Implementación completa sin simplificaciones críticas =============================================================================*

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 68) `class QuantumLeaf`
- `Universe` (line 101) `class Universe`
- `Network` (line 129) `class Network`
- `MyelinCavity` (line 171) `class MyelinCavity`
- `Bayes` (line 205) `class Bayes`

**Methods:**
- `simulate` (line 233) `def simulate(n_leaves, n_nodes, seed)`
- `dim` (line 52) `def dim(a)`
- `pt` (line 56) `def pt(k, o, c)`
- `size` (line 59) `def size(n)`
- `__post_init__` (line 74) `def __post_init__(self)`
- `spectral_density` (line 78) `def spectral_density(self, w)`
- `modular_entropy` (line 81) `def modular_entropy(self)`
- `bures_distance` (line 87) `def bures_distance(self, other)`
- `__init__` (line 102) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 110) `def _gibbs(self)`
- `_global` (line 119) `def _global(self)`
- `__init__` (line 130) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 139) `def _spectral_dim(self, k)`
- `_ramsey` (line 149) `def _ramsey(self)`
- `t_c` (line 163) `def t_c(self)`
- `__init__` (line 172) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 178) `def _free_hamiltonian(self)`
- `_loss_potential` (line 183) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 189) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 192) `def coherence_quantum(self)`
- `__init__` (line 206) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 210) `def log_lik(self, model_pred)`
- `bf` (line 217) `def bf(self)`

#### `main4.1.py`
**Path:** `main4.1.py`
**File Doc:** *============================================================================= RESMA 4.1 – VERSIÓN CORREGIDA Y VALIDADA Correcciones críticas: 1. Escala correcta para q0 2. Factor de Bayes con regularización 3. Condiciones PT-simétricas ajustadas 4. Manejo robusto de errores numéricos =============================================================================*

**Classes:**
- `RC` (line 29) `class RC`
- `Validator` (line 61) `class Validator`
- `QuantumLeaf` (line 82) `class QuantumLeaf`
- `Universe` (line 123) `class Universe`
- `Network` (line 152) `class Network`
- `MyelinCavity` (line 229) `class MyelinCavity`
- `Bayes` (line 268) `class Bayes`

**Methods:**
- `simulate` (line 307) `def simulate(n_leaves, n_nodes, seed)`
- `verify_pt_condition` (line 50) `def verify_pt_condition(cls)` - *Verifica que kappa < chi*Omega para simetría PT*
- `dim` (line 63) `def dim(a)`
- `pt` (line 68) `def pt(k, o, c)` - *Condición PT: kappa < chi*Omega*
- `size` (line 73) `def size(n)`
- `__post_init__` (line 88) `def __post_init__(self)`
- `spectral_density` (line 92) `def spectral_density(self, w)`
- `modular_entropy` (line 95) `def modular_entropy(self)`
- `bures_distance` (line 104) `def bures_distance(self, other)`
- `__init__` (line 124) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 133) `def _gibbs(self)`
- `_global` (line 142) `def _global(self)`
- `__init__` (line 153) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 163) `def _spectral_dim(self, k, n_fit)` - *Dimensión espectral corregida*
- `_ramsey` (line 199) `def _ramsey(self)` - *Número de Ramsey topológico*
- `t_c` (line 218) `def t_c(self)` - *Tiempo crítico de percolación*
- `__init__` (line 230) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 237) `def _free_hamiltonian(self)`
- `_loss_potential` (line 242) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 248) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 251) `def coherence_quantum(self)`
- `__init__` (line 269) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 273) `def log_lik(self, model_pred)` - *Verosimilitud con escalas físicas realistas*
- `ln_bf` (line 288) `def ln_bf(self)` - *Factor de Bayes con penalización de complejidad*

#### `main4.py.py`
**Path:** `main4.py.py`
**File Doc:** *============================================================================= RESMA 4.0 – CÓDIGO COMPLETO FINAL (FIX: NameError, UnboundLocalError & Numérico) Autor: Tu nombre Fecha: 2025-11-21 Descripción: Implementación completa, estable y corregida. =============================================================================*

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 69) `class QuantumLeaf`
- `Universe` (line 102) `class Universe`
- `Network` (line 130) `class Network`
- `MyelinCavity` (line 198) `class MyelinCavity`
- `Bayes` (line 232) `class Bayes`

**Methods:**
- `simulate` (line 260) `def simulate(n_leaves, n_nodes, seed)`
- `dim` (line 52) `def dim(a)`
- `pt` (line 56) `def pt(k, o, c)`
- `size` (line 60) `def size(n)`
- `__post_init__` (line 75) `def __post_init__(self)`
- `spectral_density` (line 79) `def spectral_density(self, w)`
- `modular_entropy` (line 82) `def modular_entropy(self)`
- `bures_distance` (line 88) `def bures_distance(self, other)`
- `__init__` (line 103) `def __init__(self, n_leaves, seed)`
- `_gibbs` (line 111) `def _gibbs(self)`
- `_global` (line 120) `def _global(self)`
- `__init__` (line 131) `def __init__(self, n_nodes, seed)`
- `_spectral_dim` (line 140) `def _spectral_dim(self, k, n_fit)`
- `_ramsey` (line 176) `def _ramsey(self)`
- `t_c` (line 190) `def t_c(self)`
- `__init__` (line 199) `def __init__(self, n_modes)`
- `_free_hamiltonian` (line 205) `def _free_hamiltonian(self)`
- `_loss_potential` (line 210) `def _loss_potential(self)`
- `_pt_symmetry_condition` (line 216) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 219) `def coherence_quantum(self)`
- `__init__` (line 233) `def __init__(self, pred_resma, nulls)`
- `log_lik` (line 237) `def log_lik(self, model_pred)`
- `ln_bf` (line 244) `def ln_bf(self)`

#### `main5.py`
**Path:** `main5.py`
**File Doc:** *============================================================================ 0. PRINCIPIOS FUNDAMENTALES Y SISTEMA DE UNIDADES ============================================================================*

**Classes:**
- `RESMAConstants` (line 37) `class RESMAConstants` - *Constantes físicas y parámetros de la teoría RESMA 4.0*
- `PhysicalValidator` (line 70) `class PhysicalValidator` - *Validación de rangos físicos para todas las constantes RESMA 4.0*
- `QuantumLeaf` (line 116) `class QuantumLeaf` - *Hoja L_i de la Resma como estado KMS mean-field con espacio de Hilbert standard.
Implementación RESMA 4.0 con regularización Haagerup.*
- `RESMAUniverse` (line 179) `class RESMAUniverse` - *Multiverso como foliación medible sin matrices densas, con espacio de Hilbert standard.
Memoria: O(N_leaves) con regularización de transiciones.*
- `BranchingOperator` (line 260) `class BranchingOperator` - *Operador Ĥ que abre la Resma cuando β_i es no trivial.
Implementación local con operadores de salto SYK₈ (Pilar 4).*
- `EmunaOperator` (line 318) `class EmunaOperator` - *Operador P̂_E: proyección teleológica no lineal.
Implementación con muestreo Monte Carlo y espacio de Hardy H²(ℂ⁺) (Pilar 4).*
- `LindbladFractalDynamics` (line 406) `class LindbladFractalDynamics` - *SDE: ∂_t ρ = -i[H_eff, ρ] + γ L_mod[ρ] + g[ρ, log ρ_∞] + ξ(t)
Integración por Euler-Maruyama con control de precisión (Pilar 4).*
- `MyelinCavity` (line 539) `class MyelinCavity` - *Cavidad dieléctrica con Hamiltoniano H = H_0 + iV_loss.
Implementación 1D mean-field con R-simetría Spin(7) (Pilar 4).*
- `NeuralNetworkRESMA` (line 604) `class NeuralNetworkRESMA` - *Conectoma humano NO DIRIGIDO con homología persistente.
Implementación sparse para escalado con conversión a grafo no dirigido (Pilar 4).*
- `FreedomInvariant` (line 762) `class FreedomInvariant` - *L[G] = Δ_S* / S_top[G] invariante bajo gauge U(1)_R.
Cálculo topológico sin densidades matriciales (Pilar 4).*
- `NullModels` (line 816) `class NullModels` - *Modelos nulos para cálculo de Factor de Bayes.
Basados en teorías establecidas sin postulados holográficos de RESMA.*
- `ExperimentalPredictions` (line 877) `class ExperimentalPredictions` - *Predicciones falsables contra modelos nulos teóricos.
Factor de Bayes calculado con AIC y transformaciones logarítmicas (FIX).*
- `EmpiricalValidationProtocol` (line 975) `class EmpiricalValidationProtocol` - *Protocolo experimental para falsación controlada de RESMA 4.0.
Define setups experimentales y criterios de éxito.*

**Methods:**
- `simulate_resma_multiverse` (line 1052) `def simulate_resma_multiverse(n_leaves, n_nodes, seed, validate_empirical)` - *Pipeline completo RESMA 4.0 con verificaciones de integridad y protocolo de validación.
Diseñado para ejecución en Google Colab (Pilar 4).*
- `validate_dimension` (line 74) `def validate_dimension(alpha, tolerance)` - *α ∈ (0,1) por definición de dimensión fractal, con tolerancia experimental*
- `validate_pt_symmetry` (line 84) `def validate_pt_symmetry(kappa, Omega, chi)` - *Verificar κ/Ω < χ/Ω < 1 para PT-simetría (corregido con factor de seguridad)*
- `validate_connectome_size` (line 95) `def validate_connectome_size(n_nodes)` - *Límite inferior para conectoma biológico realista*
- `validate_spectral_dimension` (line 101) `def validate_spectral_dimension(dim)` - *Validar rango físico para dimensión espectral*
- `validate_percolation_time` (line 106) `def validate_percolation_time(t_c, expected, tolerance)` - *Validar tiempo de percolación contra predicción empírica*
- `__post_init__` (line 127) `def __post_init__(self)` - *Validaciones post-construcción (Pilar 3)*
- `spectral_density` (line 133) `def spectral_density(self, omega)` - *Densidad espectral continua ρ(ω) para álgebra tipo III₁ con regularización UV.
Evidencia: SYK₈ con Spin(7) tiene espectro continuo con gap infrarrojo.*
- `modular_entropy` (line 143) `def modular_entropy(self)` - *Entropía modular S = ∫ ρ(ω)logρ(ω) dω con regularización*
- `bures_distance` (line 151) `def bures_distance(self, other)`
- `_spectral_moments` (line 163) `def _spectral_moments(self, n)` - *Momentos espectrales Tr(ρ^k) para k=1..n con regularización*
- `haagerup_weight` (line 170) `def haagerup_weight(self)` - *Peso de Haagerup para regularización del operador modular*
- `__init__` (line 185) `def __init__(self, n_leaves, seed)` - *Args:
    n_leaves: Número de hojas (target: 1e5 en Colab con mean-field)
    seed: Reproducibilidad (Pilar 4)*
- `_initialize_leaves` (line 203) `def _initialize_leaves(self)` - *Genera hojas con gaps espectrales distribuidos exponencialmente*
- `_generate_gibbs_measure` (line 217) `def _generate_gibbs_measure(self)` - *Medida de Gibbs μ(i,j) = exp(-β·W₂²(ρ_i, ρ_j)) con normalización robusta*
- `_construct_global_state` (line 239) `def _construct_global_state(self)` - *Estado global: mapa de pesos por hoja (no matriz) con regularización*
- `compute_gibbs_free_energy` (line 251) `def compute_gibbs_free_energy(self)` - *Energía libre de Gibbs para validación termodinámica*
- `__init__` (line 266) `def __init__(self, leaf, threshold)`
- `_compute_holonomy` (line 272) `def _compute_holonomy(self)` - *Defecto de holonomía como variación del gap espectral*
- `_construct_cptp_map` (line 276) `def _construct_cptp_map(self)` - *Canal de Kraus: Ĥ(ρ) = Σ K_i ρ K_i† (solo si defecto > umbral)*
- `_local_jump_operator` (line 284) `def _local_jump_operator(self, power)` - *K_j = Δ_S^(1/4) · σ_j · Δ_S^(1/4) en representación de 2×2 local.
Aproximación mean-field: operadores de Pauli escalados por gap SYK₈.*
- `apply_branching` (line 300) `def apply_branching(self, state_vector)` - *Aplicar canal CPTP a vector de estado local (dim=2) con normalización*
- `__init__` (line 324) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 331) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺): Función analítica en semiplano superior*
- `_szego_projector` (line 335) `def _szego_projector(self)` - *Proyector P_E en base de Fourier positiva (dim reducida)*
- `_evaluation_functional` (line 345) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = lim_{ε→0⁺} exp(∫ log(⟨Φ_i|E⟩ + ε) dμ(i))*
- `project` (line 361) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (composición no lineal) - ROBUSTO CONTRA COLAPSO*
- `compute_teleological_overlap` (line 396) `def compute_teleological_overlap(self)` - *Calcular overlap teleológico con estado objetivo*
- `__init__` (line 412) `def __init__(self, universe, emuna)`
- `_effective_hamiltonian` (line 419) `def _effective_hamiltonian(self)` - *H_eff = Σ c_i H_i (mean-field: matriz 2×2 efectiva con gaps SYK₈)*
- `_modular_dissipator` (line 432) `def _modular_dissipator(self, state)` - *L_mod[ρ] = Δ_S^α ρ Δ_S^α - ½{Δ_S^α, ρ} con regularización*
- `_nonlinear_term` (line 444) `def _nonlinear_term(self, state)` - *G[ρ, log ρ_∞] = g[ρ, log ρ_∞] con regularización del logaritmo*
- `_stochastic_term` (line 452) `def _stochastic_term(self, dt)` - *Término estocástico ξ(t) con correlaciones cuánticas*
- `evolve` (line 459) `def evolve(self, rho0, t_span, n_steps)` - *Integración SDE con Euler-Maruyama y control de paso adaptativo.
Returns: trayectoria [n_steps, 2, 2]*
- `_normalize_density_matrix` (line 500) `def _normalize_density_matrix(self, state)` - *Normalizar matriz densidad y forzar hermiticidad*
- `_is_physical_state` (line 509) `def _is_physical_state(self, state)` - *Verificar si el estado es físico (hermitiano, traza=1, positivo)*
- `_correct_non_physical_state` (line 523) `def _correct_non_physical_state(self, state)` - *Corregir estado no físico proyectando en el cono de estados válidos*
- `__post_init__` (line 548) `def __post_init__(self)`
- `_free_hamiltonian` (line 555) `def _free_hamiltonian(self)` - *H_0: Modos colectivos con dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 561) `def _loss_potential(self)` - *V_loss ∝ (r⊥/a₀)^{2α} con α=0.702 (SYK₈)*
- `_compute_scalar_mass` (line 569) `def _compute_scalar_mass(self)` - *Campo escalar masivo para estabilización de Spin(7)*
- `_pt_symmetry_condition` (line 573) `def _pt_symmetry_condition(self)` - *Verificar κ/Ω < χ/Ω < 1 con parámetros corregidos*
- `coherence_quantum` (line 579) `def coherence_quantum(self)` - *Discordia cuántica aproximada con corrección PT*
- `__init__` (line 610) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 625) `def _generate_fractal_graph(self)` - *Generar grafo dirigido y convertir a NO DIRIGIDO para análisis espectral.
SOLUCIÓN RESMA 4.0: Conversión explícita con to_undirected().*
- `_spectral_dimension` (line 649) `def _spectral_dimension(self)` - *d_s = -2 lim_{λ→0⁺} log N(λ)/log λ usando normalized_laplacian_spectrum.
SOLUCIÓN RESMA 4.0: Uso de función especializada de NetworkX.*
- `_topological_ramsey` (line 680) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}
Requiere ripser (instalable en Colab: !pip install ripser)*
- `_compute_betti_numbers` (line 706) `def _compute_betti_numbers(self)` - *Calcular números de Betti para análisis topológico*
- `_graph_to_distance_matrix` (line 722) `def _graph_to_distance_matrix(self)` - *Matriz de distancias shortest-path (sparse CSR) para homología*
- `critical_percolation_time` (line 735) `def critical_percolation_time(self)` - *t_c = log⟨k⟩ / log R_Q * (N/N₀)^0.25
Pilar 1: Basado en organoides corticales (Quadrato et al., Cell 2017)*
- `is_coherent_subgraph` (line 747) `def is_coherent_subgraph(self, subgraph_nodes)` - *Verificar coherencia: subgrafo > 70% del total*
- `compute_network_entropy` (line 751) `def compute_network_entropy(self)` - *Entropía de la red basada en distribución de grados*
- `__init__` (line 768) `def __init__(self, network, universe)`
- `compute_entropy_gap` (line 772) `def compute_entropy_gap(self)` - *Δ_S* = ε_c en punto excepcional con corrección de regularización*
- `compute_pontryagin_number` (line 776) `def compute_pontryagin_number(self)` - *S_top[G] = χ(G)/|V| (número de Euler normalizado)*
- `compute_freedom` (line 792) `def compute_freedom(self)` - *L[G] = Δ_S* / S_top[G] con protección de división por cero*
- `is_gauge_invariant` (line 803) `def is_gauge_invariant(self)` - *|L[G] - 1| < 0.05 en estado crítico (invariante de libertad)*
- `ising_quantum` (line 823) `def ising_quantum(network)` - *Modelo de Ising cuántico transversal en red fractal.
Predice t_c sin SYK₈ ni E₈ (teoría efectiva estándar).*
- `syk4` (line 843) `def syk4(network)` - *SYK₄ estándar (sin R-simetría Spin(7) ni E₈).
Predice α sin postulado de retículo.*
- `random_network` (line 860) `def random_network(network)` - *Red aleatoria Erdős-Rényi sin percolación cuántica ni estructura.*
- `__init__` (line 883) `def __init__(self, resma, myelin, network, freedom)`
- `predict_all` (line 891) `def predict_all(self)` - *Predicciones RESMA 4.0 con valores empíricos objetivo*
- `_predict_diffraction_peak` (line 906) `def _predict_diffraction_peak(self)` - *q₀ = 2π/L_E8 (predicción de difracción UASED)*
- `compute_log_bayes_factor` (line 912) `def compute_log_bayes_factor(self)` - *log(BF) = ΔAIC/2 donde AIC = 2k - 2ln(L)
FIX RESMA 4.0: Usar espacio logarítmico para evitar desbordamiento.*
- `__init__` (line 981) `def __init__(self, predictions)`
- `_define_protocols` (line 985) `def _define_protocols(self)` - *Definir protocolos experimentales con parámetros técnicos*
- `evaluate_feasibility` (line 1014) `def evaluate_feasibility(self, budget, time_limit)` - *Evaluar viabilidad del protocolo completo*
- `simulate_experimental_outcome` (line 1027) `def simulate_experimental_outcome(self, protocol_name)` - *Simular resultado experimental con ruido realista*

#### `monitor_extremo.py`
**Path:** `monitor_extremo.py`

**Classes:**
- `SovereigntyMonitor` (line 26) `class SovereigntyMonitor` - *Implementación del Sovereignty Monitor basada en RESMA*
- `ModeloGrande` (line 72) `class ModeloGrande(Module)` - *Modelo grande diseñado para colapsar con entrenamiento extremo*

**Functions:**
- `setup_matplotlib_for_plotting` (line 19) `def setup_matplotlib_for_plotting()`

**Methods:**
- `generar_datos_toxico` (line 103) `def generar_datos_toxico()` - *Genera datos diseñados específicamente para causar colapso*
- `experimento_colapso_forzado` (line 126) `def experimento_colapso_forzado()` - *Experimento diseñado para forzar el colapso del modelo*
- `generar_graficos_extremos` (line 339) `def generar_graficos_extremos(historial)` - *Genera gráficos del experimento extremo*
- `__init__` (line 28) `def __init__(self, epsilon_c)`
- `calcular_libertad` (line 31) `def calcular_libertad(self, weights)` - *Calcula la métrica L (libertad) de una matriz de pesos*
- `evaluar_regimen` (line 63) `def evaluar_regimen(self, L)` - *Evalúa el régimen del modelo*
- `__init__` (line 74) `def __init__(self)`
- `forward` (line 88) `def forward(self, x)`
- `get_linear_layers` (line 100) `def get_linear_layers(self)`

#### `quick_monitor.py`
**Path:** `quick_monitor.py`

**Classes:**
- `SovereigntyMonitor` (line 26) `class SovereigntyMonitor` - *Implementación del Sovereignty Monitor basada en RESMA*
- `ModeloMNISTPequeno` (line 72) `class ModeloMNISTPequeno(Module)` - *Modelo CNN pequeño optimizado para entrenamiento rápido*

**Functions:**
- `setup_matplotlib_for_plotting` (line 19) `def setup_matplotlib_for_plotting()`

**Methods:**
- `generar_datos_mnist_rapido` (line 95) `def generar_datos_mnist_rapido()` - *Genera datos sintéticos tipo MNIST para experimento rápido*
- `entrenar_modelo_rapido` (line 116) `def entrenar_modelo_rapido()` - *Entrena modelo con monitoreo L en tiempo real*
- `generar_graficos_rapido` (line 295) `def generar_graficos_rapido(historial)` - *Genera gráficos de resultados del experimento rápido*
- `__init__` (line 28) `def __init__(self, epsilon_c)`
- `calcular_libertad` (line 31) `def calcular_libertad(self, weights)` - *Calcula la métrica L (libertad) de una matriz de pesos*
- `evaluar_regimen` (line 63) `def evaluar_regimen(self, L)` - *Evalúa el régimen del modelo*
- `__init__` (line 74) `def __init__(self)`
- `forward` (line 83) `def forward(self, x)`
- `get_linear_layers` (line 92) `def get_linear_layers(self)`

#### `main_experiment.py`
**Path:** `resma2/main_experiment.py`

**Functions:**
- `set_seed` (line 26) `def set_seed(seed)`
- `run_experiment` (line 31) `def run_experiment()`

#### `main_experiments.py`
**Path:** `resma2/main_experiments.py`

**Functions:**
- `inject_noise` (line 24) `def inject_noise(x, sigma)`
- `train_epoch` (line 27) `def train_epoch(model, loader, optim, obs, epoch)`
- `run` (line 61) `def run()`

#### `monitor.py`
**Path:** `resma2/monitor.py`

**Classes:**
- `Regime` (line 12) `class Regime(Enum)`
- `LayerDiagnostics` (line 18) `class LayerDiagnostics`
- `EpochSnapshot` (line 27) `class EpochSnapshot`
- `SovereigntyMonitor` (line 35) `class SovereigntyMonitor`

**Methods:**
- `__init__` (line 36) `def __init__(self, epsilon_c, patience, umbral_soberano, umbral_espurio, track_layers, verbose)`
- `_extract_weights` (line 50) `def _extract_weights(self, model)`
- `_calculate_svd_metrics` (line 58) `def _calculate_svd_metrics(self, weight_matrix)`
- `calcular_libertad` (line 84) `def calcular_libertad(self, weights)`
- `calculate` (line 92) `def calculate(self, model)`

#### `resma_app_mnist.py`
**Path:** `resma2/resma_app_mnist.py`

**Functions:**
- `add_quantum_noise` (line 25) `def add_quantum_noise(tensor, noise_factor)` - *Inyecta ruido gaussiano simulando fluctuaciones de vacío*
- `train` (line 30) `def train(model, device, train_loader, optimizer, epoch, observer)`
- `main` (line 65) `def main()`

#### `resma_breakpoint.py`
**Path:** `resma2/resma_breakpoint.py`

**Functions:**
- `find_break_point` (line 6) `def find_break_point()`

#### `resma_combat_test.py`
**Path:** `resma2/resma_combat_test.py`

**Functions:**
- `combat_test` (line 11) `def combat_test()`

#### `resma_core.py`
**Path:** `resma2/resma_core.py`

**Classes:**
- `PTSymmetricActivation` (line 14) `class PTSymmetricActivation(Module)`
- `E8LatticeLayer` (line 35) `class E8LatticeLayer(Module)`
- `RESMABrain` (line 65) `class RESMABrain(Module)`

**Methods:**
- `__init__` (line 15) `def __init__(self, omega, chi, kappa_init)`
- `forward` (line 26) `def forward(self, x)`
- `__init__` (line 37) `def __init__(self, in_features, out_features, q_order)`
- `_generate_ramsey_mask` (line 47) `def _generate_ramsey_mask(self)`
- `forward` (line 59) `def forward(self, x)`
- `__init__` (line 66) `def __init__(self, input_dim, hidden_dim, output_dim)`
- `forward` (line 74) `def forward(self, x)`

#### `resma_noise_phase_test.py`
**Path:** `resma2/resma_noise_phase_test.py`

**Functions:**
- `add_noise` (line 34) `def add_noise(x, sigma)`
- `measure_entropy` (line 37) `def measure_entropy(gate_tensor)`

#### `resma_observer.py`
**Path:** `resma2/resma_observer.py`

**Classes:**
- `QuantumState` (line 27) `class QuantumState` - *Snapshot del estado físico-estructural de la red*
- `RESMAObserver` (line 39) `class RESMAObserver`

**Methods:**
- `to_dict` (line 36) `def to_dict(self)`
- `__init__` (line 40) `def __init__(self, model, epsilon_c)`
- `_register_hooks` (line 51) `def _register_hooks(self)` - *Inyecta sondas en las capas PT para leer telemetría en tiempo real*
- `step` (line 68) `def step(self, epoch)` - *Ejecutar al final de cada época de entrenamiento/validación.
Fusiona métricas y determina la fase.*
- `report` (line 106) `def report(self, state)` - *Imprime reporte formateado a consola*
- `plot_phase_space` (line 118) `def plot_phase_space(self, save_path)` - *Genera el diagrama de fase: Estructura vs Dinámica*
- `hook_fn` (line 53) `def hook_fn(module, input, output)`

#### `resma_overload.py`
**Path:** `resma2/resma_overload.py`

**Functions:**
- `overload_test` (line 6) `def overload_test()`

#### `resma_train.py`
**Path:** `resma2/resma_train.py`

*No symbols extracted*

#### `resma_vision.py`
**Path:** `resma2/resma_vision.py`

**Functions:**
- `add_noise` (line 11) `def add_noise(tensor, factor)`
- `visualize_resma_perception` (line 14) `def visualize_resma_perception()`

#### `resma_vision_trained.py`
**Path:** `resma2/resma_vision_trained.py`

**Functions:**
- `add_noise` (line 11) `def add_noise(tensor, factor)`
- `visualize_trained_perception` (line 14) `def visualize_trained_perception()`

#### `resma4.10.py`
**Path:** `resma4.10.py`
**File Doc:** *============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (CÓDIGO DE PRODUCCIÓN COMPLETO) ============================================================================= GUARDAR COMO: resma4.12_fixed.py FIX CRÍTICO: Bug en MyelinCavity._loss_potential (n_nodes → n_modes)*

**Classes:**
- `RESMAConstants` (line 35) `class RESMAConstants`
- `GarnierTresTiempos` (line 72) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 112) `class OperadorDesdoblamiento` - *Operador de desdoblamiento D̂_G(φ) sobre el álgebra E8 genuina.
Construcción: sistema de raíces E8 → base de Chevalley → representación adjunta 248.*
- `SilencioActivoMonitor` (line 431) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 459) `class QuantumLeaf`
- `RESMAUniverse` (line 506) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 614) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 782) `class MyelinCavity`
- `ExperimentalPredictions` (line 818) `class ExperimentalPredictions`
- `ResourceMonitor` (line 865) `class ResourceMonitor`

**Methods:**
- `guardar_checkpoint` (line 877) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 935) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 963) `def _make_serializable(obj, depth, max_depth, _visited)` - *Convierte objetos a formato serializable de forma segura.

Args:
    obj: Objeto a serializar
    depth: Nivel de profundidad actual (auto-incremental)
    max_depth: Profundidad máxima permitida
    _visited: Diccionario de objetos ya procesados (para referencias circulares)*
- `simulate_resma_garnier` (line 1067) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 51) `def verify_pt_condition(cls)`
- `__post_init__` (line 75) `def __post_init__(self)`
- `epsilon_critico` (line 87) `def epsilon_critico(self)`
- `modulation_factor` (line 90) `def modulation_factor(self)`
- `to_dict` (line 93) `def to_dict(self)`
- `from_dict` (line 103) `def from_dict(cls, data)`
- `__init__` (line 118) `def __init__(self, garnier, dimension)`
- `_generate_e8_roots` (line 149) `def _generate_e8_roots()` - *Genera las 240 raíces de E8 en R⁸.
Retorna array (240,8) con forma: [positivas (120) | negativas (120)]
donde neg[j] = -pos[j].*
- `_idx` (line 189) `def _idx(self, root_vec)` - *Índice global (0..239) de una raíz.*
- `_compute_structure_constants` (line 197) `def _compute_structure_constants(self)` - *Constantes N_{α,β} para toda raíz α, β con α+β también raíz.
Retorna dict {(i,j): N_{α_i,α_j}} con ambas orientaciones.
Convención Chevalley:
  N_{α,β} = -(p+1) si α < β,  (p+1) si α > β,
  donde p es el entero max con β - pα raíz.*
- `_adjoint_matrix` (line 238) `def _adjoint_matrix(self, cartan, roots_coeff)` - *Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.

Base: |H₀⟩..|H₇⟩ (0-7), |E_α₀⟩..|E_α₁₁₉⟩ (8-127), |E_{-α₀}⟩..|E_{-α₁₁₉}⟩ (128-247)
con roots[gi] = α para gi en 0..119, roots[gi+120] = -α.

Args:
    cartan:     array[8] coeficientes c_i para H_i.
    roots_coeff: array[240] coeficientes d_γ para E_γ.*
- `_construir_generadores_e8` (line 330) `def _construir_generadores_e8(self)` - *Construye 3 generadores genuinos del álgebra E8 en la adjunta.
Cada uno corresponde a una dirección física del formalismo Garnier T³:

  G₀ = H₁         (escala C₀ = 1.0, tiempo físico)
  G₂ = H₂         (escala C₂ = 2.7, tiempo crítico)
  G₃ = E_{α₁} + E_{-α₁}  (escala C₃ = 7.3, tiempo teleológico)

Las raíces simples de E8 son:
  α₁ = (1,-1,0,0,0,0,0,0), α₂ = (0,1,-1,0,0,0,0,0)*
- `_hadamard_generalizado` (line 399) `def _hadamard_generalizado(self)`
- `operator` (line 408) `def operator(self)`
- `calcular_alpha_modificado` (line 423) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 432) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 436) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 443) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 466) `def __post_init__(self)`
- `spectral_density` (line 470) `def spectral_density(self, omega)`
- `bures_distance` (line 476) `def bures_distance(self, other)`
- `__init__` (line 507) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 543) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 554) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 583) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 594) `def _construct_global_state(self)`
- `_calcular_libertad` (line 603) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 606) `def _calcular_coherencia(self)`
- `__init__` (line 615) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 653) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 727) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 735) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 757) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 761) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 769) `def _validar_axioma_6(self)`
- `__init__` (line 783) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 800) `def _free_hamiltonian(self)`
- `_loss_potential` (line 805) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 811) `def _compute_scalar_mass(self)`
- `__init__` (line 819) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 824) `def compute_log_bayes_factor(self)`
- `get_memory_gb` (line 867) `def get_memory_gb()`
- `log_resources` (line 872) `def log_resources()`

#### `resma4.13.py`
**Path:** `resma4.13.py`
**File Doc:** *============================================================================= RESMA 4.13 – VECTORIZACIÓN MASIVA (HACK PRO) ============================================================================= OPTIMIZACIÓN: Inicialización y medida cuántica completamente vectorizadas en operaciones NumPy (matrices (N,500) + producto matricial Q@Q.T).*

**Classes:**
- `RESMAConstants` (line 35) `class RESMAConstants`
- `GarnierTresTiempos` (line 72) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 112) `class OperadorDesdoblamiento` - *Operador de desdoblamiento D̂_G(φ) sobre el álgebra E8 genuina.
Construcción: sistema de raíces E8 → base de Chevalley → representación adjunta 248.*
- `SilencioActivoMonitor` (line 431) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 459) `class QuantumLeaf`
- `RESMAUniverse` (line 506) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 639) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 807) `class MyelinCavity`
- `ExperimentalPredictions` (line 843) `class ExperimentalPredictions`
- `ResourceMonitor` (line 890) `class ResourceMonitor`

**Methods:**
- `guardar_checkpoint` (line 902) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 960) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 988) `def _make_serializable(obj, depth, max_depth, _visited)` - *Convierte objetos a formato serializable de forma segura.

Args:
    obj: Objeto a serializar
    depth: Nivel de profundidad actual (auto-incremental)
    max_depth: Profundidad máxima permitida
    _visited: Diccionario de objetos ya procesados (para referencias circulares)*
- `simulate_resma_garnier` (line 1092) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 51) `def verify_pt_condition(cls)`
- `__post_init__` (line 75) `def __post_init__(self)`
- `epsilon_critico` (line 87) `def epsilon_critico(self)`
- `modulation_factor` (line 90) `def modulation_factor(self)`
- `to_dict` (line 93) `def to_dict(self)`
- `from_dict` (line 103) `def from_dict(cls, data)`
- `__init__` (line 118) `def __init__(self, garnier, dimension)`
- `_generate_e8_roots` (line 149) `def _generate_e8_roots()` - *Genera las 240 raíces de E8 en R⁸.
Retorna array (240,8) con forma: [positivas (120) | negativas (120)]
donde neg[j] = -pos[j].*
- `_idx` (line 189) `def _idx(self, root_vec)` - *Índice global (0..239) de una raíz.*
- `_compute_structure_constants` (line 197) `def _compute_structure_constants(self)` - *Constantes N_{α,β} para toda raíz α, β con α+β también raíz.
Retorna dict {(i,j): N_{α_i,α_j}} con ambas orientaciones.
Convención Chevalley:
  N_{α,β} = -(p+1) si α < β,  (p+1) si α > β,
  donde p es el entero max con β - pα raíz.*
- `_adjoint_matrix` (line 238) `def _adjoint_matrix(self, cartan, roots_coeff)` - *Matriz 248×248 de ad(X) para X = Σ c_i H_i + Σ d_γ E_γ.

Base: |H₀⟩..|H₇⟩ (0-7), |E_α₀⟩..|E_α₁₁₉⟩ (8-127), |E_{-α₀}⟩..|E_{-α₁₁₉}⟩ (128-247)
con roots[gi] = α para gi en 0..119, roots[gi+120] = -α.

Args:
    cartan:     array[8] coeficientes c_i para H_i.
    roots_coeff: array[240] coeficientes d_γ para E_γ.*
- `_construir_generadores_e8` (line 330) `def _construir_generadores_e8(self)` - *Construye 3 generadores genuinos del álgebra E8 en la adjunta.
Cada uno corresponde a una dirección física del formalismo Garnier T³:

  G₀ = H₁         (escala C₀ = 1.0, tiempo físico)
  G₂ = H₂         (escala C₂ = 2.7, tiempo crítico)
  G₃ = E_{α₁} + E_{-α₁}  (escala C₃ = 7.3, tiempo teleológico)

Las raíces simples de E8 son:
  α₁ = (1,-1,0,0,0,0,0,0), α₂ = (0,1,-1,0,0,0,0,0)*
- `_hadamard_generalizado` (line 399) `def _hadamard_generalizado(self)`
- `operator` (line 408) `def operator(self)`
- `calcular_alpha_modificado` (line 423) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 432) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 436) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 443) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 466) `def __post_init__(self)`
- `spectral_density` (line 470) `def spectral_density(self, omega)`
- `bures_distance` (line 476) `def bures_distance(self, other)`
- `__init__` (line 507) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 543) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 553) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 608) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 619) `def _construct_global_state(self)`
- `_calcular_libertad` (line 628) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 631) `def _calcular_coherencia(self)`
- `__init__` (line 640) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 678) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 752) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 760) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 782) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 786) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 794) `def _validar_axioma_6(self)`
- `__init__` (line 808) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 825) `def _free_hamiltonian(self)`
- `_loss_potential` (line 830) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 836) `def _compute_scalar_mass(self)`
- `__init__` (line 844) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 849) `def compute_log_bayes_factor(self)`
- `get_memory_gb` (line 892) `def get_memory_gb()`
- `log_resources` (line 897) `def log_resources()`

#### `resma4.2.py`
**Path:** `resma4.2.py`
**File Doc:** *============================================================================= RESMA 4.2 – IMPLEMENTACIÓN COMPLETA CON CORRECCIONES NUMÉRICAS Integración de RESMA 4.0 (teoría completa) + RESMA 4.1 (fixes numéricos) Autor: Colaboración Claude + Usuario Fecha: 2025-11-21 =============================================================================*

**Classes:**
- `RESMAConstants` (line 38) `class RESMAConstants` - *Constantes físicas RESMA 4.0 con correcciones PT-simétricas*
- `PhysicalValidator` (line 78) `class PhysicalValidator`
- `QuantumLeaf` (line 109) `class QuantumLeaf` - *Hoja L_i como estado KMS con espacio de Hilbert standard*
- `RESMAUniverse` (line 162) `class RESMAUniverse` - *Multiverso como foliación medible, memoria O(N_leaves)*
- `EmunaOperator` (line 224) `class EmunaOperator` - *P̂_E: proyección teleológica no lineal en H²(ℂ⁺)*
- `MyelinCavity` (line 291) `class MyelinCavity` - *Cavidad dieléctrica H = H₀ + iV_loss con Spin(7)*
- `NeuralNetworkRESMA` (line 351) `class NeuralNetworkRESMA` - *Conectoma NO DIRIGIDO con homología persistente*
- `ExperimentalPredictions` (line 474) `class ExperimentalPredictions` - *Predicciones con BF logarítmico*

**Methods:**
- `simulate_resma_complete` (line 556) `def simulate_resma_complete(n_leaves, n_nodes, seed)` - *Pipeline RESMA 4.2 completo*
- `verify_pt_condition` (line 67) `def verify_pt_condition(cls)` - *Verificar condición PT: κ < χΩ*
- `validate_dimension` (line 80) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 88) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 96) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 101) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 117) `def __post_init__(self)`
- `spectral_density` (line 121) `def spectral_density(self, omega)` - *ρ(ω) con regularización UV*
- `modular_entropy` (line 126) `def modular_entropy(self)` - *S = -∫ ρ log ρ dω*
- `bures_distance` (line 136) `def bures_distance(self, other)` - *Distancia de Bures W₂(ρ₁, ρ₂)*
- `haagerup_weight` (line 154) `def haagerup_weight(self)` - *Peso de Haagerup para regularización*
- `__init__` (line 165) `def __init__(self, n_leaves, seed)`
- `_initialize_leaves` (line 176) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 189) `def _generate_gibbs_measure(self)` - *μ(i,j) = exp(-β·W₂²(ρᵢ, ρⱼ))*
- `_construct_global_state` (line 205) `def _construct_global_state(self)` - *Estado global: pesos por hoja*
- `compute_gibbs_free_energy` (line 216) `def compute_gibbs_free_energy(self)` - *F = -ln(Tr(μ)) / β*
- `__init__` (line 227) `def __init__(self, universe, n_samples)`
- `_construct_hardy_state` (line 234) `def _construct_hardy_state(self)` - *E(z) ∈ H²(ℂ⁺)*
- `_szego_projector` (line 238) `def _szego_projector(self)` - *Proyector en frecuencias positivas*
- `_evaluation_functional` (line 246) `def _evaluation_functional(self, state_weights)` - *Φ_E[|Ψ⟩] = exp(∫ log(⟨Φᵢ|E⟩) dμ)*
- `project` (line 258) `def project(self, state_vector)` - *P̂_E = P_E ∘ Φ_E (con interpolación adaptativa)*
- `__post_init__` (line 297) `def __post_init__(self)`
- `_free_hamiltonian` (line 304) `def _free_hamiltonian(self)` - *H₀: dispersión Ω(q) = Ω₀ + q² + χq³*
- `_loss_potential` (line 310) `def _loss_potential(self)` - *V_loss ∝ (r/a₀)^(2α)*
- `_compute_scalar_mass` (line 317) `def _compute_scalar_mass(self)` - *Campo escalar para estabilización Spin(7)*
- `_pt_symmetry_condition` (line 321) `def _pt_symmetry_condition(self)` - *κ < χΩ*
- `coherence_quantum` (line 327) `def coherence_quantum(self)` - *Coherencia cuántica con verificación espectral*
- `__init__` (line 354) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 366) `def _generate_fractal_graph(self)` - *Scale-free → NO DIRIGIDO*
- `_spectral_dimension` (line 381) `def _spectral_dimension(self)` - *d_s = -2 lim log N(λ)/log λ*
- `_topological_ramsey` (line 410) `def _topological_ramsey(self)` - *R_Q(G) = min{n | β_{n-1}(G) > 0}*
- `_compute_betti_numbers` (line 429) `def _compute_betti_numbers(self)` - *Números de Betti β₀, β₁*
- `_graph_to_distance_matrix` (line 445) `def _graph_to_distance_matrix(self)` - *Matriz de distancias para homología*
- `critical_percolation_time` (line 461) `def critical_percolation_time(self)` - *t_c = 21 · (N/N₀)^0.25 / log R_Q*
- `__init__` (line 477) `def __init__(self, universe, myelin, network)`
- `predict_all` (line 483) `def predict_all(self)` - *Predicciones RESMA 4.2*
- `compute_log_bayes_factor` (line 496) `def compute_log_bayes_factor(self)` - *ln(BF) con AIC*

#### `resma4.3.py`
**Path:** `resma4.3.py`
**File Doc:** *============================================================================= RESMA 4.3.1 – FIX: FrozenInstanceError + Reanudación Inteligente =============================================================================*

**Classes:**
- `ResourceMonitor` (line 33) `class ResourceMonitor`
- `RESMAConstants` (line 110) `class RESMAConstants`
- `QuantumLeaf` (line 142) `class QuantumLeaf` - *Hoja KMS - INMUTABLE pero con caché externo*
- `RESMAUniverse` (line 190) `class RESMAUniverse` - *Multiverso con construcción lazy*
- `PhysicalValidator` (line 269) `class PhysicalValidator`
- `MyelinCavity` (line 297) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 352) `class NeuralNetworkRESMA`
- `ExperimentalPredictions` (line 485) `class ExperimentalPredictions`

**Methods:**
- `guardar_checkpoint` (line 54) `def guardar_checkpoint(data, filename)` - *Guardado atómico con backup*
- `cargar_checkpoint` (line 84) `def cargar_checkpoint(filename)` - *Cargar checkpoint con fallback*
- `simulate_resma_with_checkpointing` (line 545) `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` - *Pipeline con reanudación inteligente desde checkpoints*
- `get_memory_gb` (line 35) `def get_memory_gb()`
- `check_memory_limit` (line 40) `def check_memory_limit()`
- `log_resources` (line 49) `def log_resources()`
- `verify_pt_condition` (line 127) `def verify_pt_condition(cls)`
- `__post_init__` (line 150) `def __post_init__(self)`
- `spectral_density` (line 154) `def spectral_density(self, omega)`
- `bures_distance` (line 158) `def bures_distance(self, other)` - *Distancia Bures con caché EXTERNO (no en instancia)*
- `__init__` (line 193) `def __init__(self, n_leaves, seed)`
- `_initialize_leaves` (line 215) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 227) `def _generate_gibbs_measure(self)` - *Matriz de medida con guardado incremental*
- `_construct_global_state` (line 254) `def _construct_global_state(self)`
- `validate_dimension` (line 271) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 279) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 287) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 292) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 302) `def __post_init__(self)`
- `_free_hamiltonian` (line 312) `def _free_hamiltonian(self)`
- `_loss_potential` (line 317) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 323) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 326) `def _pt_symmetry_condition(self)`
- `coherence_quantum` (line 331) `def coherence_quantum(self)`
- `__init__` (line 353) `def __init__(self, n_nodes, seed)`
- `_generate_fractal_graph` (line 372) `def _generate_fractal_graph(self)` - *Generar grafo por lotes*
- `_spectral_dimension` (line 401) `def _spectral_dimension(self)` - *Dimensión espectral con matriz sparse*
- `_topological_ramsey` (line 425) `def _topological_ramsey(self)` - *Ramsey topológico*
- `_compute_betti_numbers` (line 444) `def _compute_betti_numbers(self)` - *Números de Betti*
- `_graph_to_distance_matrix` (line 460) `def _graph_to_distance_matrix(self)` - *Matriz de distancias sparse*
- `critical_percolation_time` (line 476) `def critical_percolation_time(self)` - *Tiempo crítico de percolación*
- `__init__` (line 486) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 492) `def compute_log_bayes_factor(self)` - *ln(BF)*

#### `resma4.4.py`
**Path:** `resma4.4.py`
**File Doc:** *============================================================================= RESMA 4.3.2 – REANUDACIÓN REAL + SERIALIZACIÓN DE OBJETOS =============================================================================*

**Classes:**
- `ResourceMonitor` (line 34) `class ResourceMonitor`
- `RESMAConstants` (line 129) `class RESMAConstants`
- `QuantumLeaf` (line 160) `class QuantumLeaf` - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 203) `class RESMAUniverse` - *Multiverso con estado serializable*
- `PhysicalValidator` (line 316) `class PhysicalValidator`
- `MyelinCavity` (line 343) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 377) `class NeuralNetworkRESMA`

**Methods:**
- `guardar_checkpoint` (line 59) `def guardar_checkpoint(data, filename)` - *Guarda el estado COMPLETO de los objetos, no solo metadatos*
- `cargar_checkpoint` (line 93) `def cargar_checkpoint(filename)` - *Carga el estado COMPLETO desde disco*
- `simulate_resma_with_checkpointing` (line 554) `def simulate_resma_with_checkpointing(n_leaves, n_nodes, seed, resume)` - *Pipeline con reanudación que realmente carga objetos*
- `get_memory_gb` (line 36) `def get_memory_gb()`
- `check_memory_limit` (line 41) `def check_memory_limit()`
- `log_resources` (line 50) `def log_resources()`
- `verify_pt_condition` (line 146) `def verify_pt_condition(cls)`
- `__post_init__` (line 168) `def __post_init__(self)`
- `spectral_density` (line 172) `def spectral_density(self, omega)`
- `bures_distance` (line 176) `def bures_distance(self, other)` - *Distancia Bures con caché externo*
- `__init__` (line 206) `def __init__(self, n_leaves, seed, leaves, measure, global_state)` - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 258) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 269) `def _generate_gibbs_measure(self)` - *Matriz de medida*
- `_construct_global_state` (line 301) `def _construct_global_state(self)`
- `validate_dimension` (line 318) `def validate_dimension(alpha, tolerance)`
- `validate_pt_symmetry` (line 326) `def validate_pt_symmetry(kappa, Omega, chi)`
- `validate_connectome_size` (line 334) `def validate_connectome_size(n_nodes)`
- `validate_spectral_dimension` (line 339) `def validate_spectral_dimension(dim)`
- `__post_init__` (line 348) `def __post_init__(self)`
- `_free_hamiltonian` (line 358) `def _free_hamiltonian(self)`
- `_loss_potential` (line 363) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 369) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 372) `def _pt_symmetry_condition(self)`
- `__init__` (line 378) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti)` - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 446) `def _generate_fractal_graph(self)` - *Generar grafo por lotes*
- `_spectral_dimension` (line 475) `def _spectral_dimension(self)` - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 499) `def _topological_ramsey(self)` - *Ramsey topológico*
- `_compute_betti_numbers` (line 518) `def _compute_betti_numbers(self)` - *Números de Betti*
- `_graph_to_distance_matrix` (line 534) `def _graph_to_distance_matrix(self)` - *Matriz de distancias sparse*

#### `resma4.5.py`
**Path:** `resma4.5.py`
**File Doc:** *============================================================================= RESMA 4.3.3 – GARNIER INTEGRADO CON CORRECCIONES DIMENSIONALES =============================================================================*

**Classes:**
- `ResourceMonitor` (line 34) `class ResourceMonitor`
- `RESMAConstants` (line 135) `class RESMAConstants`
- `GarnierTresTiempos` (line 163) `class GarnierTresTiempos` - *Toro temporal T³ con parámetros ADIMENSIONALES.
C0, C2, C3 son ratios de escala, no velocidades.*
- `OperadorDesdoblamiento` (line 201) `class OperadorDesdoblamiento` - *D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E
Representación toy de E8 (248x248)*
- `SilencioActivoMonitor` (line 269) `class SilencioActivoMonitor` - *Monitor de Silencio-Activo: ΔS_loop < ε_c(ϕ)*
- `QuantumLeaf` (line 337) `class QuantumLeaf` - *Hoja KMS - INMUTABLE (SIN CAMBIOS)*
- `RESMAUniverse` (line 380) `class RESMAUniverse` - *Multiverso con estado serializable y desdoblamiento Garnier*
- `MyelinCavity` (line 486) `class MyelinCavity` - *Cavidad PT-simétrica (SIN CAMBIOS)*
- `NeuralNetworkRESMA` (line 517) `class NeuralNetworkRESMA` - *Red neuronal con embedding Garnier*
- `ExperimentalPredictions` (line 658) `class ExperimentalPredictions` - *Cálculos experimentales (SIN CAMBIOS)*

**Methods:**
- `guardar_checkpoint` (line 59) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 88) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 113) `def _make_serializable(obj)` - *Convierte objetos a formato serializable*
- `simulate_resma_garnier` (line 695) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)` - *Pipeline único con Garnier integrado*
- `get_memory_gb` (line 36) `def get_memory_gb()`
- `check_memory_limit` (line 41) `def check_memory_limit()`
- `log_resources` (line 50) `def log_resources()`
- `verify_pt_condition` (line 152) `def verify_pt_condition(cls)`
- `__post_init__` (line 170) `def __post_init__(self)`
- `factor_escala` (line 181) `def factor_escala(self, tiempo_idx)` - *Factor de escala para cada tiempo: 0=lento, 2=modular, 3=teleológico*
- `epsilon_critico` (line 185) `def epsilon_critico(self)` - *Entropía crítica de percolación (ADIMENSIONAL).
log(2) es la entropía de un bit cuántico crítico.*
- `to_dict` (line 192) `def to_dict(self)` - *Para serialización*
- `from_dict` (line 197) `def from_dict(cls, data)`
- `__init__` (line 206) `def __init__(self, garnier, dimension)`
- `_construir_generadores_E8` (line 214) `def _construir_generadores_E8(self)` - *Construye 3 generadores temporales (antis-Hermitianos)*
- `_hadamard_generalizado` (line 226) `def _hadamard_generalizado(self)` - *Operador de Hadamard en dimensión 248 (unitario)*
- `operator` (line 235) `def operator(self)` - *Construye D̂_G(ϕ) dimensionalmente consistente*
- `aplicar_a_estado` (line 254) `def aplicar_a_estado(self, estado)` - *Aplica desdoblamiento a un estado cuántico |Ψ⟩*
- `calcular_alpha_modificado` (line 260) `def calcular_alpha_modificado(self, alpha_base)` - *α'(ϕ) = α · tanh(C0/C3 · cos(ϕ₃))
Garantiza α' ∈ [0, α]*
- `__init__` (line 273) `def __init__(self, garnier, network)`
- `calcular_delta_s_loop` (line 278) `def calcular_delta_s_loop(self, rho_red)` - *ΔS_loop = S_vN(ρ_red) - log(b₁ + 1)
rho_red: matriz densidad reducida (si es None, se calcula)*
- `_calcular_rho_reducida_aproximada` (line 299) `def _calcular_rho_reducida_aproximada(self)` - *Aproximación: ρ_red = diag(grados) / sum(grados)*
- `es_silencio_activo` (line 307) `def es_silencio_activo(self, rho_red)` - *Verifica Silencio-Activo y calcula Libertad L.
Retorna: (condicion, libertad_L)*
- `umbral_percolacion` (line 324) `def umbral_percolacion(self)` - *Umbral de percolación para soberanía: 70% (Axioma 6)*
- `__post_init__` (line 345) `def __post_init__(self)`
- `spectral_density` (line 349) `def spectral_density(self, omega)`
- `bures_distance` (line 353) `def bures_distance(self, other)`
- `__init__` (line 383) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)` - *Constructor que puede recibir estado serializado*
- `_initialize_leaves` (line 420) `def _initialize_leaves(self)`
- `_generate_gibbs_measure` (line 431) `def _generate_gibbs_measure(self)` - *Matriz de medida sin desdoblamiento*
- `_aplicar_desdoblamiento_a_medida` (line 451) `def _aplicar_desdoblamiento_a_medida(self, measure)` - *Aplica D̂_G(ϕ) a la medida:
- M_ij → M_ij * (C0/C3)^(cos(ϕ₃))
- Normaliza después*
- `_construct_global_state` (line 471) `def _construct_global_state(self)`
- `_calcular_libertad_universo` (line 481) `def _calcular_libertad_universo(self)` - *Libertad del universo: L = 1/ε_c*
- `__init__` (line 488) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 499) `def _free_hamiltonian(self)`
- `_loss_potential` (line 504) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 510) `def _compute_scalar_mass(self)`
- `_pt_symmetry_condition` (line 513) `def _pt_symmetry_condition(self)`
- `__init__` (line 520) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)` - *Constructor que puede recibir grafo ya construido*
- `_generate_fractal_graph` (line 559) `def _generate_fractal_graph(self)` - *Generar grafo por lotes con conectividad controlada*
- `_spectral_dimension` (line 591) `def _spectral_dimension(self)` - *Dimensión espectral con eigenvalores sparse*
- `_topological_ramsey` (line 615) `def _topological_ramsey(self)` - *Ramsey topológico simplificado*
- `_compute_betti_numbers` (line 627) `def _compute_betti_numbers(self)` - *Números de Betti aproximados por ciclos locales*
- `_calcular_rho_reducida` (line 636) `def _calcular_rho_reducida(self)` - *Matriz densidad reducida del conectoma*
- `validar_axioma_6` (line 644) `def validar_axioma_6(self)` - *Verifica: conectividad > 70% para soberanía*
- `__init__` (line 661) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 666) `def compute_log_bayes_factor(self)` - *Calcula Factor de Bayes integrando Garnier*

#### `resma4.6.py`
**Path:** `resma4.6.py`
**File Doc:** *============================================================================= RESMA 4.3.5 – ZPE-SILENCIO ANTAGONISMO + CONECTOMA HUMANO =============================================================================*

**Classes:**
- `ResourceMonitor` (line 31) `class ResourceMonitor`
- `RESMAConstants` (line 139) `class RESMAConstants` - *Constantes físicas fundamentales*
- `GarnierTresTiempos` (line 169) `class GarnierTresTiempos` - ***TORO TEMPORAL T³ CON CANCELACIÓN ZPE**
- phi: Fase de desdoblamiento que controla anulación ZPE
- zpe_level: Nivel de fluctuaciones de punto cero [0,1]*
- `OperadorDesdoblamiento` (line 233) `class OperadorDesdoblamiento` - ***D̂_G(ϕ) = exp(i Σ_i φ_i H_i) · H_E**
**CONTRA-ZPE**: Opera en subespacio sin fluctuaciones*
- `SilencioActivoMonitor` (line 312) `class SilencioActivoMonitor` - ***MONITOR DE ANTAGONISMO ZPE-SILENCIO**
- Detecta cuando fluctuaciones cuánticas son coherentemente anuladas
- Mide nivel de "ruido de fondo cuántico" vs "silencio ontológico"*
- `QuantumLeaf` (line 441) `class QuantumLeaf` - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 484) `class RESMAUniverse` - *Multiverso con ZPE-Silencio integrado*
- `MyelinCavity` (line 608) `class MyelinCavity` - *Cavidad PT-simétrica con medición ZPE*
- `NeuralNetworkRESMA` (line 659) `class NeuralNetworkRESMA` - *Red neuronal con validación ZPE-Silencio*
- `ExperimentalPredictions` (line 853) `class ExperimentalPredictions` - *Cálculos experimentales unificados ZPE-Silencio*

**Methods:**
- `guardar_checkpoint` (line 56) `def guardar_checkpoint(data, filename)` - *Guarda estado completo con manejo robusto de errores*
- `cargar_checkpoint` (line 86) `def cargar_checkpoint(filename)` - *Carga checkpoint con fallback automático*
- `_make_serializable` (line 112) `def _make_serializable(obj)` - *Convierte objetos recursivamente a formato serializable*
- `simulate_resma_garnier` (line 906) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart, target_connectivity)` - *Pipeline completo RESMA 4.3.5 con antagonismo ZPE-Silencio*
- `get_memory_gb` (line 33) `def get_memory_gb()`
- `check_memory_limit` (line 38) `def check_memory_limit(threshold)`
- `log_resources` (line 47) `def log_resources()`
- `verify_pt_condition` (line 158) `def verify_pt_condition(cls)`
- `__post_init__` (line 181) `def __post_init__(self)`
- `factor_escala` (line 194) `def factor_escala(self, tiempo_idx)` - *Factor de escala con supresión ZPE*
- `epsilon_critico` (line 200) `def epsilon_critico(self)` - ***UMBRAL CRÍTICO CON ZPE**:
Cuando zpe_level → 0, ε_c → 0 (Silencio perfecto no necesita umbral)*
- `to_dict` (line 208) `def to_dict(self)` - *Serialización completa*
- `from_dict` (line 220) `def from_dict(cls, data)` - *Deserialización*
- `__init__` (line 238) `def __init__(self, garnier, dimension)`
- `_construir_generadores_E8_ZPE` (line 246) `def _construir_generadores_E8_ZPE(self)` - *GENERADORES CON CANCELACIÓN ZPE INTEGRADA*
- `_hadamard_generalizado_ZPE` (line 266) `def _hadamard_generalizado_ZPE(self)` - *HADAMARD CON ESPACIO NULO ZPE*
- `operator` (line 284) `def operator(self)` - *Construye D̂_G(ϕ) con cancelación ZPE*
- `alpha_modificado` (line 304) `def alpha_modificado(self, alpha_base)` - ***α'(ϕ) = α · tanh(C₀/C₃ · cos(ϕ₃) · (1 - zpe_level))***
- `__init__` (line 318) `def __init__(self, garnier, network)`
- `calcular_delta_s_loop` (line 327) `def calcular_delta_s_loop(self, rho_red)` - ***ΔS_loop = S_vN(ρ_red) - S_top + S_ZPE**
**NUEVO**: La entropía ZPE se SUMA a la entropía total*
- `_calcular_rho_reducida_aproximada` (line 367) `def _calcular_rho_reducida_aproximada(self)` - *Matriz densidad con modulación ZPE*
- `es_silencio_activo` (line 383) `def es_silencio_activo(self, rho_red)` - ***DETECCIÓN DE ANTAGONISMO**:
Retorna: (condicion, libertad_L, nivel_ZPE_cancelado)

**CONDICIÓN**: ZPE < 1% AND ΔS_loop < ε_c*
- `umbral_percolacion` (line 411) `def umbral_percolacion(self)` - *Umbral para soberanía: 70%*
- `modo_goldstone` (line 415) `def modo_goldstone(self)` - ***MODO GOLDSTONE DEL DOBLE CUÁNTICO**:
Excitación colectiva que anuncia ruptura de simetría ZPE*
- `__post_init__` (line 449) `def __post_init__(self)`
- `spectral_density` (line 453) `def spectral_density(self, omega)`
- `bures_distance` (line 457) `def bures_distance(self, other)`
- `__init__` (line 487) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 521) `def _initialize_leaves(self)` - *Inicializa hojas con temperatura efectiva afectada por ZPE*
- `_generate_gibbs_measure` (line 535) `def _generate_gibbs_measure(self)` - *Genera medida de Gibbs*
- `_aplicar_desdoblamiento_a_medida` (line 557) `def _aplicar_desdoblamiento_a_medida(self, measure)` - *Aplica desdoblamiento con supresión ZPE*
- `_construct_global_state` (line 584) `def _construct_global_state(self)` - *Construye estado global normalizado*
- `_calcular_libertad_universo` (line 603) `def _calcular_libertad_universo(self)` - *Libertad intrínseca con supresión ZPE*
- `__init__` (line 611) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 624) `def _free_hamiltonian(self)` - *Hamiltoniano con energía ZPE incluida*
- `_loss_potential` (line 633) `def _loss_potential(self)` - *Potencial de pérdida PT*
- `_compute_scalar_mass` (line 640) `def _compute_scalar_mass(self)`
- `_calcular_zpe` (line 643) `def _calcular_zpe(self)` - ***ENERGÍA DE PUNTO CERO TOTAL**:
E_ZPE = Σ_i ½ħω_i*
- `_pt_symmetry_condition` (line 655) `def _pt_symmetry_condition(self)`
- `__init__` (line 662) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_fractal_graph` (line 711) `def _generate_fractal_graph(self)` - *Genera grafo con densidad 0.75 (conectoma humano)*
- `_spectral_dimension` (line 753) `def _spectral_dimension(self)` - *Dimensión espectral con ZPE*
- `_topological_ramsey` (line 777) `def _topological_ramsey(self)` - *Ramsey topológico*
- `_compute_betti_numbers` (line 788) `def _compute_betti_numbers(self)` - *Números de Betti reales*
- `_calcular_rho_reducida` (line 804) `def _calcular_rho_reducida(self)` - *Matriz densidad con supresión ZPE*
- `_calcular_zpe_conectoma` (line 820) `def _calcular_zpe_conectoma(self)` - ***ENERGÍA ZPE DEL CONECTOMA**:
E_ZPE = Σ_i ½ħω_i (modos de Laplaciano)*
- `validar_axioma_6` (line 836) `def validar_axioma_6(self)` - ***AXIOMA 6**: Conectividad > 70% para soberanía*
- `__init__` (line 856) `def __init__(self, universe, myelin, network)`
- `compute_log_bayes_factor` (line 861) `def compute_log_bayes_factor(self)` - *Calcula Factor de Bayes con antagonismo ZPE-Silencio*

#### `resma4.7.py`
**Path:** `resma4.7.py`
**File Doc:** *============================================================================= RESMA 4.3.4 – CORRECCIONES CRÍTICAS GARNIER-MALET =============================================================================*

**Classes:**
- `RESMAConstants` (line 23) `class RESMAConstants`
- `GarnierTresTiempos` (line 46) `class GarnierTresTiempos` - *Toro temporal T³ con parámetros físicamente consistentes.
Basado en la teoría del desdoblamiento del tiempo de Garnier-Malet.*
- `OperadorDesdoblamiento` (line 94) `class OperadorDesdoblamiento` - *Operador de desdoblamiento D̂_G(φ) con estructura E8 simplificada*
- `SilencioActivoMonitor` (line 139) `class SilencioActivoMonitor` - *Monitor de condición de Silencio-Activo: ΔS_loop < ε_c(φ)*
- `QuantumLeaf` (line 190) `class QuantumLeaf`
- `RESMAUniverse` (line 223) `class RESMAUniverse` - *Multiverso cuántico con desdoblamiento Garnier-Malet*
- `NeuralNetworkRESMA` (line 336) `class NeuralNetworkRESMA` - *Red neuronal con topología realista que satisface Axioma 6*
- `ExperimentalPredictions` (line 484) `class ExperimentalPredictions` - *Cálculo de Factor de Bayes y predicciones*

**Methods:**
- `simulate_resma_garnier` (line 537) `def simulate_resma_garnier(n_leaves, n_nodes, seed)` - *Pipeline completo RESMA-Garnier con correcciones*
- `__post_init__` (line 53) `def __post_init__(self)`
- `_compute_coupling` (line 67) `def _compute_coupling(self)` - *Fuerza de acoplamiento entre tiempos*
- `factor_escala` (line 72) `def factor_escala(self, tiempo_idx)` - *Factor de escala temporal*
- `epsilon_critico` (line 77) `def epsilon_critico(self)` - *Entropía crítica con corrección de acoplamiento:
ε_c = log(2) · (C0/C3)² · (1 + ξ)*
- `modulation_factor` (line 85) `def modulation_factor(self)` - *Factor de modulación para la medida cuántica:
M = exp(-|φ₃ - π|/C3)
Máximo cuando φ₃ ≈ π (apertura temporal óptima)*
- `__init__` (line 98) `def __init__(self, garnier, dimension)`
- `_construir_generadores` (line 103) `def _construir_generadores(self)` - *Generadores temporales (anti-Hermitianos normalizados)*
- `operator` (line 115) `def operator(self)` - *Construye D̂_G(φ) = exp(i Σ φᵢHᵢ)*
- `aplicar_modulacion` (line 120) `def aplicar_modulacion(self, state_vector)` - *Aplica desdoblamiento a vector de estado*
- `calcular_alpha_modificado` (line 126) `def calcular_alpha_modificado(self, alpha_base)` - *α'(φ) = α · |cos(φ₃)|^(C0/C3)
Garantiza α' ∈ [0, α]*
- `__init__` (line 143) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 147) `def calcular_delta_s_loop(self, rho_red, b1)` - *ΔS_loop = S_vN(ρ) - log(b₁ + 1)

Args:
    rho_red: Matriz densidad reducida
    b1: Primer número de Betti (ciclos independientes)*
- `es_silencio_activo` (line 166) `def es_silencio_activo(self, rho_red, b1)` - *Verifica condición y calcula libertad L = 1/(ΔS + ε_c)

Returns:
    (condicion_satisfecha, libertad)*
- `spectral_density` (line 197) `def spectral_density(self, omega)`
- `bures_distance` (line 203) `def bures_distance(self, other)` - *Distancia de Bures simplificada*
- `__init__` (line 226) `def __init__(self, n_leaves, seed, garnier)`
- `_initialize_leaves` (line 252) `def _initialize_leaves(self)` - *Genera hojas con gaps distribuidos exponencialmente*
- `_generate_modulated_measure` (line 265) `def _generate_modulated_measure(self)` - *Genera medida de transición modulada por Garnier:
M_ij = exp(-β d²_ij) · φ(garnier)*
- `_construct_global_state` (line 312) `def _construct_global_state(self)` - *Estado global como distribución diagonal*
- `_calcular_libertad` (line 322) `def _calcular_libertad(self)` - *Libertad del universo: L_U = 1/ε_c*
- `_calcular_coherencia` (line 326) `def _calcular_coherencia(self)` - *Coherencia cuántica: suma de elementos off-diagonal*
- `__init__` (line 341) `def __init__(self, n_nodes, seed, garnier)`
- `_generate_realistic_network` (line 379) `def _generate_realistic_network(self)` - *Genera red con conectividad > 70% usando modelo realista:
- Watts-Strogatz para mundo pequeño
- Aumentación para alcanzar umbral*
- `_compute_betti_numbers` (line 424) `def _compute_betti_numbers(self)` - *Números de Betti: b0=componentes, b1=ciclos*
- `_spectral_dimension` (line 433) `def _spectral_dimension(self)` - *Dimensión espectral del Laplaciano*
- `_topological_ramsey` (line 455) `def _topological_ramsey(self)` - *Número de Ramsey topológico*
- `_calcular_rho_reducida` (line 460) `def _calcular_rho_reducida(self)` - *Matriz densidad de la red (normalizada por grados)*
- `_validar_axioma_6` (line 470) `def _validar_axioma_6(self)` - *Verifica conectividad > 70%*
- `__init__` (line 487) `def __init__(self, universe, network)`
- `compute_log_bayes_factor` (line 491) `def compute_log_bayes_factor(self)` - *ln(BF) ∝ log(L_red · L_univ)
Veredicto basado en libertad total*

#### `resma4.8.py`
**Path:** `resma4.8.py`
**File Doc:** *============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (Validada y lista para ejecución) ============================================================================= Código completo y ejecutable sin interrupciones de markdown Guardar como: resma_4_3_6.py*

**Classes:**
- `RESMAConstants` (line 37) `class RESMAConstants`
- `ResourceMonitor` (line 70) `class ResourceMonitor`
- `GarnierTresTiempos` (line 158) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 209) `class OperadorDesdoblamiento`
- `SilencioActivoMonitor` (line 256) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 284) `class QuantumLeaf`
- `RESMAUniverse` (line 331) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 433) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 584) `class MyelinCavity`
- `ExperimentalPredictions` (line 620) `class ExperimentalPredictions`

**Methods:**
- `guardar_checkpoint` (line 91) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 117) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 141) `def _make_serializable(obj)`
- `simulate_resma_garnier` (line 667) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 54) `def verify_pt_condition(cls)`
- `get_memory_gb` (line 72) `def get_memory_gb()`
- `check_memory_limit` (line 77) `def check_memory_limit()`
- `log_resources` (line 86) `def log_resources()`
- `__post_init__` (line 161) `def __post_init__(self)`
- `_compute_coupling` (line 179) `def _compute_coupling(self)`
- `epsilon_critico` (line 182) `def epsilon_critico(self)`
- `modulation_factor` (line 186) `def modulation_factor(self)`
- `to_dict` (line 189) `def to_dict(self)`
- `from_dict` (line 199) `def from_dict(cls, data)`
- `__init__` (line 210) `def __init__(self, garnier, dimension)`
- `_construir_generadores_aleatorios` (line 219) `def _construir_generadores_aleatorios(self)`
- `_hadamard_generalizado` (line 228) `def _hadamard_generalizado(self)`
- `operator` (line 233) `def operator(self)`
- `calcular_alpha_modificado` (line 248) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 257) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 261) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 268) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 291) `def __post_init__(self)`
- `spectral_density` (line 295) `def spectral_density(self, omega)`
- `bures_distance` (line 301) `def bures_distance(self, other)`
- `__init__` (line 332) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 362) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 373) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 402) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 413) `def _construct_global_state(self)`
- `_calcular_libertad` (line 422) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 425) `def _calcular_coherencia(self)`
- `__init__` (line 434) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 472) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 529) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 537) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 559) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 563) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 571) `def _validar_axioma_6(self)`
- `__init__` (line 585) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 602) `def _free_hamiltonian(self)`
- `_loss_potential` (line 607) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 613) `def _compute_scalar_mass(self)`
- `__init__` (line 621) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 626) `def compute_log_bayes_factor(self)`

#### `resma4.9.py`
**Path:** `resma4.9.py`
**File Doc:** *============================================================================= RESMA 4.3.6 – FUSIÓN CRÍTICA (CÓDIGO DE PRODUCCIÓN COMPLETO) ============================================================================= GUARDAR COMO: resma4.9_fixed.py COMPATIBLE CON: resma_checkpoint_v4_6.pkl FIX CRÍTICO: Conversión automática de listas a numpy arrays al cargar checkpoint*

**Classes:**
- `RESMAConstants` (line 34) `class RESMAConstants`
- `GarnierTresTiempos` (line 71) `class GarnierTresTiempos`
- `OperadorDesdoblamiento` (line 111) `class OperadorDesdoblamiento`
- `SilencioActivoMonitor` (line 158) `class SilencioActivoMonitor`
- `QuantumLeaf` (line 186) `class QuantumLeaf`
- `RESMAUniverse` (line 233) `class RESMAUniverse`
- `NeuralNetworkRESMA` (line 341) `class NeuralNetworkRESMA`
- `MyelinCavity` (line 492) `class MyelinCavity`
- `ExperimentalPredictions` (line 528) `class ExperimentalPredictions`
- `ResourceMonitor` (line 575) `class ResourceMonitor`

**Methods:**
- `guardar_checkpoint` (line 587) `def guardar_checkpoint(data, filename)`
- `cargar_checkpoint` (line 615) `def cargar_checkpoint(filename)`
- `_make_serializable` (line 639) `def _make_serializable(obj)`
- `simulate_resma_garnier` (line 655) `def simulate_resma_garnier(n_leaves, n_nodes, seed, resume, force_restart)`
- `verify_pt_condition` (line 50) `def verify_pt_condition(cls)`
- `__post_init__` (line 74) `def __post_init__(self)`
- `epsilon_critico` (line 86) `def epsilon_critico(self)`
- `modulation_factor` (line 89) `def modulation_factor(self)`
- `to_dict` (line 92) `def to_dict(self)`
- `from_dict` (line 102) `def from_dict(cls, data)`
- `__init__` (line 112) `def __init__(self, garnier, dimension)`
- `_construir_generadores_aleatorios` (line 121) `def _construir_generadores_aleatorios(self)`
- `_hadamard_generalizado` (line 130) `def _hadamard_generalizado(self)`
- `operator` (line 135) `def operator(self)`
- `calcular_alpha_modificado` (line 150) `def calcular_alpha_modificado(self, alpha_base)`
- `__init__` (line 159) `def __init__(self, garnier)`
- `calcular_delta_s_loop` (line 163) `def calcular_delta_s_loop(self, rho_red, b1)`
- `es_silencio_activo` (line 170) `def es_silencio_activo(self, rho_red, b1)`
- `__post_init__` (line 193) `def __post_init__(self)`
- `spectral_density` (line 197) `def spectral_density(self, omega)`
- `bures_distance` (line 203) `def bures_distance(self, other)`
- `__init__` (line 234) `def __init__(self, n_leaves, seed, leaves, measure, global_state, garnier)`
- `_initialize_leaves` (line 270) `def _initialize_leaves(self)`
- `_generate_complete_measure` (line 281) `def _generate_complete_measure(self)`
- `_aplicar_modulacion_garnier` (line 310) `def _aplicar_modulacion_garnier(self, measure)`
- `_construct_global_state` (line 321) `def _construct_global_state(self)`
- `_calcular_libertad` (line 330) `def _calcular_libertad(self)`
- `_calcular_coherencia` (line 333) `def _calcular_coherencia(self)`
- `__init__` (line 342) `def __init__(self, n_nodes, seed, graph, dim_spectral, ramsey, betti, garnier)`
- `_generate_realistic_modular_network` (line 380) `def _generate_realistic_modular_network(self)`
- `_compute_betti_numbers` (line 437) `def _compute_betti_numbers(self)`
- `_spectral_dimension` (line 445) `def _spectral_dimension(self)`
- `_topological_ramsey` (line 467) `def _topological_ramsey(self)`
- `_calcular_rho_reducida` (line 471) `def _calcular_rho_reducida(self)`
- `_validar_axioma_6` (line 479) `def _validar_axioma_6(self)`
- `__init__` (line 493) `def __init__(self, axon_length, radius, n_modes)`
- `_free_hamiltonian` (line 510) `def _free_hamiltonian(self)`
- `_loss_potential` (line 515) `def _loss_potential(self)`
- `_compute_scalar_mass` (line 521) `def _compute_scalar_mass(self)`
- `__init__` (line 529) `def __init__(self, universe, network, myelin)`
- `compute_log_bayes_factor` (line 534) `def compute_log_bayes_factor(self)`
- `get_memory_gb` (line 577) `def get_memory_gb()`
- `log_resources` (line 582) `def log_resources()`

#### `sovereignty_monitor.py`
**Path:** `sovereignty_monitor.py`

**Classes:**
- `SovereigntyMonitor` (line 29) `class SovereigntyMonitor` - *Implementación del Sovereignty Monitor basada en RESMA

Calcula L = 1 / (|S_vN(ρ) − log(rank(W) + 1)| + ε_c)*
- `CNNMNIST` (line 94) `class CNNMNIST(Module)` - *CNN para MNIST con arquitectura diseñada para monitoreo*
- `ExperimentoCompleto` (line 146) `class ExperimentoCompleto` - *Experimento completo para validar el Sovereignty Monitor*

**Functions:**
- `setup_matplotlib_for_plotting` (line 21) `def setup_matplotlib_for_plotting()` - *Setup matplotlib para visualización*

**Methods:**
- `cargar_datos` (line 126) `def cargar_datos()` - *Carga y prepara el dataset MNIST*
- `main` (line 433) `def main()` - *Función principal*
- `__init__` (line 35) `def __init__(self, epsilon_c)`
- `calcular_libertad` (line 38) `def calcular_libertad(self, weights)` - *Calcula la métrica L (libertad) de una matriz de pesos

Returns:
    tuple: (L, S_vn, rank_effective)*
- `evaluar_regimen` (line 85) `def evaluar_regimen(self, L)` - *Evalúa el régimen del modelo*
- `__init__` (line 96) `def __init__(self)`
- `forward` (line 110) `def forward(self, x)`
- `get_linear_layers` (line 122) `def get_linear_layers(self)` - *Retorna todas las capas lineales para monitoreo*
- `__init__` (line 149) `def __init__(self, num_epochs)`
- `calcular_metricas_sovereignty` (line 189) `def calcular_metricas_sovereignty(self)` - *Calcula métricas L para todas las capas lineales*
- `entrenar_epoca` (line 210) `def entrenar_epoca(self, epoca)` - *Entrena una época completa*
- `evaluar_epoca` (line 233) `def evaluar_epoca(self)` - *Evalúa el modelo en el conjunto de validación*
- `ejecutar_experimento` (line 253) `def ejecutar_experimento(self)` - *Ejecuta el experimento completo*
- `generar_graficos` (line 362) `def generar_graficos(self)` - *Genera gráficos comprehensivos de resultados*

#### `test_simple.py`
**Path:** `test_simple.py`

**Functions:**
- `test_basic_math` (line 8) `def test_basic_math()` - *Test de las matemáticas básicas RESMA*

#### `test_ultra_simple.py`
**Path:** `test_ultra_simple.py`

*No symbols extracted*

#### `train_mini_resma.py`
**Path:** `train_mini_resma.py`

**Functions:**
- `main` (line 8) `def main()`

#### `train_profile.py`
**Path:** `train_profile.py`

**Functions:**
- `main` (line 9) `def main()`

#### `visualize_resma.py`
**Path:** `visualize_resma.py`

**Functions:**
- `setup_matplotlib_for_plotting` (line 6) `def setup_matplotlib_for_plotting()` - *Setup matplotlib and seaborn for plotting with proper configuration.
Call this function before creating any plots to ensure proper rendering.*
- `diagnosticar_modelo` (line 30) `def diagnosticar_modelo(checkpoint_path)` - *Cargar y visualizar estado de red entrenada*

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
