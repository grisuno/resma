# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 42 | **Total Symbols Extracted:** 905 | **Total Imports:** 404

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
    resma4_5_py_ResourceMonitor["ResourceMonitor"]
    class resma4_5_py_ResourceMonitor cls;
    resma4_5_py --> resma4_5_py_ResourceMonitor
    resma4_5_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_5_py_guardar_checkpoint fn;
    resma4_5_py --> resma4_5_py_guardar_checkpoint
    resma4_5_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_5_py_cargar_checkpoint fn;
    resma4_5_py --> resma4_5_py_cargar_checkpoint
    resma4_5_py__make_serializable["_make_serializable"]
    class resma4_5_py__make_serializable fn;
    resma4_5_py --> resma4_5_py__make_serializable
    resma4_5_py_RESMAConstants["RESMAConstants"]
    class resma4_5_py_RESMAConstants cls;
    resma4_5_py --> resma4_5_py_RESMAConstants
    resma4_3_py["resma4.3.py (py)"]
    class resma4_3_py mod;
    resma4_3_py_ResourceMonitor["ResourceMonitor"]
    class resma4_3_py_ResourceMonitor cls;
    resma4_3_py --> resma4_3_py_ResourceMonitor
    resma4_3_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_3_py_guardar_checkpoint fn;
    resma4_3_py --> resma4_3_py_guardar_checkpoint
    resma4_3_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_3_py_cargar_checkpoint fn;
    resma4_3_py --> resma4_3_py_cargar_checkpoint
    resma4_3_py_RESMAConstants["RESMAConstants"]
    class resma4_3_py_RESMAConstants cls;
    resma4_3_py --> resma4_3_py_RESMAConstants
    resma4_3_py_QuantumLeaf["QuantumLeaf"]
    class resma4_3_py_QuantumLeaf cls;
    resma4_3_py --> resma4_3_py_QuantumLeaf
    resma4_10_py["resma4.10.py (py)"]
    class resma4_10_py mod;
    resma4_10_py_RESMAConstants["RESMAConstants"]
    class resma4_10_py_RESMAConstants cls;
    resma4_10_py --> resma4_10_py_RESMAConstants
    resma4_10_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_10_py_GarnierTresTiempos cls;
    resma4_10_py --> resma4_10_py_GarnierTresTiempos
    resma4_10_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_10_py_OperadorDesdoblamiento cls;
    resma4_10_py --> resma4_10_py_OperadorDesdoblamiento
    resma4_10_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_10_py_SilencioActivoMonitor cls;
    resma4_10_py --> resma4_10_py_SilencioActivoMonitor
    resma4_10_py_QuantumLeaf["QuantumLeaf"]
    class resma4_10_py_QuantumLeaf cls;
    resma4_10_py --> resma4_10_py_QuantumLeaf
    resma4_13_py["resma4.13.py (py)"]
    class resma4_13_py mod;
    resma4_13_py_RESMAConstants["RESMAConstants"]
    class resma4_13_py_RESMAConstants cls;
    resma4_13_py --> resma4_13_py_RESMAConstants
    resma4_13_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_13_py_GarnierTresTiempos cls;
    resma4_13_py --> resma4_13_py_GarnierTresTiempos
    resma4_13_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_13_py_OperadorDesdoblamiento cls;
    resma4_13_py --> resma4_13_py_OperadorDesdoblamiento
    resma4_13_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_13_py_SilencioActivoMonitor cls;
    resma4_13_py --> resma4_13_py_SilencioActivoMonitor
    resma4_13_py_QuantumLeaf["QuantumLeaf"]
    class resma4_13_py_QuantumLeaf cls;
    resma4_13_py --> resma4_13_py_QuantumLeaf
    resma4_8_py["resma4.8.py (py)"]
    class resma4_8_py mod;
    resma4_8_py_RESMAConstants["RESMAConstants"]
    class resma4_8_py_RESMAConstants cls;
    resma4_8_py --> resma4_8_py_RESMAConstants
    resma4_8_py_ResourceMonitor["ResourceMonitor"]
    class resma4_8_py_ResourceMonitor cls;
    resma4_8_py --> resma4_8_py_ResourceMonitor
    resma4_8_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_8_py_guardar_checkpoint fn;
    resma4_8_py --> resma4_8_py_guardar_checkpoint
    resma4_8_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_8_py_cargar_checkpoint fn;
    resma4_8_py --> resma4_8_py_cargar_checkpoint
    resma4_8_py__make_serializable["_make_serializable"]
    class resma4_8_py__make_serializable fn;
    resma4_8_py --> resma4_8_py__make_serializable
    resma4_6_py["resma4.6.py (py)"]
    class resma4_6_py mod;
    resma4_6_py_ResourceMonitor["ResourceMonitor"]
    class resma4_6_py_ResourceMonitor cls;
    resma4_6_py --> resma4_6_py_ResourceMonitor
    resma4_6_py_guardar_checkpoint["guardar_checkpoint"]
    class resma4_6_py_guardar_checkpoint fn;
    resma4_6_py --> resma4_6_py_guardar_checkpoint
    resma4_6_py_cargar_checkpoint["cargar_checkpoint"]
    class resma4_6_py_cargar_checkpoint fn;
    resma4_6_py --> resma4_6_py_cargar_checkpoint
    resma4_6_py__make_serializable["_make_serializable"]
    class resma4_6_py__make_serializable fn;
    resma4_6_py --> resma4_6_py__make_serializable
    resma4_6_py_RESMAConstants["RESMAConstants"]
    class resma4_6_py_RESMAConstants cls;
    resma4_6_py --> resma4_6_py_RESMAConstants
    main5_py["main5.py (py)"]
    class main5_py mod;
    main5_py_RESMAConstants["RESMAConstants"]
    class main5_py_RESMAConstants cls;
    main5_py --> main5_py_RESMAConstants
    main5_py_PhysicalValidator["PhysicalValidator"]
    class main5_py_PhysicalValidator cls;
    main5_py --> main5_py_PhysicalValidator
    main5_py_QuantumLeaf["QuantumLeaf"]
    class main5_py_QuantumLeaf cls;
    main5_py --> main5_py_QuantumLeaf
    main5_py_RESMAUniverse["RESMAUniverse"]
    class main5_py_RESMAUniverse cls;
    main5_py --> main5_py_RESMAUniverse
    main5_py_BranchingOperator["BranchingOperator"]
    class main5_py_BranchingOperator cls;
    main5_py --> main5_py_BranchingOperator
    resma4_9_py["resma4.9.py (py)"]
    class resma4_9_py mod;
    resma4_9_py_RESMAConstants["RESMAConstants"]
    class resma4_9_py_RESMAConstants cls;
    resma4_9_py --> resma4_9_py_RESMAConstants
    resma4_9_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_9_py_GarnierTresTiempos cls;
    resma4_9_py --> resma4_9_py_GarnierTresTiempos
    resma4_9_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_9_py_OperadorDesdoblamiento cls;
    resma4_9_py --> resma4_9_py_OperadorDesdoblamiento
    resma4_9_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_9_py_SilencioActivoMonitor cls;
    resma4_9_py --> resma4_9_py_SilencioActivoMonitor
    resma4_9_py_QuantumLeaf["QuantumLeaf"]
    class resma4_9_py_QuantumLeaf cls;
    resma4_9_py --> resma4_9_py_QuantumLeaf
    main_py["main.py (py)"]
    class main_py mod;
    main_py_RESMAConstants["RESMAConstants"]
    class main_py_RESMAConstants cls;
    main_py --> main_py_RESMAConstants
    main_py_PhysicalValidator["PhysicalValidator"]
    class main_py_PhysicalValidator cls;
    main_py --> main_py_PhysicalValidator
    main_py_QuantumLeaf["QuantumLeaf"]
    class main_py_QuantumLeaf cls;
    main_py --> main_py_QuantumLeaf
    main_py_RESMAUniverse["RESMAUniverse"]
    class main_py_RESMAUniverse cls;
    main_py --> main_py_RESMAUniverse
    main_py_BranchingOperator["BranchingOperator"]
    class main_py_BranchingOperator cls;
    main_py --> main_py_BranchingOperator
    main2_py["main2.py (py)"]
    class main2_py mod;
    main2_py_RESMAConstants["RESMAConstants"]
    class main2_py_RESMAConstants cls;
    main2_py --> main2_py_RESMAConstants
    main2_py_PhysicalValidator["PhysicalValidator"]
    class main2_py_PhysicalValidator cls;
    main2_py --> main2_py_PhysicalValidator
    main2_py_QuantumLeaf["QuantumLeaf"]
    class main2_py_QuantumLeaf cls;
    main2_py --> main2_py_QuantumLeaf
    main2_py_RESMAUniverse["RESMAUniverse"]
    class main2_py_RESMAUniverse cls;
    main2_py --> main2_py_RESMAUniverse
    main2_py_BranchingOperator["BranchingOperator"]
    class main2_py_BranchingOperator cls;
    main2_py --> main2_py_BranchingOperator
    resma4_2_py["resma4.2.py (py)"]
    class resma4_2_py mod;
    resma4_2_py_RESMAConstants["RESMAConstants"]
    class resma4_2_py_RESMAConstants cls;
    resma4_2_py --> resma4_2_py_RESMAConstants
    resma4_2_py_PhysicalValidator["PhysicalValidator"]
    class resma4_2_py_PhysicalValidator cls;
    resma4_2_py --> resma4_2_py_PhysicalValidator
    resma4_2_py_QuantumLeaf["QuantumLeaf"]
    class resma4_2_py_QuantumLeaf cls;
    resma4_2_py --> resma4_2_py_QuantumLeaf
    resma4_2_py_RESMAUniverse["RESMAUniverse"]
    class resma4_2_py_RESMAUniverse cls;
    resma4_2_py --> resma4_2_py_RESMAUniverse
    resma4_2_py_EmunaOperator["EmunaOperator"]
    class resma4_2_py_EmunaOperator cls;
    resma4_2_py --> resma4_2_py_EmunaOperator
    main3_py["main3.py (py)"]
    class main3_py mod;
    main3_py_RC["RC"]
    class main3_py_RC cls;
    main3_py --> main3_py_RC
    main3_py_Validator["Validator"]
    class main3_py_Validator cls;
    main3_py --> main3_py_Validator
    main3_py_QuantumLeaf["QuantumLeaf"]
    class main3_py_QuantumLeaf cls;
    main3_py --> main3_py_QuantumLeaf
    main3_py_Universe["Universe"]
    class main3_py_Universe cls;
    main3_py --> main3_py_Universe
    main3_py_Network["Network"]
    class main3_py_Network cls;
    main3_py --> main3_py_Network
    main4_py_py["main4.py.py (py)"]
    class main4_py_py mod;
    main4_py_py_RC["RC"]
    class main4_py_py_RC cls;
    main4_py_py --> main4_py_py_RC
    main4_py_py_Validator["Validator"]
    class main4_py_py_Validator cls;
    main4_py_py --> main4_py_py_Validator
    main4_py_py_QuantumLeaf["QuantumLeaf"]
    class main4_py_py_QuantumLeaf cls;
    main4_py_py --> main4_py_py_QuantumLeaf
    main4_py_py_Universe["Universe"]
    class main4_py_py_Universe cls;
    main4_py_py --> main4_py_py_Universe
    main4_py_py_Network["Network"]
    class main4_py_py_Network cls;
    main4_py_py --> main4_py_py_Network
    sovereignty_monitor_py["sovereignty_monitor.py (py)"]
    class sovereignty_monitor_py mod;
    sovereignty_monitor_py_setup_matplotlib_for_plotting["setup_matplotlib_for_plotting"]
    class sovereignty_monitor_py_setup_matplotlib_for_plotting fn;
    sovereignty_monitor_py --> sovereignty_monitor_py_setup_matplotlib_for_plotting
    sovereignty_monitor_py_SovereigntyMonitor["SovereigntyMonitor"]
    class sovereignty_monitor_py_SovereigntyMonitor cls;
    sovereignty_monitor_py --> sovereignty_monitor_py_SovereigntyMonitor
    sovereignty_monitor_py_CNNMNIST["CNNMNIST"]
    class sovereignty_monitor_py_CNNMNIST cls;
    sovereignty_monitor_py --> sovereignty_monitor_py_CNNMNIST
    sovereignty_monitor_py_cargar_datos["cargar_datos"]
    class sovereignty_monitor_py_cargar_datos fn;
    sovereignty_monitor_py --> sovereignty_monitor_py_cargar_datos
    sovereignty_monitor_py_ExperimentoCompleto["ExperimentoCompleto"]
    class sovereignty_monitor_py_ExperimentoCompleto cls;
    sovereignty_monitor_py --> sovereignty_monitor_py_ExperimentoCompleto
    resma4_7_py["resma4.7.py (py)"]
    class resma4_7_py mod;
    resma4_7_py_RESMAConstants["RESMAConstants"]
    class resma4_7_py_RESMAConstants cls;
    resma4_7_py --> resma4_7_py_RESMAConstants
    resma4_7_py_GarnierTresTiempos["GarnierTresTiempos"]
    class resma4_7_py_GarnierTresTiempos cls;
    resma4_7_py --> resma4_7_py_GarnierTresTiempos
    resma4_7_py_OperadorDesdoblamiento["OperadorDesdoblamiento"]
    class resma4_7_py_OperadorDesdoblamiento cls;
    resma4_7_py --> resma4_7_py_OperadorDesdoblamiento
    resma4_7_py_SilencioActivoMonitor["SilencioActivoMonitor"]
    class resma4_7_py_SilencioActivoMonitor cls;
    resma4_7_py --> resma4_7_py_SilencioActivoMonitor
    resma4_7_py_QuantumLeaf["QuantumLeaf"]
    class resma4_7_py_QuantumLeaf cls;
    resma4_7_py --> resma4_7_py_QuantumLeaf
    main4_1_py["main4.1.py (py)"]
    class main4_1_py mod;
    main4_1_py_RC["RC"]
    class main4_1_py_RC cls;
    main4_1_py --> main4_1_py_RC
    main4_1_py_Validator["Validator"]
    class main4_1_py_Validator cls;
    main4_1_py --> main4_1_py_Validator
    main4_1_py_QuantumLeaf["QuantumLeaf"]
    class main4_1_py_QuantumLeaf cls;
    main4_1_py --> main4_1_py_QuantumLeaf
    main4_1_py_Universe["Universe"]
    class main4_1_py_Universe cls;
    main4_1_py --> main4_1_py_Universe
    main4_1_py_Network["Network"]
    class main4_1_py_Network cls;
    main4_1_py --> main4_1_py_Network
    monitor_extremo_py["monitor_extremo.py (py)"]
    class monitor_extremo_py mod;
    monitor_extremo_py_setup_matplotlib_for_plotting["setup_matplotlib_for_plotting"]
    class monitor_extremo_py_setup_matplotlib_for_plotting fn;
    monitor_extremo_py --> monitor_extremo_py_setup_matplotlib_for_plotting
    monitor_extremo_py_SovereigntyMonitor["SovereigntyMonitor"]
    class monitor_extremo_py_SovereigntyMonitor cls;
    monitor_extremo_py --> monitor_extremo_py_SovereigntyMonitor
    monitor_extremo_py_ModeloGrande["ModeloGrande"]
    class monitor_extremo_py_ModeloGrande cls;
    monitor_extremo_py --> monitor_extremo_py_ModeloGrande
    monitor_extremo_py_generar_datos_toxico["generar_datos_toxico"]
    class monitor_extremo_py_generar_datos_toxico fn;
    monitor_extremo_py --> monitor_extremo_py_generar_datos_toxico
    monitor_extremo_py_experimento_colapso_forzado["experimento_colapso_forzado"]
    class monitor_extremo_py_experimento_colapso_forzado fn;
    monitor_extremo_py --> monitor_extremo_py_experimento_colapso_forzado
    quick_monitor_py["quick_monitor.py (py)"]
    class quick_monitor_py mod;
    quick_monitor_py_setup_matplotlib_for_plotting["setup_matplotlib_for_plotting"]
    class quick_monitor_py_setup_matplotlib_for_plotting fn;
    quick_monitor_py --> quick_monitor_py_setup_matplotlib_for_plotting
    quick_monitor_py_SovereigntyMonitor["SovereigntyMonitor"]
    class quick_monitor_py_SovereigntyMonitor cls;
    quick_monitor_py --> quick_monitor_py_SovereigntyMonitor
    quick_monitor_py_ModeloMNISTPequeno["ModeloMNISTPequeno"]
    class quick_monitor_py_ModeloMNISTPequeno cls;
    quick_monitor_py --> quick_monitor_py_ModeloMNISTPequeno
    quick_monitor_py_generar_datos_mnist_rapido["generar_datos_mnist_rapido"]
    class quick_monitor_py_generar_datos_mnist_rapido fn;
    quick_monitor_py --> quick_monitor_py_generar_datos_mnist_rapido
    quick_monitor_py_entrenar_modelo_rapido["entrenar_modelo_rapido"]
    class quick_monitor_py_entrenar_modelo_rapido fn;
    quick_monitor_py --> quick_monitor_py_entrenar_modelo_rapido
    resma2_main_experiments_py["main_experiments.py (py)"]
    class resma2_main_experiments_py mod;
    resma2_main_experiments_py_inject_noise["inject_noise"]
    class resma2_main_experiments_py_inject_noise fn;
    resma2_main_experiments_py --> resma2_main_experiments_py_inject_noise
    resma2_main_experiments_py_train_epoch["train_epoch"]
    class resma2_main_experiments_py_train_epoch fn;
    resma2_main_experiments_py --> resma2_main_experiments_py_train_epoch
    resma2_main_experiments_py_run["run"]
    class resma2_main_experiments_py_run fn;
    resma2_main_experiments_py --> resma2_main_experiments_py_run
    resma2_resma_app_mnist_py["resma_app_mnist.py (py)"]
    class resma2_resma_app_mnist_py mod;
    resma2_resma_app_mnist_py_add_quantum_noise["add_quantum_noise"]
    class resma2_resma_app_mnist_py_add_quantum_noise fn;
    resma2_resma_app_mnist_py --> resma2_resma_app_mnist_py_add_quantum_noise
    resma2_resma_app_mnist_py_train["train"]
    class resma2_resma_app_mnist_py_train fn;
    resma2_resma_app_mnist_py --> resma2_resma_app_mnist_py_train
    resma2_resma_app_mnist_py_main["main"]
    class resma2_resma_app_mnist_py_main fn;
    resma2_resma_app_mnist_py --> resma2_resma_app_mnist_py_main
    test_ultra_simple_py["test_ultra_simple.py (py)"]
    class test_ultra_simple_py mod;
    garnier_nn_py["garnier_nn.py (py)"]
    class garnier_nn_py mod;
    garnier_nn_py_GarnierLayer["GarnierLayer"]
    class garnier_nn_py_GarnierLayer cls;
    garnier_nn_py --> garnier_nn_py_GarnierLayer
    garnier_nn_py_SilencioActivoNetwork["SilencioActivoNetwork"]
    class garnier_nn_py_SilencioActivoNetwork cls;
    garnier_nn_py --> garnier_nn_py_SilencioActivoNetwork
    garnier_nn_py___init__["__init__"]
    class garnier_nn_py___init__ fn;
    garnier_nn_py --> garnier_nn_py___init__
    garnier_nn_py_forward["forward"]
    class garnier_nn_py_forward fn;
    garnier_nn_py --> garnier_nn_py_forward
    garnier_nn_py___init__["__init__"]
    class garnier_nn_py___init__ fn;
    garnier_nn_py --> garnier_nn_py___init__
    resma2_resma_observer_py["resma_observer.py (py)"]
    class resma2_resma_observer_py mod;
    resma2_resma_observer_py_QuantumState["QuantumState"]
    class resma2_resma_observer_py_QuantumState cls;
    resma2_resma_observer_py --> resma2_resma_observer_py_QuantumState
    resma2_resma_observer_py_RESMAObserver["RESMAObserver"]
    class resma2_resma_observer_py_RESMAObserver cls;
    resma2_resma_observer_py --> resma2_resma_observer_py_RESMAObserver
    resma2_resma_observer_py_to_dict["to_dict"]
    class resma2_resma_observer_py_to_dict fn;
    resma2_resma_observer_py --> resma2_resma_observer_py_to_dict
    resma2_resma_observer_py___init__["__init__"]
    class resma2_resma_observer_py___init__ fn;
    resma2_resma_observer_py --> resma2_resma_observer_py___init__
    resma2_resma_observer_py__register_hooks["_register_hooks"]
    class resma2_resma_observer_py__register_hooks fn;
    resma2_resma_observer_py --> resma2_resma_observer_py__register_hooks
    visualize_resma_py["visualize_resma.py (py)"]
    class visualize_resma_py mod;
    visualize_resma_py_setup_matplotlib_for_plotting["setup_matplotlib_for_plotting"]
    class visualize_resma_py_setup_matplotlib_for_plotting fn;
    visualize_resma_py --> visualize_resma_py_setup_matplotlib_for_plotting
    visualize_resma_py_diagnosticar_modelo["diagnosticar_modelo"]
    class visualize_resma_py_diagnosticar_modelo fn;
    visualize_resma_py --> visualize_resma_py_diagnosticar_modelo
    train_profile_py["train_profile.py (py)"]
    class train_profile_py mod;
    train_profile_py_main["main"]
    class train_profile_py_main fn;
    train_profile_py --> train_profile_py_main
    resma2_resma_core_py["resma_core.py (py)"]
    class resma2_resma_core_py mod;
    resma2_resma_core_py_PTSymmetricActivation["PTSymmetricActivation"]
    class resma2_resma_core_py_PTSymmetricActivation cls;
    resma2_resma_core_py --> resma2_resma_core_py_PTSymmetricActivation
    resma2_resma_core_py_E8LatticeLayer["E8LatticeLayer"]
    class resma2_resma_core_py_E8LatticeLayer cls;
    resma2_resma_core_py --> resma2_resma_core_py_E8LatticeLayer
    resma2_resma_core_py_RESMABrain["RESMABrain"]
    class resma2_resma_core_py_RESMABrain cls;
    resma2_resma_core_py --> resma2_resma_core_py_RESMABrain
    resma2_resma_core_py___init__["__init__"]
    class resma2_resma_core_py___init__ fn;
    resma2_resma_core_py --> resma2_resma_core_py___init__
    resma2_resma_core_py_forward["forward"]
    class resma2_resma_core_py_forward fn;
    resma2_resma_core_py --> resma2_resma_core_py_forward
    resma2_monitor_py["monitor.py (py)"]
    class resma2_monitor_py mod;
    resma2_monitor_py_Regime["Regime"]
    class resma2_monitor_py_Regime cls;
    resma2_monitor_py --> resma2_monitor_py_Regime
    resma2_monitor_py_LayerDiagnostics["LayerDiagnostics"]
    class resma2_monitor_py_LayerDiagnostics cls;
    resma2_monitor_py --> resma2_monitor_py_LayerDiagnostics
    resma2_monitor_py_EpochSnapshot["EpochSnapshot"]
    class resma2_monitor_py_EpochSnapshot cls;
    resma2_monitor_py --> resma2_monitor_py_EpochSnapshot
    resma2_monitor_py_SovereigntyMonitor["SovereigntyMonitor"]
    class resma2_monitor_py_SovereigntyMonitor cls;
    resma2_monitor_py --> resma2_monitor_py_SovereigntyMonitor
    resma2_monitor_py___init__["__init__"]
    class resma2_monitor_py___init__ fn;
    resma2_monitor_py --> resma2_monitor_py___init__
    demo_mini_resma_py["demo_mini_resma.py (py)"]
    class demo_mini_resma_py mod;
    demo_mini_resma_py_GarnierLayer["GarnierLayer"]
    class demo_mini_resma_py_GarnierLayer cls;
    demo_mini_resma_py --> demo_mini_resma_py_GarnierLayer
    demo_mini_resma_py_demo_resma["demo_resma"]
    class demo_mini_resma_py_demo_resma fn;
    demo_mini_resma_py --> demo_mini_resma_py_demo_resma
    demo_mini_resma_py___init__["__init__"]
    class demo_mini_resma_py___init__ fn;
    demo_mini_resma_py --> demo_mini_resma_py___init__
    demo_mini_resma_py_forward["forward"]
    class demo_mini_resma_py_forward fn;
    demo_mini_resma_py --> demo_mini_resma_py_forward
    resma2_main_experiment_py["main_experiment.py (py)"]
    class resma2_main_experiment_py mod;
    resma2_main_experiment_py_set_seed["set_seed"]
    class resma2_main_experiment_py_set_seed fn;
    resma2_main_experiment_py --> resma2_main_experiment_py_set_seed
    resma2_main_experiment_py_run_experiment["run_experiment"]
    class resma2_main_experiment_py_run_experiment fn;
    resma2_main_experiment_py --> resma2_main_experiment_py_run_experiment
    resma2_resma_noise_phase_test_py["resma_noise_phase_test.py (py)"]
    class resma2_resma_noise_phase_test_py mod;
    resma2_resma_noise_phase_test_py_add_noise["add_noise"]
    class resma2_resma_noise_phase_test_py_add_noise fn;
    resma2_resma_noise_phase_test_py --> resma2_resma_noise_phase_test_py_add_noise
    resma2_resma_noise_phase_test_py_measure_entropy["measure_entropy"]
    class resma2_resma_noise_phase_test_py_measure_entropy fn;
    resma2_resma_noise_phase_test_py --> resma2_resma_noise_phase_test_py_measure_entropy
    resma2_resma_vision_py["resma_vision.py (py)"]
    class resma2_resma_vision_py mod;
    resma2_resma_vision_py_add_noise["add_noise"]
    class resma2_resma_vision_py_add_noise fn;
    resma2_resma_vision_py --> resma2_resma_vision_py_add_noise
    resma2_resma_vision_py_visualize_resma_perception["visualize_resma_perception"]
    class resma2_resma_vision_py_visualize_resma_perception fn;
    resma2_resma_vision_py --> resma2_resma_vision_py_visualize_resma_perception
    resma2_resma_combat_test_py["resma_combat_test.py (py)"]
    class resma2_resma_combat_test_py mod;
    resma2_resma_combat_test_py_combat_test["combat_test"]
    class resma2_resma_combat_test_py_combat_test fn;
    resma2_resma_combat_test_py --> resma2_resma_combat_test_py_combat_test
    train_mini_resma_py["train_mini_resma.py (py)"]
    class train_mini_resma_py mod;
    train_mini_resma_py_main["main"]
    class train_mini_resma_py_main fn;
    train_mini_resma_py --> train_mini_resma_py_main
    resma2_resma_vision_trained_py["resma_vision_trained.py (py)"]
    class resma2_resma_vision_trained_py mod;
    resma2_resma_vision_trained_py_add_noise["add_noise"]
    class resma2_resma_vision_trained_py_add_noise fn;
    resma2_resma_vision_trained_py --> resma2_resma_vision_trained_py_add_noise
    resma2_resma_vision_trained_py_visualize_trained_perception["visualize_trained_perception"]
    class resma2_resma_vision_trained_py_visualize_trained_perception fn;
    resma2_resma_vision_trained_py --> resma2_resma_vision_trained_py_visualize_trained_perception
    resma2_resma_train_py["resma_train.py (py)"]
    class resma2_resma_train_py mod;
    resma2_resma_breakpoint_py["resma_breakpoint.py (py)"]
    class resma2_resma_breakpoint_py mod;
    resma2_resma_breakpoint_py_find_break_point["find_break_point"]
    class resma2_resma_breakpoint_py_find_break_point fn;
    resma2_resma_breakpoint_py --> resma2_resma_breakpoint_py_find_break_point
    resma2_resma_overload_py["resma_overload.py (py)"]
    class resma2_resma_overload_py mod;
    resma2_resma_overload_py_overload_test["overload_test"]
    class resma2_resma_overload_py_overload_test fn;
    resma2_resma_overload_py --> resma2_resma_overload_py_overload_test
    difract_py["difract.py (py)"]
    class difract_py mod;
    difract_py_visualize_uased_geometry["visualize_uased_geometry"]
    class difract_py_visualize_uased_geometry fn;
    difract_py --> difract_py_visualize_uased_geometry
    test_simple_py["test_simple.py (py)"]
    class test_simple_py mod;
    test_simple_py_test_basic_math["test_basic_math"]
    class test_simple_py_test_basic_math fn;
    test_simple_py --> test_simple_py_test_basic_math
    app_py["app.py (py)"]
    class app_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
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

## Architecture Reference

### PY (41 files)

#### `app.py`
**Path:** `app.py`

*No symbols extracted*

#### `demo_mini_resma.py`
**Path:** `demo_mini_resma.py`

**Classes:**
- `GarnierLayer` (line 8) `class GarnierLayer` - *Capa neuronal con temporalidad Garnier T³ (simplificada para demo)*

**Functions:**
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
- `GarnierLayer` (line 9) `class GarnierLayer` - *Capa neuronal con temporalidad Garnier T³*
- `SilencioActivoNetwork` (line 67) `class SilencioActivoNetwork` - *Red neuronal completa con arquitectura RESMA-Garnier*

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 68) `class QuantumLeaf`
- `Universe` (line 101) `class Universe`
- `Network` (line 129) `class Network`
- `MyelinCavity` (line 171) `class MyelinCavity`
- `Bayes` (line 205) `class Bayes`

**Functions:**
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

**Classes:**
- `RC` (line 29) `class RC`
- `Validator` (line 61) `class Validator`
- `QuantumLeaf` (line 82) `class QuantumLeaf`
- `Universe` (line 123) `class Universe`
- `Network` (line 152) `class Network`
- `MyelinCavity` (line 229) `class MyelinCavity`
- `Bayes` (line 268) `class Bayes`

**Functions:**
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

**Classes:**
- `RC` (line 33) `class RC`
- `Validator` (line 50) `class Validator`
- `QuantumLeaf` (line 69) `class QuantumLeaf`
- `Universe` (line 102) `class Universe`
- `Network` (line 130) `class Network`
- `MyelinCavity` (line 198) `class MyelinCavity`
- `Bayes` (line 232) `class Bayes`

**Functions:**
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

**Functions:**
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
- `ModeloGrande` (line 72) `class ModeloGrande` - *Modelo grande diseñado para colapsar con entrenamiento extremo*

**Functions:**
- `setup_matplotlib_for_plotting` (line 19) `def setup_matplotlib_for_plotting()`
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
- `ModeloMNISTPequeno` (line 72) `class ModeloMNISTPequeno` - *Modelo CNN pequeño optimizado para entrenamiento rápido*

**Functions:**
- `setup_matplotlib_for_plotting` (line 19) `def setup_matplotlib_for_plotting()`
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

**Functions:**
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
- `PTSymmetricActivation` (line 14) `class PTSymmetricActivation`
- `E8LatticeLayer` (line 35) `class E8LatticeLayer`
- `RESMABrain` (line 65) `class RESMABrain`

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Classes:**
- `RESMAConstants` (line 38) `class RESMAConstants` - *Constantes físicas RESMA 4.0 con correcciones PT-simétricas*
- `PhysicalValidator` (line 78) `class PhysicalValidator`
- `QuantumLeaf` (line 109) `class QuantumLeaf` - *Hoja L_i como estado KMS con espacio de Hilbert standard*
- `RESMAUniverse` (line 162) `class RESMAUniverse` - *Multiverso como foliación medible, memoria O(N_leaves)*
- `EmunaOperator` (line 224) `class EmunaOperator` - *P̂_E: proyección teleológica no lineal en H²(ℂ⁺)*
- `MyelinCavity` (line 291) `class MyelinCavity` - *Cavidad dieléctrica H = H₀ + iV_loss con Spin(7)*
- `NeuralNetworkRESMA` (line 351) `class NeuralNetworkRESMA` - *Conectoma NO DIRIGIDO con homología persistente*
- `ExperimentalPredictions` (line 474) `class ExperimentalPredictions` - *Predicciones con BF logarítmico*

**Functions:**
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

**Classes:**
- `ResourceMonitor` (line 33) `class ResourceMonitor`
- `RESMAConstants` (line 110) `class RESMAConstants`
- `QuantumLeaf` (line 142) `class QuantumLeaf` - *Hoja KMS - INMUTABLE pero con caché externo*
- `RESMAUniverse` (line 190) `class RESMAUniverse` - *Multiverso con construcción lazy*
- `PhysicalValidator` (line 269) `class PhysicalValidator`
- `MyelinCavity` (line 297) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 352) `class NeuralNetworkRESMA`
- `ExperimentalPredictions` (line 485) `class ExperimentalPredictions`

**Functions:**
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

**Classes:**
- `ResourceMonitor` (line 34) `class ResourceMonitor`
- `RESMAConstants` (line 129) `class RESMAConstants`
- `QuantumLeaf` (line 160) `class QuantumLeaf` - *Hoja KMS - INMUTABLE*
- `RESMAUniverse` (line 203) `class RESMAUniverse` - *Multiverso con estado serializable*
- `PhysicalValidator` (line 316) `class PhysicalValidator`
- `MyelinCavity` (line 343) `class MyelinCavity`
- `NeuralNetworkRESMA` (line 377) `class NeuralNetworkRESMA`

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Functions:**
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

**Functions:**
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
- `CNNMNIST` (line 94) `class CNNMNIST` - *CNN para MNIST con arquitectura diseñada para monitoreo*
- `ExperimentoCompleto` (line 146) `class ExperimentoCompleto` - *Experimento completo para validar el Sovereignty Monitor*

**Functions:**
- `setup_matplotlib_for_plotting` (line 21) `def setup_matplotlib_for_plotting()` - *Setup matplotlib para visualización*
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
