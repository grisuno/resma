# Architecture

## Internal Dependencies

- `resma2/main_experiment.py` -> `resma2/resma_core.py`
- `resma2/main_experiment.py` -> `resma2/resma_observer.py`
- `resma2/main_experiments.py` -> `resma2/monitor.py`
- `resma2/main_experiments.py` -> `resma2/resma_core.py`
- `resma2/main_experiments.py` -> `resma2/resma_observer.py`
- `resma2/resma_app_mnist.py` -> `resma2/resma_core.py`
- `resma2/resma_app_mnist.py` -> `resma2/resma_observer.py`
- `resma2/resma_breakpoint.py` -> `resma2/resma_core.py`
- `resma2/resma_combat_test.py` -> `resma2/resma_core.py`
- `resma2/resma_noise_phase_test.py` -> `resma2/resma_core.py`
- `resma2/resma_observer.py` -> `resma2/monitor.py`
- `resma2/resma_overload.py` -> `resma2/resma_core.py`
- `resma2/resma_train.py` -> `resma2/resma_core.py`
- `resma2/resma_vision.py` -> `resma2/resma_core.py`
- `resma2/resma_vision_trained.py` -> `resma2/resma_core.py`
- `test_ultra_simple.py` -> `garnier_nn.py`
- `train_mini_resma.py` -> `garnier_nn.py`
- `train_profile.py` -> `garnier_nn.py`

## External Imports

- `app.py` -> os
- `demo_mini_resma.py` -> logging, networkx, numpy, torch, torch.nn, typing
- `difract.py` -> matplotlib.pyplot, numpy
- `garnier_nn.py` -> logging, networkx, numpy, time, torch, torch.nn, typing
- `main.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.interpolate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing
- `main2.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.interpolate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing
- `main3.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing
- `main4.1.py` -> dataclasses, logging, networkx, numpy, ripser, scipy.integrate, scipy.linalg, scipy.sparse.linalg, typing, warnings
- `main4.py.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing
- `main5.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.interpolate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing, warnings
- `monitor_extremo.py` -> matplotlib.pyplot, numpy, time, torch, torch.nn, torch.nn.functional, torch.optim, traceback, typing, warnings
- `quick_monitor.py` -> matplotlib.pyplot, numpy, time, torch, torch.nn, torch.nn.functional, torch.optim, traceback, typing, warnings
- `resma2/main_experiment.py` -> numpy, random, torch, torch.nn
- `resma2/main_experiments.py` -> torch, torch.nn, torch.optim, torch.utils.data, torchvision
- `resma2/monitor.py` -> dataclasses, enum, numpy, torch, typing, warnings
- `resma2/resma_app_mnist.py` -> numpy, torch, torch.nn, torch.optim, torch.utils.data, torchvision
- `resma2/resma_breakpoint.py` -> matplotlib.pyplot, numpy, torch
- `resma2/resma_combat_test.py` -> matplotlib.pyplot, numpy, torch, torch.utils.data, torchvision
- `resma2/resma_core.py` -> networkx, numpy, torch, torch.nn, torch.nn.functional, typing
- `resma2/resma_noise_phase_test.py` -> matplotlib.pyplot, numpy, torch, torch.nn.functional, torchvision
- `resma2/resma_observer.py` -> dataclasses, json, matplotlib.pyplot, numpy, torch, typing
- `resma2/resma_overload.py` -> matplotlib.pyplot, numpy, torch
- `resma2/resma_train.py` -> torch, torch.nn, torch.utils.data, torchvision
- `resma2/resma_vision.py` -> matplotlib.pyplot, numpy, torch, torch.nn, torchvision
- `resma2/resma_vision_trained.py` -> matplotlib.pyplot, numpy, torch, torchvision
- `resma4.10.py` -> dataclasses, datetime, gc, itertools, logging, networkx, numpy, os, pathlib, pickle, psutil, scipy.integrate, scipy.linalg, scipy.sparse.linalg, time, typing, warnings, weakref
- `resma4.13.py` -> dataclasses, datetime, gc, itertools, logging, networkx, numpy, os, pathlib, pickle, psutil, scipy.integrate, scipy.linalg, scipy.sparse.linalg, time, typing, warnings, weakref
- `resma4.2.py` -> dataclasses, logging, networkx, numpy, pint, psutil, ripser, scipy.integrate, scipy.interpolate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, typing, warnings
- `resma4.3.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pickle, pint, psutil, ripser, scipy.integrate, scipy.linalg, scipy.sparse, time, typing, warnings, weakref
- `resma4.4.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pickle, pint, psutil, ripser, scipy.integrate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, time, typing, warnings, weakref
- `resma4.5.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pathlib, pickle, pint, psutil, scipy.integrate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, time, typing, warnings, weakref
- `resma4.6.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pathlib, pickle, pint, psutil, scipy.integrate, scipy.linalg, time, typing, warnings, weakref
- `resma4.7.py` -> dataclasses, gc, logging, networkx, numpy, scipy.integrate, scipy.linalg, scipy.sparse.linalg, typing, warnings
- `resma4.8.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pathlib, pickle, psutil, scipy.integrate, scipy.linalg, scipy.sparse, scipy.sparse.linalg, time, typing, warnings, weakref
- `resma4.9.py` -> dataclasses, datetime, gc, logging, networkx, numpy, os, pathlib, pickle, psutil, scipy.integrate, scipy.linalg, scipy.sparse.linalg, typing, warnings, weakref
- `sovereignty_monitor.py` -> matplotlib.pyplot, numpy, os, torch, torch.nn, torch.nn.functional, torch.optim, torchvision, traceback, typing, warnings
- `test_simple.py` -> numpy, torch
- `test_ultra_simple.py` -> logging, networkx, numpy, time, torch, torch.nn, typing
- `train_mini_resma.py` -> logging, os, torch, torch.utils.data, torchvision
- `train_profile.py` -> logging, os, time, torch, torch.utils.data, torchvision
- `visualize_resma.py` -> matplotlib.pyplot, networkx, numpy, seaborn, torch, warnings
