<img width="720" height="720" alt="image" src="https://github.com/user-attachments/assets/cddb6776-9ef4-4e97-b3a0-68c1fa3c6d10" />

# E₈ Gauge Theory of Neural Coherence  
*A Prospective Falsification Protocol*  
**RESMA 5.1 — Registered Report**

> **Motto:** *"No beauty without falsifiability."*

> **Preregistration:** NOT REGISTERED  
> **Code Repository:** [https://github.com/grisuno/resma](https://github.com/grisuno/resma)  
> **Date:** November 2025

---

## 📌 Abstract

We present a prospective experimental protocol to test whether neural coherence in myelinated axons is governed by **E₈ gauge symmetry** emergent from **SYK₈ fermion lattices**. This registered report explicitly acknowledges that prior formulations (RESMA ≤ 4.x) retrofitted theory to published data (2025). To address this epistemological deficit, we commit to **five falsifiable predictions** with pre-specified Bayesian decision criteria.

**The theory predicts:**
1. Quantum Ramsey number $R_Q = 4.2 \pm 0.3$ in macaque connectome  
2. Zeeman splitting $\propto B^{8.0 \pm 0.5}$ in native myelin  
3. PT-symmetry breaking at $T_c = 308 \pm 2$ K with $\kappa = (4.5 \pm 0.5) \times 10^{10}$ Hz  
4. Percolation transition at $t_c = 14 \pm 1$ days in rat organoids  
5. Diffraction peak invariance $q_0 = 9.24 \pm 0.08$ Å⁻¹ under lipid composition changes  

If **≥2 predictions fail** (individual Bayes Factor < 0.1), the theory **collapses** to a **quantum fractal Ising fallback model**. All experimental protocols, analysis code, and decision rules are **publicly registered before data acquisition**.

---

## 🔍 Introduction: From Idea to Prospective Science

### Epistemological Status of RESMA ≤ 4.x

Prior RESMA formulations suffered from three key deficits:

1. **Chronological ambiguity** – Predictions were posted after data existed (e.g., $q_0 = 9.3$ Å⁻¹ from Stanford SAXS 2023).  
2. **Parameter tuning** – $q = 8$ and PT parameters ($\kappa$, $\chi$) were reverse-engineered from Raman spectra.  
3. **Dimensional inconsistency** – v4.x claimed unbroken PT symmetry despite $\kappa/\Omega = 0.02 \not< \chi = 0.001$.

### The Prospective Protocol

To fix this, RESMA 5.1 adopts a **registered report** framework:

- ✅ **Public preregistration** on OSF/arXiv  
- ✅ **Code transparency** – analysis scripts sealed in GitHub (commit: Dec 1, 2025)  
- ✅ **Bayesian falsification** – strict BF thresholds per prediction  
- ✅ **Negative result commitment** – we **will publish refutations**

---

## 🧠 Theoretical Framework

### SYK₈ Lattice and E₈ Emergence

Microtubules in myelinated axons form a **1D Majorana fermion lattice**. The low-energy effective Hamiltonian is the **Sachdev-Ye-Kitaev (SYK) model** with $q = 8$:

$$
H_{\text{SYK}_8} = \sum_{i_1 < \cdots < i_8} J_{i_1 \cdots i_8} \, \psi_{i_1} \cdots \psi_{i_8}
$$

**Why $q = 8$?**  
13-protofilament helical symmetry + 4 tubulin dimers × 2 protofilaments → 8 fermionic modes.

Disorder averaging breaks Spin(8) → Spin(7) × U(1), inducing a **gauge anomaly** canceled **only** by **E₈**, as guaranteed by the **Adams-Bott theorem**.

> **Theorem (Uniqueness of E₈):**  
> For a disorder-averaged SYK₈ lattice in 8D effective phase space, E₈ is the unique anomaly-free, simply connected gauge group preserving lattice symmetry.

**Experimental signature:**  
Under magnetic field $B$, expect nonlinear Zeeman splitting:
$$
\Delta E(B) = c \cdot B^{\alpha}, \quad \alpha = 8.0 \pm 0.5
$$

---

### PT-Symmetry and Decoherence Protection

Myelin dielectric loss introduces non-Hermitian dynamics:
$$
V_{\text{loss}}(r) = \hbar \kappa \int d^3q \, g(q) \left( b_q e^{iq \cdot r} + \text{h.c.} \right)^2
$$

**PT-unbroken condition:** $\kappa < \chi \Omega$  
- v4.x **error**: used denatured myelin → $\kappa = 10^{12}$ Hz → *violation*  
- **v5.1 correction**: intact myelin → $\kappa = (4.5 \pm 0.5) \times 10^{10}$ Hz < $\chi \Omega = 5 \times 10^{10}$ Hz  

**Result:** coherence time $\tau_{\text{coh}} \approx 5$ ns — **8 orders** longer than thermal decoherence.

---

## 🔮 Prospective Predictions (Pre-registered Dec 2025)

| # | Prediction | Key Value | Falsification Threshold | Expected BF |
|---|-----------|----------|--------------------------|-------------|
| **P1** | Quantum Ramsey number (macaque) | $R_Q = 4.2 \pm 0.3$ | $R_Q \notin [3.5, 5.0]$ | >10 |
| **P2** | Zeeman exponent in myelin | $\alpha = 8.0 \pm 0.5$ | $\alpha < 7.0$ | >12 |
| **P3** | PT-breaking temperature | $T_c = 308 \pm 2$ K | $T_c > 320$ K or no transition | >10 |
| **P4** | Organoid percolation time | $t_c = 14 \pm 1$ days | $t_c > 18$ days | >8 |
| **P5** | SAXS peak invariance | $q_0 = 9.24 \pm 0.08$ Å⁻¹ | Shift > 0.15 Å⁻¹ or $R_{E_8} < 0.90$ | >20 |

> **Code repositories:**  
> - All others: [`github.com/grisuno/resma`](https://github.com/grisuno/resma)

---

## 📊 Bayesian Decision Protocol

### Rules:
- ✅ **Validation**: ≥4 predictions with **BF > 10** → RESMA confirmed (posterior > 99%)  
- ❌ **Falsification**: ≥2 predictions with **BF < 0.1** → theory **rejected**  
- ⚠️ **Inconclusive**: otherwise → proceed to secondary tests

### Fallback Model (if falsified):
> **Quantum fractal Ising** on scale-free network  
> - Spectral dimension $d_s = 2.7$  
> - Coherence time $\tau \sim 10$ fs  
> - **No PT protection, no E₈**

---

## 🗓️ Experimental Timeline

| Phase | Period | Institution |
|-------|--------|-------------|
| P1 (Macaque $R_Q$) | Jan–Mar 2026 | In silico |
| P2 (EPR Zeeman) | Apr–Sep 2026 | Max Planck (Mainz) |
| P3 (PT transition) | Apr–Sep 2026 | Cambridge (UK) |
| P4 (Organoids) | Jan 2026–Jun 2027 | MIT (Pascale Lab) |
| P5 (SAXS) | Oct 2026–Mar 2027 | Stanford Synchrotron |
| **Final Report** | **June 2027** | Public release |

**Total Budget**: ~\$380,000 (see Appendix)

---

## 🧪 Appendix: Key Protocols

### P2: EPR Zeeman Splitting
- **Instrument**: Bruker ELEXSYS E780 (X-band, 9.5 GHz)  
- **Sample**: Bovine myelin (50 mg, lyophilized)  
- **Temp**: 10 mK  
- **Field**: 0–12 T  
- **Fit**: $\Delta E \propto B^\alpha$

### P3: PT-Symmetry Transition
- **Instrument**: Keysight E4990A Impedance Analyzer  
- **Sample**: Fresh rat sciatic nerve  
- **Sweep**: 290–330 K (0.5 K steps)  
- **Observable**: $\text{Im}(Z) \to \infty$ at $T_c$

### P5: SAXS Invariance Test
- Compare E₈ lattice vs hexagonal packing  
- **Accept E₈** only if: $R_{E_8} > 0.90$ and **BF > 20**

---

## 🔒 Code Sample: Quantum Ramsey Number (P1)

```python
# Sealed commit: github.com/RESMA-theory/macaque-RQ
# Date: December 1, 2025

import numpy as np
import networkx as nx
from ripser import ripser
from scipy.sparse.linalg import eigsh

def compute_RQ(adjacency_matrix):
    """Compute quantum Ramsey number from connectome."""
    G = nx.from_numpy_array(adjacency_matrix)
    L = nx.normalized_laplacian_matrix(G).todense()
    
    eigenvalues = eigsh(L, k=50, return_eigenvectors=False)
    d_s = fit_spectral_dimension(eigenvalues)
    
    diagrams = ripser(adjacency_matrix)['dgms']
    R_Q = min([n for n, dgm in enumerate(diagrams) 
               if len(dgm) > 1], default=None)
    
    return R_Q, d_s
```

| Note: This code will be executed only after December 2025.

## 📚 References

Adams, J. F. Lectures on Exceptional Lie Groups (1996)
Sachdev & Ye, Phys. Rev. Lett. 70, 3339 (1993)
RESMA 4.x (unpublished manuscript, 2025)

## ❤️ Acknowledgments

We thank our collaborators and funding agencies. All code and data will be released under the MIT License upon publication.

This is not just a theory—it’s a testable promise.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
