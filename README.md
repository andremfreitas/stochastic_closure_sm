# Stochastic closures for turbulence in the Sabra shell model

Code accompanying the paper

- [On the importance of stochasticity in closures of turbulence](https://iopscience.iop.org/article/10.1209/0295-5075/ae5a56)  
  *Freitas et al., Europhysics Letters (2026)*

---

## Description

This repository contains scripts to generate training data and to train stochastic neural-network closures for Large-Eddy Simulations (LES) of the Sabra shell model.

For full methodological and theoretical details, please refer to the papers (see at the bottom of this page).

---

## Repository structure

```text
.
├── solvers/
│   ├── solver_deterministic.py
│   ├── solver_landau_lifshitz_ensemble.py
├── nn_closure/
│   ├── train_langevin_closure.py
│   ├── run_langevin_rollout.py
│   └── run_langevin_ensemble_variance.py
├── k41_closure/
│   └── run_k41_ensemble.py
├── requirements.txt
├── README.md
└── LICENSE
```

- **solvers/**: deterministic Sabra solver (data generation) and stochastic Landau–Lifshitz ensemble solver  
- **nn_closure/**: training and inference scripts for the neural Langevin closure  
- **k41_closure/**: phenomenological stochastic K41/LES-KOL closure  

---

## How to run

A minimal workflow is:

1. Generate data
   ```bash
   python solvers/solver_deterministic.py
   ```
   This produces `dataset.npz`.

2. Train neural Langevin closure
   ```bash
   python nn_closure/train_langevin_closure.py
   ```

3. Run inference
   ```bash
   python nn_closure/run_langevin_rollout.py
   ```

4. Run ensemble statistics
   ```bash
   python nn_closure/run_langevin_ensemble_variance.py
   ```

5. Run reference stochastic models
   ```bash
   python solvers/solver_landau_lifshitz_ensemble.py
   python k41_closure/run_k41_ensemble.py
   ```

The scripts are mostly self-contained; parameters are set directly in each file or via simple CLI flags.

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Note**
- `solver_landau_lifshitz_ensemble.py` and `k41_closure/run_k41_ensemble.py` use JAX  
- `nn_closure/*` uses TensorFlow/Keras  

---

## Citation

If you use this repository, please cite:

```bibtex
@article{freitas26,
  author={Freitas, André and Biferale, Luca and Desbrun, Mathieu and Eyink, Gregory and Mailybaev, Alexei and Um, Kiwon},
  title={On the importance of stochasticity in closures of turbulence},
  journal={Europhysics Letters},
  url={http://iopscience.iop.org/article/10.1209/0295-5075/ae5a56},
  year={2026},
}
```

```bibtex
@article{freitas25,
  title = {Solver-in-the-loop approach to closure of shell models of turbulence},
  author = {Freitas, André and Um, Kiwon and Desbrun, Mathieu and Buzzicotti, Michele and Biferale, Luca},
  journal = {Phys. Rev. Fluids},
  volume = {10},
  issue = {4},
  pages = {044602},
  year = {2025},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevFluids.10.044602}
}
```

---

## License

MIT License (see `LICENSE`).