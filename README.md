# Stochastic closures for turbulence in the Sabra shell model

This repository contains the code used in the paper:

> **On the importance of stochasticity in closures of turbulence**.

It provides three complementary components:

1. **Landau–Lifshitz fluctuating-hydrodynamics reference solver** (stochastic DNS-like shell-model simulations).
2. **Data-driven stochastic closure (neural Langevin closure)** for LES.
3. **Phenomenological stochastic closure (K41 / multiplier-based)** for LES.

The main purpose is to compare finite-time uncertainty growth (ensemble variance propagation) between fully resolved stochastic dynamics and reduced-order closures.

---

## Repository structure

```text
.
├── solver_landau_lifshitz_ensemble.py
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

### Script roles

- `solver_landau_lifshitz_ensemble.py`  
  Generates ensemble trajectories for a **single initial condition** with independent stochastic forcing in the Sabra–Landau–Lifshitz model.

- `nn_closure/train_langevin_closure.py`  
  Trains the **neural Langevin closure** (solver-in-the-loop / a posteriori training).

- `nn_closure/run_langevin_rollout.py`  
  Runs long rollout inference using trained neural closure models and saves trajectory snapshots.

- `nn_closure/run_langevin_ensemble_variance.py`  
  Runs IC × ensemble inference and saves ensemble statistics (variance, and optionally mean).

- `k41_closure/run_k41_ensemble.py`  
  Runs the **phenomenological stochastic K41 (multiplier-based) closure** for one IC and many ensemble members.

---

## Installation

Use a Python environment with the dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Note: `solver_landau_lifshitz_ensemble.py` and `k41_closure/run_k41_ensemble.py` rely on **JAX**; the neural closure scripts rely on **TensorFlow/Keras**.

---

## Typical workflow

### 1) Generate fluctuating-hydrodynamics reference trajectories

```bash
python solver_landau_lifshitz_ensemble.py --ic-index 0 --n-ens 1024
```

This produces an `.npz` file with complex shell trajectories for the chosen initial condition and ensemble size.

### 2) Train the neural Langevin closure

```bash
python nn_closure/train_langevin_closure.py
```

The training script expects a pre-generated dataset (default filename in the script is `u_40_2.npz`) and writes `.keras` model checkpoints.

### 3) Run neural closure inference

Trajectory rollout:

```bash
python nn_closure/run_langevin_rollout.py
```

Variance-focused ensemble run:

```bash
python nn_closure/run_langevin_ensemble_variance.py --noise-steps 1000000
```

### 4) Run the phenomenological K41 stochastic closure

```bash
python k41_closure/run_k41_ensemble.py --ic-index 0 --n-ens 1024
```

---

## Naming cleanup

For clarity and reproducibility, the main scripts were renamed from earlier internal names:

- `solver_save_linear_repeated_ic.py` → `solver_landau_lifshitz_ensemble.py`
- `nn_closure/train_stochastic.py` → `nn_closure/train_langevin_closure.py`
- `nn_closure/inf_v2.py` → `nn_closure/run_langevin_rollout.py`
- `nn_closure/inf_v2_many_same_ics.py` → `nn_closure/run_langevin_ensemble_variance.py`
- `k41_closure/smk_save_linear_repeated_ic.py` → `k41_closure/run_k41_ensemble.py`

---

## Citation

If you use this repository, please cite:

- Freitas et al., *On the importance of stochasticity in closures of turbulence*.

(You can replace this entry with the final journal reference once available.)

---

## License

MIT License (see `LICENSE`).
