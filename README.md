# Solver-in-the-Loop Closure for Shell Models of Turbulence

Code accompanying the papers

**Solver-in-the-loop approach to closure of shell models of turbulence**  
*Phys. Rev. Fluids (2025)*

**A posteriori closure of turbulence models: are symmetries preserved?**  
*European Journal of Mechanics / B Fluids (2026)*


---

## Description

This repository contains the code used to train and evaluate neural network closures for the **Sabra shell model of turbulence** using a **solver-in-the-loop (a posteriori) training strategy**.

Instead of learning instantaneous subgrid terms independently of the dynamics, the neural network is embedded inside the numerical solver during training. The model therefore learns how its predictions influence the time evolution of the system.

The numerical experiments presented in the two papers above were performed using the scripts contained in this repository.

For full methodological and theoretical details, please refer to the papers.


---

## Repository structure

```
solver.py          Generate DNS data from the fully resolved shell model
train.py           Train the neural closure using solver-in-the-loop learning
inf.py             Run inference with a trained model

requirements.txt   Python dependencies
README.md          Documentation
LICENSE            Repository license
```


---

## Installation

Create a Python environment and install the required dependencies:

```
pip install -r requirements.txt
```


---

## 1. Generate training data

Run the shell-model solver to generate DNS trajectories:

```
python solver.py
```

The output dataset contains shell velocities with the structure

```
u[shell, initial_condition, time]
```

These trajectories serve as ground-truth data for training the closure model.


---

## 2. Train the neural closure

Training is performed by embedding the neural network inside the reduced solver and unrolling the dynamics for several time steps before computing the loss.

```
python train.py path_to_dataset.npz
```

The trained model is saved as a `.keras` file.


---

## 3. Run inference

Once a model has been trained, long-time trajectories of the reduced system can be generated with

```
python inf.py
```

The script loads the trained model, evolves the reduced system forward in time, and saves the predicted trajectories.


---

## Numerical parameters

Typical parameters used in the experiments:

```
Number of shells (DNS): 40
Cutoff shell:           14
Viscosity:              1e-12
DNS timestep:           1e-8
LES timestep:           1e-5
```


---

## Citation

If you use this code in your research, please cite the following works.

### Solver-in-the-loop closure

```
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

### Symmetry preservation in a posteriori closures

```
@article{freitas26,
title = {A posteriori closure of turbulence models: Are symmetries preserved?},
journal = {European Journal of Mechanics - B/Fluids},
volume = {119},
pages = {204496},
year = {2026},
issn = {0997-7546},
doi = {https://doi.org/10.1016/j.euromechflu.2026.204496},
url = {https://www.sciencedirect.com/science/article/pii/S0997754626000439},
author = {André Freitas and Kiwon Um and Mathieu Desbrun and Michele Buzzicotti and Luca Biferale},
```


---

## License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.
