# Stochastic Sabra model with thermal noise (Landau–Lifshitz style).
# LINEAR save schedule with a user-chosen stride.
# Runs ONE initial condition (chosen index) with MANY ensemble members,
# each with a different realisation of the noise.

import jax
import jax.numpy as jnp
from jax import random, lax
from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import time
from tqdm import tqdm
import sys
from pathlib import Path
import os
import argparse

# ----------------------------- argparse -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ic-index", type=int, default=0, help="IC index in ic.npz to use")
parser.add_argument("--n-ens", type=int, default=256, help="Number of ensemble members")
parser.add_argument("--out", type=str, default=None, help="Optional output path")
args = parser.parse_args()

ic_index = args.ic_index
n_ens = args.n_ens

# ----------------------------- knobs -----------------------------
save_all   = False      # if True, save all N shells (sets Ncut=N)
include_t0 = True     # set True to also store t=0 snapshot

saved_steps_total = 300_000_000   # total number of time steps to advance
save_stride       = 100_000       # LINEAR sampling stride (in steps)

# Noise parameters (choose path A or B)
use_theta_eta = True     # A: compute Theta from theta_eta, Re, alpha; B: set Theta_direct below
alpha         = 0        # 0 or 3 as in the paper
Re            = 10**12   # nominal Re for the Theta scaling
theta_eta     = 2.83e-8  # ABL value used in the paper
Theta_direct  = 0.0      # used only if use_theta_eta = False

# RNG for noise
noise_seed = 98765

# Fixed parameters
N    = 40                 # number of shells
Ncut = 15                 # we'll only SAVE the first Ncut shells unless save_all=True
nu   = 1.0 / Re           # viscosity
dt   = 1e-8               # integration step
a, b, c = (1.0, -0.50, -0.50)
eps0 = 0.5 / (2**0.5)
eps1 = 0.7 * eps0

# ---------------------- compute Θ (noise prefactor) ----------------------
if use_theta_eta:
    beta = 3.0 * (alpha + 2.0) / 4.0
    Theta = (Re ** (-beta)) * theta_eta
else:
    Theta = Theta_direct

# ---------------------- wavenumbers, IC, forcing ------------------------
k_1d = jnp.array([2**n for n in range(N)], dtype=jnp.float64)

# Load ICs: aux shape (N, n_ics_in_file)
aux_all = np.load("ic.npz")["aux"]
if aux_all.ndim != 2 or aux_all.shape[0] != N:
    raise ValueError(f"Expected ic.npz['aux'] shape (N, n_ics), got {aux_all.shape}")

n_ics_in_file = aux_all.shape[1]
if ic_index < 0 or ic_index >= n_ics_in_file:
    raise ValueError(f"--ic-index {ic_index} out of range (n_ics={n_ics_in_file})")

# Take one IC and tile across ensemble members: (N, n_ens)
aux0 = jnp.asarray(aux_all[:, ic_index], dtype=jnp.complex128)[:, None]  # (N,1)
aux = jnp.tile(aux0, (1, n_ens))                                         # (N,n_ens)

batch_size = n_ens
print(f"Running ONE IC (index {ic_index}) with n_ens={n_ens}")
print("aux shape:", aux.shape)

# tile k, ek, forcing for this batch
k = jnp.tile(k_1d[:, None], (1, batch_size)).astype(jnp.complex128)  # (N, batch)
ek_1d = jnp.exp(-nu * dt * (k_1d**2) / 2.0).astype(jnp.complex128)
ek = jnp.tile(ek_1d[:, None], (1, batch_size))

forcing_vec = []
for n in range(N):
    if n == 0:
        forcing_vec.append(eps0 + eps0*1j)
    elif n == 1:
        forcing_vec.append(eps1 + eps1*1j)
    else:
        forcing_vec.append(0.0 + 0.0j)
forcing = jnp.array(forcing_vec, dtype=jnp.complex128)
forcing = jnp.tile(forcing[:, None], (1, batch_size))

# ------------------- stochastic noise scale per step ---------------------
kpow = (k_1d ** (1.0 + 0.5*alpha))[:, None]                        # (N,1) float64
sigma_step = (jnp.sqrt(Theta * dt) * kpow).astype(jnp.complex128)  # (N,1) broadcasts over batch

# ----------------------------- RHS & integrator --------------------------
@jax.jit
def G(u):
    coupling = jnp.expand_dims(((a * k[1, :] * jnp.conj(u[1, :]) * u[2, :]) * 1j), axis=0)
    coupling = jnp.concatenate(
        [coupling,
         jnp.expand_dims(((a * k[2, :] * jnp.conj(u[2, :]) * u[3, :]
                           + b * k[1, :] * jnp.conj(u[0, :]) * u[2, :]) * 1j), axis=0)],
        axis=0
    )
    for n in range(2, N-2):
        coupling = jnp.concatenate(
            [coupling,
             jnp.expand_dims(((a * k[n + 1, :] * jnp.conj(u[n + 1, :]) * u[n + 2, :]
                               + b * k[n, :] * jnp.conj(u[n - 1, :]) * u[n + 1, :]
                               - c * k[n - 1, :] * u[n - 1, :] * u[n - 2, :]) * 1j), axis=0)],
            axis=0
        )
    coupling = jnp.concatenate(
        [coupling,
         jnp.expand_dims(((b * k[N-2, :] * jnp.conj(u[N-3, :]) * u[N-1, :]
                           - c * k[N-3, :] * u[N-3, :] * u[N-4, :]) * 1j), axis=0)],
        axis=0
    )
    coupling = jnp.concatenate(
        [coupling,
         jnp.expand_dims(((-c * k[N-2, :] * u[N-2, :] * u[N-3, :]) * 1j), axis=0)],
        axis=0
    )
    return coupling

@jax.jit
def RK4_det(u):
    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1/2)))
    A3 = dt * (forcing + G(ek * u + A2/2))
    A4 = dt * (forcing + G(u*(ek**2) + ek*A3))
    return (ek**2)*(u + A1/6) + ek*(A2 + A3)/3 + A4/6

def _one_step(carry, _):
    aux_, key = carry
    aux_ = RK4_det(aux_)
    key, k_noise = random.split(key)
    z = random.normal(k_noise, aux_.shape + (2,), dtype=jnp.float64)  # (N, n_ens, 2)
    dW = (z[..., 0] + 1j * z[..., 1])                                # independent across (shell,ens)
    aux_ = aux_ + sigma_step * dW
    return (aux_, key), None

def advance_n(aux_, key, n_steps: int):
    (aux_, key), _ = lax.scan(_one_step, (aux_, key), xs=None, length=n_steps)
    return aux_, key

advance_n = jax.jit(advance_n, static_argnames=("n_steps",))

# ------------------------- linear save schedule --------------------------
if save_stride < 1:
    raise ValueError("save_stride must be >= 1")

if include_t0:
    save_steps = np.arange(0, saved_steps_total + 1, save_stride, dtype=int)
else:
    save_steps = np.arange(save_stride, saved_steps_total + 1, save_stride, dtype=int)

n_saves = len(save_steps)
print("Save steps (head):", save_steps[:10], "... total:", n_saves)

# ---------------------- allocate & initialize ---------------------------
if save_all:
    Ncut = N

u_save = jnp.zeros((Ncut, n_ens, n_saves), dtype=jnp.complex128)
key_noise = random.PRNGKey(noise_seed)

save_col = 0
prev_step = 0
if include_t0:
    u_save = u_save.at[:, :, save_col].set(aux[:Ncut, :])
    save_col += 1
    prev_step = 0

# ---------------------- main integration loop --------------------------
start_time = time.time()
for target in tqdm(save_steps[save_col:], desc="Advancing to save points"):
    n_to_advance = int(target - prev_step)
    if n_to_advance > 0:
        aux, key_noise = advance_n(aux, key_noise, n_to_advance)
        prev_step = target

    if not bool(jnp.all(jnp.isfinite(aux))):
        bad = np.argwhere(~np.isfinite(np.array(aux)))
        print(f"NaN/Inf detected at step {target}, first few indices: {bad[:5]}")
        sys.exit(1)

    u_save = u_save.at[:, :, save_col].set(aux[:Ncut, :])
    save_col += 1

end_time = time.time()
print("\nTime:", end_time - start_time, "s")
print("u_save shape:", u_save.shape)

# ----------------------------- save output -----------------------------
Path("npzs").mkdir(parents=True, exist_ok=True)

meta_dict = dict(
    ic_index=int(ic_index),
    n_ens=int(n_ens),
    alpha=int(alpha),
    Re=float(Re),
    use_theta_eta=bool(use_theta_eta),
    theta_eta=float(theta_eta),
    Theta=float(Theta),
    nu=float(nu),
    dt=float(dt),
    N=int(N),
    Ncut=int(Ncut),
    saved_steps_total=int(saved_steps_total),
    save_stride=int(save_stride),
    save_steps=[int(s) for s in save_steps],
    include_t0=bool(include_t0),
    noise_seed=int(noise_seed),
)

suffix = f"ic{ic_index}_ens{n_ens}_alpha{alpha}_ncut{Ncut}"
tmp_path = f"npzs/_tmp_u_gt_{suffix}.npz"
final_path = args.out if args.out is not None else f"npzs/u_gt_{suffix}.npz"

np.savez_compressed(tmp_path, u=np.array(u_save), meta=meta_dict)
os.replace(tmp_path, final_path)
print(f"[done] wrote {Path(final_path).resolve()}")

