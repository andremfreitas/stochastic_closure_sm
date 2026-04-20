"""
Run the stochastic K41/SMK closure for one IC and many ensemble members.

The script loads one initial condition from a reference dataset, evolves the
resolved shells with the stochastic closure, and saves linearly sampled
trajectories.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser(
    description="Stochastic K41/SMK closure with linear saves for one IC and many ensembles."
)
parser.add_argument("--ic-index", type=int, default=0, help="IC index in the reference dataset")
parser.add_argument("--n-ens", type=int, default=256, help="Number of ensemble members")
parser.add_argument("--t0", type=int, default=0, help="Time index for the initial condition")
parser.add_argument("--out", type=str, default="npzs/k41_ensemble_one_ic.npz", help="Output npz path")
args = parser.parse_args()

# -----------------------
# User settings
# -----------------------
GT_PATH = "../u_40_2.npz"
OUT_PATH = args.out

N_les = 15
N_state = N_les + 2

ic_index = int(args.ic_index)
n_ic = 1
n_ens = int(args.n_ens)
t0 = int(args.t0)

lam = 2.0
a, b, c = 1.0, -0.5, -0.5
nu = 1e-12
dt = 1e-5
eps0 = 0.5 / (2.0 ** 0.5)
eps1 = 0.25

transient_T = 0.0
sim_T = 5.0
transient_steps = int(round(transient_T / dt))
sim_steps = int(round(sim_T / dt))

include_t0 = True
save_stride = 100

sigma_x = 0.5
sigma_eta = 0.5
seed = 12345

same_ou0_within_ic = True

# -----------------------
# Linear save schedule
# -----------------------
if save_stride < 1:
    raise ValueError("save_stride must be >= 1")

if include_t0:
    save_steps = np.arange(0, sim_steps + 1, save_stride, dtype=int)
else:
    save_steps = np.arange(save_stride, sim_steps + 1, save_stride, dtype=int)

if len(save_steps) == 0 or save_steps[-1] != sim_steps:
    save_steps = np.unique(np.append(save_steps, sim_steps)).astype(int)

n_saves = len(save_steps)
print("Save steps (head):", save_steps[:15], "... total:", n_saves)

# -----------------------
# Load IC and build batch
# -----------------------
data = np.load(GT_PATH)
u_dns = data["u"]

if u_dns.shape[0] < N_les:
    raise ValueError(f"Need at least {N_les} shells, got {u_dns.shape[0]}")
if t0 < 0 or t0 >= u_dns.shape[2]:
    raise ValueError(f"t0={t0} out of bounds (time axis length {u_dns.shape[2]})")
if ic_index < 0 or ic_index >= u_dns.shape[1]:
    raise ValueError(f"ic_index={ic_index} out of bounds (n_ics_in_file={u_dns.shape[1]})")

u0_res = u_dns[:N_les, ic_index, t0].astype(np.complex128)
u0_flat = np.repeat(u0_res[:, None], repeats=n_ens, axis=1)

batch = n_ens
u0 = np.zeros((N_state, batch), dtype=np.complex128)
u0[:N_les, :] = u0_flat

# -----------------------
# Fixed vectors
# -----------------------
k1d = (lam ** np.arange(N_state)).astype(np.float64)
k = jnp.asarray(np.tile(k1d[None, :], (batch, 1)).T, dtype=jnp.complex128)

ek1d = np.exp(-nu * dt * (k1d ** 2) / 2.0).astype(np.float64)
ek = jnp.asarray(np.tile(ek1d[None, :], (batch, 1)).T, dtype=jnp.complex128)

forcing1d = np.zeros((N_state,), dtype=np.complex128)
forcing1d[0] = eps0 + 1j * eps0
forcing1d[1] = eps1 + 1j * eps1
forcing = jnp.asarray(np.tile(forcing1d[None, :], (batch, 1)).T, dtype=jnp.complex128)

# -----------------------
# OU timescales
# -----------------------
tau_n = np.array([
    1.32614011e+00, 8.69406980e-01, 5.50297141e-01, 3.50757779e-01,
    2.32389103e-01, 1.53029338e-01, 9.97293854e-02, 6.49296182e-02,
    4.33559623e-02, 2.82462335e-02, 1.85303692e-02, 1.21645417e-02,
    7.93147293e-03, 5.24364222e-03, 3.44624715e-03, 2.25742063e-03,
    1.48881111e-03, 9.77027276e-04, 6.39891668e-04, 4.22310751e-04,
    2.77499574e-04, 1.82726037e-04, 1.21530982e-04, 8.00729289e-05,
    5.33820111e-05, 3.58833650e-05, 2.40917980e-05, 1.67558359e-05,
    1.23504498e-05, 1.06849673e-05, 1.27659532e-05, 2.33765212e-05,
])

tau_sp1 = float(tau_n[N_les])
tau_sp2 = float(tau_n[N_les + 1])

mu_eta = -(sigma_eta ** 2) / 4.0
z0 = lam ** (-1.0 / 3.0)

# -----------------------
# Resolved Sabra nonlinearity
# -----------------------
@jax.jit
def G_les(u):
    out = jnp.zeros_like(u)

    out = out.at[0, :].set(1j * (a * k[1, :] * jnp.conj(u[1, :]) * u[2, :]))
    out = out.at[1, :].set(1j * (
        a * k[2, :] * jnp.conj(u[2, :]) * u[3, :]
        + b * k[1, :] * jnp.conj(u[0, :]) * u[2, :]
    ))

    def body(n, out_):
        term = 1j * (
            a * k[n + 1, :] * jnp.conj(u[n + 1, :]) * u[n + 2, :]
            + b * k[n, :] * jnp.conj(u[n - 1, :]) * u[n + 1, :]
            - c * k[n - 1, :] * u[n - 1, :] * u[n - 2, :]
        )
        return out_.at[n, :].set(term)

    out = lax.fori_loop(2, N_les, body, out)
    return out

# -----------------------
# Stochastic SMK closure
# -----------------------
@jax.jit
def apply_stoch_smk(u, x1, x2, eta1, eta2):
    us = u[N_les - 1, :]
    usm1 = u[N_les - 2, :]

    phi_s = jnp.angle(us)
    phi_sm1 = jnp.angle(usm1)

    amp_sp1 = jnp.abs(us) * z0 * jnp.exp(eta1)
    amp_sp2 = amp_sp1 * z0 * jnp.exp(eta2)

    delta_sp1 = 0.5 * jnp.pi + x1
    delta_sp2 = 0.5 * jnp.pi + x2

    phi_sp1 = phi_s + phi_sm1 + delta_sp1
    phi_sp2 = phi_sp1 + phi_s + delta_sp2

    u_sp1 = amp_sp1 * jnp.exp(1j * phi_sp1)
    u_sp2 = amp_sp2 * jnp.exp(1j * phi_sp2)

    u = u.at[N_les, :].set(u_sp1)
    u = u.at[N_les + 1, :].set(u_sp2)
    return u

# -----------------------
# ETD-RK4 step
# -----------------------
@jax.jit
def RK4_etd(u, x1, x2, eta1, eta2):
    def F(uu):
        uu = apply_stoch_smk(uu, x1, x2, eta1, eta2)
        return forcing + G_les(uu)

    u1 = apply_stoch_smk(u, x1, x2, eta1, eta2)
    A1 = dt * F(u1)

    u2 = ek * (u + A1 / 2.0)
    u2 = apply_stoch_smk(u2, x1, x2, eta1, eta2)
    A2 = dt * F(u2)

    u3 = ek * u + A2 / 2.0
    u3 = apply_stoch_smk(u3, x1, x2, eta1, eta2)
    A3 = dt * F(u3)

    u4 = u * (ek ** 2) + ek * A3
    u4 = apply_stoch_smk(u4, x1, x2, eta1, eta2)
    A4 = dt * F(u4)

    u_new = (ek ** 2) * (u + A1 / 6.0) + ek * (A2 + A3) / 3.0 + A4 / 6.0
    u_new = apply_stoch_smk(u_new, x1, x2, eta1, eta2)
    return u_new

# -----------------------
# OU update
# -----------------------
@jax.jit
def ou_update(x, xi, tau, sigma, mean):
    alpha = jnp.exp(-dt / tau)
    var_add = (sigma * sigma / 2.0) * (1.0 - jnp.exp(-2.0 * dt / tau))
    return mean + alpha * (x - mean) + jnp.sqrt(var_add) * xi

# -----------------------
# One integrator step
# -----------------------
def _one_step(carry, _):
    u, x1, x2, eta1, eta2, key = carry

    key, kn = random.split(key)
    xi = random.normal(kn, shape=(batch, 4), dtype=jnp.float64)

    x1 = ou_update(x1, xi[:, 0], tau_sp1, sigma_x, 0.0)
    x2 = ou_update(x2, xi[:, 1], tau_sp2, sigma_x, 0.0)
    eta1 = ou_update(eta1, xi[:, 2], tau_sp1, sigma_eta, mu_eta)
    eta2 = ou_update(eta2, xi[:, 3], tau_sp2, sigma_eta, mu_eta)

    u = RK4_etd(u, x1, x2, eta1, eta2)
    return (u, x1, x2, eta1, eta2, key), None

def advance_n(carry, n_steps: int):
    carry, _ = lax.scan(_one_step, carry, xs=None, length=n_steps)
    return carry

advance_n = jax.jit(advance_n, static_argnames=("n_steps",))

# -----------------------
# Initialise OU variables
# -----------------------
key = random.PRNGKey(seed)

def init_ou(key):
    k1, k2, k3, k4 = random.split(key, 4)

    if same_ou0_within_ic:
        x1_0 = random.normal(k1, (1,), dtype=jnp.float64) * (sigma_x / jnp.sqrt(2.0))
        x2_0 = random.normal(k2, (1,), dtype=jnp.float64) * (sigma_x / jnp.sqrt(2.0))
        eta1_0 = mu_eta + random.normal(k3, (1,), dtype=jnp.float64) * (sigma_eta / jnp.sqrt(2.0))
        eta2_0 = mu_eta + random.normal(k4, (1,), dtype=jnp.float64) * (sigma_eta / jnp.sqrt(2.0))
        x1 = jnp.repeat(x1_0, repeats=n_ens)
        x2 = jnp.repeat(x2_0, repeats=n_ens)
        eta1 = jnp.repeat(eta1_0, repeats=n_ens)
        eta2 = jnp.repeat(eta2_0, repeats=n_ens)
    else:
        x1 = random.normal(k1, (batch,), dtype=jnp.float64) * (sigma_x / jnp.sqrt(2.0))
        x2 = random.normal(k2, (batch,), dtype=jnp.float64) * (sigma_x / jnp.sqrt(2.0))
        eta1 = mu_eta + random.normal(k3, (batch,), dtype=jnp.float64) * (sigma_eta / jnp.sqrt(2.0))
        eta2 = mu_eta + random.normal(k4, (batch,), dtype=jnp.float64) * (sigma_eta / jnp.sqrt(2.0))

    return x1, x2, eta1, eta2

key, kou = random.split(key)
x1, x2, eta1, eta2 = init_ou(kou)

u = jnp.asarray(u0, dtype=jnp.complex128)
u = apply_stoch_smk(u, x1, x2, eta1, eta2)
carry = (u, x1, x2, eta1, eta2, key)

# -----------------------
# Transient
# -----------------------
print(f"Transient: {transient_steps} steps")
if transient_steps > 0:
    carry = advance_n(carry, transient_steps)

# -----------------------
# Save trajectories
# -----------------------
save_u = np.zeros((N_les, n_ens, n_saves), dtype=np.complex128)

save_col = 0
prev_step = 0

if include_t0 and save_steps[0] == 0:
    u_now = np.array(carry[0][:N_les, :], dtype=np.complex128)
    save_u[:, :, save_col] = u_now
    save_col += 1
    prev_step = 0

print(f"Run: ic_index={ic_index}, n_ens={n_ens}, batch={batch}, n_saves={n_saves}")

for target in tqdm(save_steps[save_col:]):
    n_to_advance = int(target - prev_step)
    if n_to_advance > 0:
        carry = advance_n(carry, n_to_advance)
        prev_step = target

    u_now = np.array(carry[0][:N_les, :], dtype=np.complex128)
    save_u[:, :, save_col] = u_now
    save_col += 1

# -----------------------
# Save output
# -----------------------
Path("npzs").mkdir(parents=True, exist_ok=True)

meta = dict(
    GT_PATH=str(GT_PATH),
    ic_index=int(ic_index),
    t0=int(t0),
    dt=float(dt),
    nu=float(nu),
    eps0=float(eps0),
    eps1=float(eps1),
    lam=float(lam),
    a=float(a),
    b=float(b),
    c=float(c),
    N_les=int(N_les),
    N_state=int(N_state),
    n_ic=int(n_ic),
    n_ens=int(n_ens),
    batch=int(batch),
    transient_steps=int(transient_steps),
    sim_steps=int(sim_steps),
    save_stride=int(save_stride),
    save_steps=[int(s) for s in save_steps],
    include_t0=bool(include_t0),
    sigma_x=float(sigma_x),
    sigma_eta=float(sigma_eta),
    mu_eta=float(mu_eta),
    tau_sp1=float(tau_sp1),
    tau_sp2=float(tau_sp2),
    seed=int(seed),
    same_ou0_within_ic=bool(same_ou0_within_ic),
)

tmp_path = "npzs/_tmp_sabra_smk_u_linear.npz"
final_path = OUT_PATH
np.savez_compressed(tmp_path, u=save_u, meta=meta)
os.replace(tmp_path, final_path)
print(f"[done] wrote {Path(final_path).resolve()}")
print("save_u shape:", save_u.shape)