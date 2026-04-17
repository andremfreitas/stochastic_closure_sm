#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted ML inference (v2_updt-style: deterministic NN mean + additive stochasticity on last 2 shells)
to run:

  - many ICs
  - an ensemble per IC (independent stochastic draws per ensemble member)
  - log-spaced saving in step space (x points per decade)
  - saving ONLY ensemble variance (and optionally mean) for each shell and IC

Input:
  data_path: npz with u of shape (N, n_ics, T)

Output:
  one .npz per .keras model with:
    var_u: (N, n_ics, n_saves)  float64  (complex variance E|u - Eu|^2 over ensemble)
    (optional) mean_u: (N, n_ics, n_saves) complex64
    meta: dict including save_steps, dt, n_ens, etc.
"""

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

np.random.seed(42)
tf.random.set_seed(42)

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--noise-steps",type=int,default=1_000_000,help="Number of rollout steps with noise (0 = no noise, >=num_steps = always on)")
args, _ = parser.parse_known_args()


# ============================================================
# Precision (KEEP CONSISTENT)
# ============================================================
PRECISION = 32  # 32 or 64
if PRECISION == 64:
    np_c_prec = np.complex128
    tf_c_prec = tf.complex128
    ilayer = tf.float64
    tf.keras.backend.set_floatx("float64")
else:
    np_c_prec = np.complex64
    tf_c_prec = tf.complex64
    ilayer = tf.float32
    tf.keras.backend.set_floatx("float32")

# ============================================================
# Fixed parameters
# ============================================================
N = 15
nu = 1e-12
dt = 1e-5
a, b, c = (1.0, -0.50, -0.50)

eps0 = 0.5 / (2**0.5)
eps1 = 0.25

k_list, ek_list = [], []
forcing_list = []
for n in range(N):
    kn = 2**n
    k_list.append(kn)
    ek_list.append(np.exp(-nu * dt * kn * kn / 2.0))
    if n == 0:
        forcing_list.append(eps0 + 1j * eps0)
    elif n == 1:
        forcing_list.append(eps1 + 1j * eps1)

k2_list = [2**n for n in range(N + 2)]

k0 = np.array(k_list, dtype=np_c_prec)                 # (N,)
ek0 = np.array(ek_list, dtype=np_c_prec)               # (N,)
forcing = np.array(forcing_list, dtype=np_c_prec)
forcing0 = np.concatenate((forcing, np.zeros(N - 2, dtype=np_c_prec)))  # (N,)
k2_0 = np.array(k2_list, dtype=np_c_prec)              # (N+2,)

# TF scalar dt for graph stability
dt_c  = tf.constant(dt, dtype=tf_c_prec)

# ============================================================
# User settings
# ============================================================
data_path = "u_40_2.npz"
models_dir = "m15_v2_updt/"
out_dir = models_dir  # or set to something else like "npzs_ml_var_v2"

NOISE_STEPS = args.noise_steps     # oarger than num_steps if you want fully stochastic

# rollout horizon (steps)
num_steps = 500_000

# ensemble per IC
n_ens = 1024
t0 = 0
n_ics_use = None          # None => use all ICs, else int

# log saving
points_per_decade = 8
include_t0 = True

# training params
N1 = 3
N2 = 2
jit_boolean = True

# outputs
SAVE_MEAN = True
OUT_PREFIX = "m15_v2_updt_1step_noise"

# ============================================================
# Log save schedule: uniform density in log10(step)
# ============================================================
def make_log_schedule(max_step: int, points_per_decade: int = 7, min_step: int = 1) -> list[int]:
    """
    Roughly points_per_decade samples per decade over [min_step, max_step].
    Returns integer step indices, includes max_step, and (optionally) t=0 handled outside.
    """
    if max_step < min_step:
        return []

    log_min = np.log10(min_step)
    log_max = np.log10(max_step)
    decades = log_max - log_min

    n_intervals = int(np.ceil(decades * points_per_decade))
    logs = np.linspace(log_min, log_max, n_intervals + 1)

    pts = np.unique(np.rint(10**logs).astype(int))
    pts = pts[(pts >= min_step) & (pts <= max_step)]

    if pts.size == 0 or pts[0] != min_step:
        pts = np.insert(pts, 0, min_step)
    if pts[-1] != max_step:
        pts = np.append(pts, max_step)

    return pts.tolist()

save_steps = make_log_schedule(num_steps, points_per_decade=points_per_decade, min_step=1)
if include_t0 and (len(save_steps) == 0 or save_steps[0] != 0):
    save_steps = [0] + save_steps
n_saves = len(save_steps)
print("Save steps (head):", save_steps[:15], "... total:", n_saves)

# ============================================================
# Load data
# ============================================================
data_gt = np.load(data_path)
u_gt = data_gt["u"].astype(np_c_prec)  # (N, n_ics, T)

if u_gt.ndim != 3 or u_gt.shape[0] != N:
    raise ValueError(f"Expected u_gt shape (N, n_ics, T) with N={N}, got {u_gt.shape}")

T0 = u_gt.shape[2]
if t0 < 0 or t0 >= T0:
    raise ValueError(f"t0={t0} out of bounds for T={T0}")

n_ics_total = u_gt.shape[1]
n_ics = n_ics_total if n_ics_use is None else min(int(n_ics_use), n_ics_total)

print("Finished loading data")
print(f"Using n_ics={n_ics} (out of {n_ics_total}), T={T0}")
print(f"num_steps={num_steps}, n_ens={n_ens}, batch={n_ics*n_ens}, n_saves={n_saves}")

# ============================================================
# sigma_last2 (same as training)
# ============================================================
alpha = 0.1
u_last2 = u_gt[-2:, :n_ics, :]  # (2, n_ics, T)
sigma_last2_np = alpha * (dt**0.5) * (np.real(k0[-2:])**0.5) * (np.mean(np.abs(u_last2)**2, axis=(1, 2))**0.5) ** (3/2)
print(f"std last 2 shells: {sigma_last2_np}")

sigma_last2 = tf.constant(
    sigma_last2_np.astype(np.float32 if PRECISION == 32 else np.float64),
    dtype=ilayer
)

# ============================================================
# Tile constants for full batch (IC × ensemble)
# ============================================================
def tile_constants(bs: int):
    k = tf.convert_to_tensor(np.transpose(np.tile(k0, (bs, 1))), dtype=tf_c_prec)  # (N, bs)
    ek = tf.convert_to_tensor(np.transpose(np.tile(ek0, (bs, 1))), dtype=tf_c_prec)
    forcing_tf = tf.convert_to_tensor(np.transpose(np.tile(forcing0, (bs, 1))), dtype=tf_c_prec)
    k2 = tf.convert_to_tensor(np.transpose(np.tile(k2_0, (bs, 1))), dtype=tf_c_prec)  # (N+2, bs)
    return k, ek, forcing_tf, k2

# ============================================================
# Dynamics (TF graph) — uses globals set per run
# ============================================================
k = None
ek = None
forcing_tf = None
k2 = None

@tf.function(jit_compile=jit_boolean)
def G(u):
    u = tf.cast(u, tf_c_prec)

    coupling = tf.expand_dims(((a * k[1, :] * tf.math.conj(u[1, :]) * u[2, :]) * 1j), axis=0)
    coupling = tf.concat(
        [coupling,
         tf.expand_dims(((a * k[2, :] * tf.math.conj(u[2, :]) * u[3, :] +
                         b * k[1, :] * tf.math.conj(u[0, :]) * u[2, :]) * 1j), axis=0)],
        axis=0
    )

    for n in range(2, N - 2):
        coupling = tf.concat(
            [coupling,
             tf.expand_dims(((a * k[n + 1, :] * tf.math.conj(u[n + 1, :]) * u[n + 2, :] +
                             b * k[n, :] * tf.math.conj(u[n - 1, :]) * u[n + 1, :] -
                             c * k[n - 1, :] * u[n - 1, :] * u[n - 2, :]) * 1j), axis=0)],
            axis=0
        )

    coupling = tf.concat(
        [coupling,
         tf.expand_dims(((b * k[N - 2, :] * tf.math.conj(u[N - 3, :]) * u[N - 1, :] -
                         c * k[N - 3, :] * u[N - 3, :] * u[N - 4, :]) * 1j), axis=0)],
        axis=0
    )

    coupling = tf.concat(
        [coupling,
         tf.expand_dims(((-c * k[N - 2, :] * u[N - 2, :] * u[N - 3, :]) * 1j), axis=0)],
        axis=0
    )
    return coupling

@tf.function(jit_compile=jit_boolean)
def RK4(u):
    u = tf.cast(u, tf_c_prec)

    A1 = dt_c * (forcing_tf + G(u))
    A2 = dt_c * (forcing_tf + G(ek * (u + A1 / 2)))
    A3 = dt_c * (forcing_tf + G(ek * u + A2 / 2))
    A4 = dt_c * (forcing_tf + G(u * (ek**2) + ek * A3))
    u_next = (ek**2) * (u + A1 / 6) + ek * (A2 + A3) / 3 + A4 / 6
    return u_next

@tf.function(jit_compile=jit_boolean)
def time_evol_noise(model, aux):
    aux = tf.cast(aux, tf_c_prec)

    aux_real = tf.transpose(tf.math.real(aux))
    aux_im   = tf.transpose(tf.math.imag(aux))
    aux_tot  = tf.stack([aux_real, aux_im], axis=-1)

    pred = model(aux_tot[:, -N1:, :])

    u11_mu = tf.cast(tf.complex(pred[:, 0, 0], pred[:, 0, 1]), tf_c_prec)
    u12_mu = tf.cast(tf.complex(pred[:, 1, 0], pred[:, 1, 1]), tf_c_prec)

    aux_next = RK4(aux)
    bs = tf.shape(u11_mu)[0]

    eta9  = tf.random.normal((bs, 2), dtype=ilayer)
    eta10 = tf.random.normal((bs, 2), dtype=ilayer)

    eta9_c  = tf.cast(tf.complex(eta9[:, 0],  eta9[:, 1])  / (2.0**0.5), tf_c_prec)
    eta10_c = tf.cast(tf.complex(eta10[:, 0], eta10[:, 1]) / (2.0**0.5), tf_c_prec)

    stoch_9  = tf.cast(sigma_last2[0], tf_c_prec) * eta9_c
    stoch_10 = tf.cast(sigma_last2[1], tf_c_prec) * eta10_c

    corr_9_det = dt_c * (1j) * (a * k2[-3, :] * u11_mu * tf.math.conj(aux[-1, :]))
    corr_10_det = dt_c * (1j) * (a * k2[-2, :] * u12_mu * tf.math.conj(u11_mu) +
                                 b * k2[-3, :] * u11_mu * tf.math.conj(aux[-2, :]))

    aux_9  = aux_next[-2, :] + corr_9_det  + stoch_9
    aux_10 = aux_next[-1, :] + corr_10_det + stoch_10

    return tf.concat([aux_next[:N - 2, :],
                      tf.expand_dims(aux_9,  axis=0),
                      tf.expand_dims(aux_10, axis=0)], axis=0)


@tf.function(jit_compile=jit_boolean)
def time_evol_no_noise(model, aux):
    aux = tf.cast(aux, tf_c_prec)

    aux_real = tf.transpose(tf.math.real(aux))
    aux_im   = tf.transpose(tf.math.imag(aux))
    aux_tot  = tf.stack([aux_real, aux_im], axis=-1)

    pred = model(aux_tot[:, -N1:, :])

    u11_mu = tf.cast(tf.complex(pred[:, 0, 0], pred[:, 0, 1]), tf_c_prec)
    u12_mu = tf.cast(tf.complex(pred[:, 1, 0], pred[:, 1, 1]), tf_c_prec)

    aux_next = RK4(aux)

    corr_9_det = dt_c * (1j) * (a * k2[-3, :] * u11_mu * tf.math.conj(aux[-1, :]))
    corr_10_det = dt_c * (1j) * (a * k2[-2, :] * u12_mu * tf.math.conj(u11_mu) +
                                 b * k2[-3, :] * u11_mu * tf.math.conj(aux[-2, :]))

    aux_9  = aux_next[-2, :] + corr_9_det
    aux_10 = aux_next[-1, :] + corr_10_det

    return tf.concat([aux_next[:N - 2, :],
                      tf.expand_dims(aux_9,  axis=0),
                      tf.expand_dims(aux_10, axis=0)], axis=0)



# ============================================================
# Build (IC, ensemble) initial condition: (N, n_ics*n_ens)
# ============================================================
u0_ic = u_gt[:, :n_ics, t0]  # (N, n_ics)
u0_ic_ens = np.repeat(u0_ic[:, :, None], repeats=n_ens, axis=2)  # (N, n_ics, n_ens)
u0_all = u0_ic_ens.reshape(N, n_ics * n_ens)  # (N, batch)
batch = n_ics * n_ens

# ============================================================
# Ensemble variance helper (complex variance E|u - Eu|^2)
# ============================================================
def ensemble_var_and_mean(aux_np: np.ndarray, n_ics: int, n_ens: int):
    u3 = aux_np.reshape(N, n_ics, n_ens)          # (N, n_ics, n_ens)
    mu = u3.mean(axis=2)                          # (N, n_ics)
    var = np.mean(np.abs(u3 - mu[:, :, None])**2, axis=2)  # (N, n_ics)
    return var, mu

# ============================================================
# Run inference over all *.keras models (single batch = all IC×ens)
# ============================================================
if not os.path.isdir(models_dir):
    raise FileNotFoundError(f"models_dir not found: {models_dir}")

keras_files = sorted([fn for fn in os.listdir(models_dir) if fn.endswith(".keras")])
if len(keras_files) == 0:
    raise FileNotFoundError(f"No .keras models found in {models_dir}")

Path(out_dir).mkdir(parents=True, exist_ok=True)

for model_name in keras_files:
    path_to_model = os.path.join(models_dir, model_name)
    print(f"\nLoading model: {path_to_model}")
    model = tf.keras.models.load_model(path_to_model)

    # tile constants for the full batch
    k, ek, forcing_tf, k2 = tile_constants(batch)

    # init state
    aux = tf.convert_to_tensor(u0_all, dtype=tf_c_prec)  # (N, batch)

    # allocate outputs
    var_storage = np.zeros((N, n_ics, n_saves), dtype=np.float64)
    mean_storage = np.zeros((N, n_ics, n_saves), dtype=np.complex64) if SAVE_MEAN else None

    # optionally save t=0
    save_col = 0
    prev_step = 0
    if include_t0 and save_steps and save_steps[0] == 0:
        aux_np = aux.numpy()
        var0, mu0 = ensemble_var_and_mean(aux_np, n_ics, n_ens)
        var_storage[:, :, save_col] = var0
        if SAVE_MEAN:
            mean_storage[:, :, save_col] = mu0.astype(np.complex64)
        save_col += 1
        prev_step = 0

    start_time = time.time()

    # advance-to-target loop (log saving)
    for target in tqdm(save_steps[save_col:], desc=f"Rollout {model_name}"):
        n_to_advance = int(target - prev_step)
        for j in range(n_to_advance):
            global_step = prev_step + j  # current step index before advancing by 1
            if global_step < NOISE_STEPS:
                aux = time_evol_noise(model, aux)
            else:
                aux = time_evol_no_noise(model, aux)
        prev_step = target

        aux_np = aux.numpy()
        var_t, mu_t = ensemble_var_and_mean(aux_np, n_ics, n_ens)
        var_storage[:, :, save_col] = var_t
        if SAVE_MEAN:
            mean_storage[:, :, save_col] = mu_t.astype(np.complex64)
        save_col += 1

        # optional NaN/Inf check (cheap)
        if not np.isfinite(var_storage[:, :, save_col - 1]).all():
            print(f"[error] Non-finite variance at step {target}")
            sys.exit(1)

    end_time = time.time()
    print(f"Time duration = {end_time - start_time:.3f} s | saved {save_col} points")

    out_path = os.path.join(
        out_dir,
        f"{OUT_PREFIX}_{os.path.splitext(model_name)[0]}_nics{n_ics}_nens{n_ens}_ppd{points_per_decade}_T{num_steps}_Nnoise{NOISE_STEPS}.npz"
    )

    meta = dict(
        model_name=model_name,
        models_dir=models_dir,
        data_path=data_path,
        t0=int(t0),
        dt=float(dt),
        nu=float(nu),
        N=int(N),
        N1=int(N1),
        N2=int(N2),
        n_ics=int(n_ics),
        n_ens=int(n_ens),
        batch=int(batch),
        num_steps=int(num_steps),
        points_per_decade=int(points_per_decade),
        include_t0=bool(include_t0),
        save_steps=[int(s) for s in save_steps],
        sigma_last2=sigma_last2_np.astype(np.float64),
        PRECISION=int(PRECISION),
        jit_boolean=bool(jit_boolean),
    )

    if SAVE_MEAN:
        np.savez_compressed(out_path, var_u=var_storage, mean_u=mean_storage, meta=meta)
    else:
        np.savez_compressed(out_path, var_u=var_storage, meta=meta)

    print(f"Saved: {out_path}")

