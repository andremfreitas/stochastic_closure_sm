#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import time
import os
import sys
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------
# Precision
# -----------------------
PRECISION = 32

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

# -----------------------
# Fixed parameters
# -----------------------
N = 15
nu = 1e-12
dt = 1e-5
a, b, c = (1.0, -0.50, -0.50)

k_list, ek_list = [], []
eps0 = 0.5 / (2**0.5)
eps1 = 0.25

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

# -----------------------
# User settings
# -----------------------
data_path = "u_40_2.npz"
models_dir = "m15_v2_updt/"
out_dir = models_dir

num_steps = 100_000
batch_size = 1024

# save every stride
save_stride = 100  
num_steps_eff = (num_steps // save_stride) * save_stride  # drop tail to keep exact spacing
num_save = num_steps_eff // save_stride
t_idx = np.arange(save_stride, num_steps_eff + 1, save_stride, dtype=np.int64)  # stored RK step numbers

# Must match training
N1 = 3
N2 = 2

jit_boolean = True

# -----------------------
# Load data
# -----------------------
data_gt = np.load(data_path)
u_gt = data_gt["u"].astype(np_c_prec)  # (N, n_ics, T)

if u_gt.ndim != 3 or u_gt.shape[0] != N:
    raise ValueError(f"Expected u_gt shape (N, n_ics, T) with N={N}, got {u_gt.shape}")

n_ics = u_gt.shape[1]
batch_size = n_ics
T0 = u_gt.shape[2]
print("Finished loading data")
print(f"Baseline data n_ics = {n_ics}, T = {T0}")
print(f"num_steps rollout = {num_steps} (effective saved horizon {num_steps_eff}), save_stride = {save_stride}")
print(f"num_save snapshots = {num_save}, batch_size = {batch_size}")

if n_ics % batch_size != 0:
    raise ValueError("n_ics must be divisible by batch_size (same assumption as before).")

# -----------------------
# sigma_last2 (same as training)
# -----------------------
alpha = 0.1
u_last2 = u_gt[-2:, :, :]  # (2, n_ics, T)
sigma_last2 = alpha * dt**0.5 * np.real(k0[-2:])**0.5 * (np.mean(np.abs(u_last2)**2, axis=(1, 2))**0.5)**(3/2) # (2,)
print(f'std last 2 shells: {sigma_last2}')
sigma_last2 = tf.constant(sigma_last2.astype(np.float32 if PRECISION == 32 else np.float64), dtype=ilayer)

# globals overwritten per-batch
k = None
ek = None
forcing_tf = None
k2 = None

@tf.function(jit_compile=jit_boolean)
def G(u):
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
    A1 = dt * (forcing_tf + G(u))
    A2 = dt * (forcing_tf + G(ek * (u + A1 / 2)))
    A3 = dt * (forcing_tf + G(ek * u + A2 / 2))
    A4 = dt * (forcing_tf + G(u * (ek**2) + ek * A3))
    u_next = (ek**2) * (u + A1 / 6) + ek * (A2 + A3) / 3 + A4 / 6
    return u_next

@tf.function(jit_compile=jit_boolean)
def time_evol(model, aux):
    # aux: (N, bs) complex
    aux_real = tf.transpose(tf.math.real(aux))  # (bs, N)
    aux_im = tf.transpose(tf.math.imag(aux))    # (bs, N)
    aux_tot = tf.stack([aux_real, aux_im], axis=-1)  # (bs, N, 2)

    pred = model(aux_tot[:, -N1:, :])  # (bs, N2, 2)
    u11_mu = tf.complex(pred[:, 0, 0], pred[:, 0, 1])  # (bs,)
    u12_mu = tf.complex(pred[:, 1, 0], pred[:, 1, 1])  # (bs,)

    aux_next = RK4(aux)

    bs = tf.shape(u11_mu)[0]
    eta9 = tf.random.normal((bs, 2), dtype=ilayer)
    eta10 = tf.random.normal((bs, 2), dtype=ilayer)
    eta9_c = tf.complex(eta9[:, 0], eta9[:, 1]) / 2**0.5
    eta10_c = tf.complex(eta10[:, 0], eta10[:, 1]) / 2**0.5

    stoch_9 = tf.cast(sigma_last2[0], tf_c_prec) * eta9_c
    stoch_10 = tf.cast(sigma_last2[1], tf_c_prec) * eta10_c

    corr_9_det = dt * 1j * (a * k2[-3, :] * u11_mu * tf.math.conj(aux[-1, :]))
    corr_10_det = dt * 1j * (a * k2[-2, :] * u12_mu * tf.math.conj(u11_mu) +
                             b * k2[-3, :] * u11_mu * tf.math.conj(aux[-2, :]))

    aux_9 = aux_next[-2, :] + corr_9_det + stoch_9
    aux_10 = aux_next[-1, :] + corr_10_det + stoch_10

    aux_updt = tf.concat(
        [aux_next[:N - 2, :], tf.expand_dims(aux_9, axis=0), tf.expand_dims(aux_10, axis=0)],
        axis=0
    )
    return aux_updt

# -----------------------
# Run inference over all *.keras models
# -----------------------
if not os.path.isdir(models_dir):
    raise FileNotFoundError(f"models_dir not found: {models_dir}")

keras_files = sorted([fn for fn in os.listdir(models_dir) if fn.endswith(".keras")])
if len(keras_files) == 0:
    raise FileNotFoundError(f"No .keras models found in {models_dir}")

for model_name in keras_files:
    path_to_model = os.path.join(models_dir, model_name)
    print(f"\nLoading model: {path_to_model}")
    model = tf.keras.models.load_model(path_to_model)

    # Loop over IC batches
    for ib in range(0, n_ics, batch_size):
        batch_start = ib
        batch_end = ib + batch_size
        bs = batch_size

        # Tile constants for this bs
        k = tf.convert_to_tensor(np.transpose(np.tile(k0, (bs, 1))), dtype=tf_c_prec)  # (N, bs)
        ek = tf.convert_to_tensor(np.transpose(np.tile(ek0, (bs, 1))), dtype=tf_c_prec)
        forcing_tf = tf.convert_to_tensor(np.transpose(np.tile(forcing0, (bs, 1))), dtype=tf_c_prec)
        k2 = tf.convert_to_tensor(np.transpose(np.tile(k2_0, (bs, 1))), dtype=tf_c_prec)  # (N+2, bs)

        # IC at t=0
        aux = tf.convert_to_tensor(u_gt[:, batch_start:batch_end, 0], dtype=tf_c_prec)  # (N, bs)

        # Storage only for snapshots
        u_storage = tf.Variable(tf.zeros([N, bs, num_save], dtype=tf_c_prec))

        print(f"Batch [{batch_start}:{batch_end}] aux.shape = {aux.shape}")
        start_time = time.time()

        j = 0
        for i in tqdm(range(num_steps_eff), desc=f"Rollout {model_name} batch {batch_start}-{batch_end}"):
            aux = time_evol(model, aux)

            # save every stride (i is 0-based, so save at i+1 divisible by stride)
            if (i + 1) % save_stride == 0:
                u_storage[:, :, j].assign(aux)
                j += 1

        end_time = time.time()
        print(f"Time duration (batch) = {end_time - start_time:.3f} s | saved {j} snapshots")

        out_path = os.path.join(
            out_dir,
            f"u_{os.path.splitext(model_name)[0]}_stride{save_stride}_b{batch_start}_{batch_end}.npz"
        )
        np.savez_compressed(out_path, u=u_storage.numpy(), t_idx=t_idx, save_stride=save_stride, dt=dt)
        print(f"Saved: {out_path}")
