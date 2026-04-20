"""
Train the neural Langevin closure on trajectories stored in `dataset.npz`.

The model predicts the last two closure variables from the last `N1` resolved
shells and is trained by unrolling in time (solver-in-the-loop approach) against the deterministic reference
data.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import time
import sys
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

np.random.seed(42)
tf.random.set_seed(42)

PRECISION = 32

if PRECISION == 64:
    np_c_prec = np.complex128
    tf_c_prec = tf.complex128
    ilayer = tf.float64
    tf.keras.backend.set_floatx('float64')
elif PRECISION == 32:
    np_c_prec = np.complex64
    tf_c_prec = tf.complex64
    ilayer = tf.float32

# Fixed Parameters & Useful vectors
N = 15  # Num of shells
nu = 10**-12  # viscosity
dt = 1 * 10**-5  # integration step
a, b, c = (1.0, -0.50, -0.50)
k, ek, forcing = [], [], []
eps0 = 0.5 / (2**0.5)
eps1 = 0.25
for n in range(N):
    k.append(2**n)
    ek.append(np.exp(-nu * dt * k[n] * k[n] / 2.0))
    if n == 0:
        forcing.append(eps0 + eps0*1j)
    if n == 1:
        forcing.append(eps1 + eps1*1j)
k2 = []
for n in range(N+2):
    k2.append(2**n)

k0 = np.array(k, dtype=np_c_prec)
ek0 = np.array(ek, dtype=np_c_prec)
forcing = np.array(forcing, dtype=np_c_prec)
forcing0 = np.concatenate((forcing, np.zeros(N - 2, dtype=np_c_prec)))

batch_size = 1024

k = np.transpose(np.tile(k0, (batch_size, 1)))
ek = np.transpose(np.tile(ek0, (batch_size, 1)))
forcing = np.transpose(np.tile(forcing0, (batch_size, 1)))
k2_0 = np.array(k2, dtype=np_c_prec)
k2 = np.transpose(np.tile(k2_0, (batch_size, 1)))

def corrector(N1, N2, num_layers, hidden_size, batch_size=None):
    input_u = keras.Input(shape=(N1, 2), batch_size=batch_size, name="state_input")

    x = keras.layers.Flatten()(input_u)
    for _ in range(num_layers):
        x = keras.layers.Dense(units=hidden_size, activation="relu")(x)

    x = keras.layers.Dense(units=N2 * 2, activation="linear")(x)
    output = keras.layers.Reshape((N2, 2))(x)
    return keras.Model(inputs=input_u, outputs=output)

# Example usage
N1 = 3  # Number of input shells (features are Re/Im)
N2 = 2  # Predict two complex numbers (two shells worth)
num_layers = 7
hidden_size = 256

model = corrector(N1, N2, num_layers, hidden_size, batch_size)
model.summary()
variables = model.trainable_variables

lr0 = 3e-4
optimizer = keras.optimizers.Adam(lr0)
model.compile(optimizer=optimizer)
epochs = 10

jit_boolean = True

print("\n")

##############
# Loading data
data_path = "dataset.npz"
data_gt = np.load(data_path)
u_gt = data_gt["u"]
print(u_gt.shape)
print("Finished loading data")
###############

n_ics0 = u_gt.shape[1]
num_steps0 = u_gt.shape[2]

print(f"Baseline data n_ics = {n_ics0}")
print(f"num_steps = {num_steps0}")

alpha = 0.1

u_last2 = u_gt[-2:, :, :]  # (2, n_ics, T)
sigma_last2 = alpha * dt**0.5 * np.real(k0[-2:])**0.5 * (np.mean(np.abs(u_last2)**2, axis=(1, 2))**0.5)**(3/2) # (2,)
print(f'std last 2 shells: {sigma_last2}')
sigma_last2 = tf.constant(sigma_last2.astype(np.float32 if PRECISION == 32 else np.float64), dtype=ilayer)


@tf.function(jit_compile=jit_boolean)
def G(u):
    coupling = tf.expand_dims(((a * k[0 + 1, :] * tf.math.conj(u[0 + 1, :]) * u[0 + 2, :]) * 1j), axis=0)
    coupling = tf.concat([coupling, tf.expand_dims(((a * k[1 + 1, :] * tf.math.conj(u[1 + 1, :]) * u[1 + 2, :] + b * k[1, :] * tf.math.conj(u[1 - 1, :]) * u[1 + 1, :]) * 1j), axis=0)], axis=0)

    for n in range(2, N-2):
        coupling = tf.concat([coupling,
                tf.expand_dims(((a * k[n + 1, :] * tf.math.conj(u[n + 1, :]) * u[n + 2, :] + b * k[n, :] * tf.math.conj(u[n - 1, :]) * u[n + 1, :] - c * k[n - 1, :] * u[n - 1, :]
                 * u[n - 2, :]) * 1j), axis=0)], axis=0)

    coupling = tf.concat([coupling, tf.expand_dims(((b * k[N-2, :] * tf.math.conj(u[N-2 - 1, :]) * u[N-2 + 1, :] - c * k[N-2 - 1, :] * u[N-2 - 1, :] * u[N-2 - 2, :]) * 1j), axis=0)], axis=0)
    coupling = tf.concat([coupling, tf.expand_dims(((-c * k[N-1 - 1, :] * u[N-1 - 1, :] * u[N-1 - 2, :]) * 1j), axis=0)], axis=0)

    return coupling

@tf.function(jit_compile=True)
def RK4(u):
    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1/2)))
    A3 = dt * (forcing + G(ek * u + A2/2))
    A4 = dt * (forcing + G(u*(ek**2) + ek*A3))

    u = (ek**2)*(u + A1/6) + ek*(A2 + A3)/3 + A4/6
    return u

@tf.function(jit_compile=False)
def training_loop(u0, gt_tensor, msteps):
    with tf.GradientTape() as tape:
        u = u0
        for i in range(msteps - 1):
            aux = u[:, :, i]  # (N, bs) complex

            aux_real = tf.transpose(tf.math.real(aux))
            aux_im = tf.transpose(tf.math.imag(aux))
            aux_real_3d = tf.expand_dims(aux_real, axis=-1)
            aux_im_3d = tf.expand_dims(aux_im, axis=-1)
            aux_tot = tf.concat([aux_real_3d, aux_im_3d], axis=-1)  # (bs, N, 2)

            pred = model(aux_tot[:, -N1:, :])  # (bs, N2, 2)
            u11_mu = tf.complex(pred[:, 0, 0], pred[:, 0, 1])  # (bs,)
            u12_mu = tf.complex(pred[:, 1, 0], pred[:, 1, 1])  # (bs,)

            aux_next = tf.convert_to_tensor(RK4(u[:, :, i]))  # (N, bs) complex

            eta9 = tf.random.normal((tf.shape(u11_mu)[0], 2), dtype=ilayer)
            eta10 = tf.random.normal((tf.shape(u12_mu)[0], 2), dtype=ilayer)
            eta9_c = tf.complex(eta9[:, 0], eta9[:, 1]) / 2**0.5
            eta10_c = tf.complex(eta10[:, 0], eta10[:, 1]) / 2**0.5

            stoch_9 = tf.cast(sigma_last2[0], tf_c_prec) * eta9_c
            stoch_10 = tf.cast(sigma_last2[1], tf_c_prec) * eta10_c

            corr_9_det = dt * 1j * (a * k2[-3] * u11_mu * tf.math.conj(aux[-1, :]))
            corr_10_det = dt * 1j * (a * k2[-2] * u12_mu * tf.math.conj(u11_mu) + b * k2[-3] * u11_mu * tf.math.conj(aux[-2, :]))

            aux_9 = aux_next[-2, :] + corr_9_det + stoch_9
            aux_10 = aux_next[-1, :] + corr_10_det + stoch_10

            aux_updt = tf.concat([aux_next[:N - 2, :], tf.expand_dims(aux_9, axis=0), tf.expand_dims(aux_10, axis=0)], axis=0)
            aux_updt_3d = tf.expand_dims(aux_updt, axis=-1)
            u = tf.concat([u, aux_updt_3d], axis=-1)

        loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(tf.math.abs(u[-6:, :, :] - gt_tensor[-6:, :, :])), axis=(1, 2)) /
            ((tf.sqrt(tf.reduce_mean(tf.square(tf.abs(gt_tensor[-6:, :, :])), axis=(1, 2)))) *
             tf.sqrt(tf.reduce_mean(tf.square(tf.abs(u[-6:, :, :])), axis=(1, 2))))
        )

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

losses = []
batchwise_losses = []

msteps_interval = [2, 4, 10, 15, 20, 25, 30, 40, 50]
msteps_chosen = 15

msteps_sched = []
epochs = []

for msteps in msteps_interval:
    msteps_sched.append(msteps)
    if msteps == msteps_chosen:
        epochs.append(50)
        break
    epochs.append(1)

print(msteps_sched)
print(epochs)

directory = "outputs_langevin_closure/"

if not os.path.exists(directory):
    os.makedirs(directory)

def plot_loss(losses, msteps):
    plt.plot(losses, linewidth=2)
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"{directory}loss_{msteps}.png")
    plt.close()

def closest_divisible(x, y):
    return x - (x % y)

start = time.time()
for j in range(len(epochs)):
    msteps = msteps_sched[j]
    print(f"\n msteps = {msteps} \n")

    num_steps_new = closest_divisible(num_steps0, msteps)
    gt_reshaped = tf.reshape(u_gt[:, :, :num_steps_new], [N, int(n_ics0 * num_steps_new / msteps), msteps])
    gt_reshaped = tf.transpose(gt_reshaped, (1, 0, 2))

    data = tf.data.Dataset.from_tensor_slices(gt_reshaped)
    bdata = data.batch(batch_size=batch_size)
    losses = []
    for epoch in range(epochs[j]):
        sbdata = bdata.shuffle(buffer_size=batch_size)
        for gt in sbdata:
            gt = tf.transpose(gt, (1, 0, 2))
            ic = gt[:N, :, :1]
            bs = gt.shape[1]
            k = np.transpose(np.tile(k0, (bs, 1)))
            ek = np.transpose(np.tile(ek0, (bs, 1)))
            forcing = np.transpose(np.tile(forcing0, (bs, 1)))
            k2 = np.transpose(np.tile(k2_0, (bs, 1)))

            loss = training_loop(ic, gt, msteps_sched[j])
            batchwise_losses.append(loss.numpy())
            if np.isnan(loss.numpy()):
                print('NaN found in loss. Stopping the run ...')
                sys.exit(1)

        losses.append(sum(batchwise_losses[-batch_size:]) / batch_size)
        print(f"Epoch {epoch}, Loss: {sum(batchwise_losses[-batch_size:]) / batch_size}")

plot_loss(losses, msteps)
tf.keras.models.save_model(model, f"{directory}m{msteps}.keras")

end = time.time()
print(f"Training duration:{end-start}s")
