import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from tqdm import tqdm

config.update("jax_enable_x64", True)


# ==========================================================
# Simulation parameters
# ==========================================================

N = 40                # total number of shells
Ncut = 15             # LES cutoff shell (not used here but kept for reference)

nu = 1e-12            # viscosity
dt = 1e-8             # integration timestep

a, b, c = (1.0, -0.5, -0.5)

n_ics = 256           # number of initial conditions
num_steps = 1_000_000


# ==========================================================
# Construct shell wavenumbers and forcing
# ==========================================================

k = []
ek = []
forcing = []

eps0 = 0.5 / (2**0.5)
eps1 = 0.7 * eps0

for n in range(N):

    kn = 2**n
    k.append(kn)

    ek.append(np.exp(-nu * dt * kn * kn / 2.0))

    if n == 0:
        forcing.append(eps0 + 1j * eps0)
    elif n == 1:
        forcing.append(eps1 + 1j * eps1)

k = jnp.array(k, dtype=jnp.complex128)
ek = jnp.array(ek, dtype=jnp.complex128)

forcing = jnp.array(forcing, dtype=jnp.complex128)
forcing = jnp.concatenate((forcing, jnp.zeros(N - 2)))

# tile vectors across batch dimension
k = jnp.tile(k[:, None], (1, n_ics))
ek = jnp.tile(ek[:, None], (1, n_ics))
forcing = jnp.tile(forcing[:, None], (1, n_ics))


# ==========================================================
# Sabra shell model nonlinear term
# ==========================================================

@jax.jit
def G(u):

    coupling = jnp.expand_dims(
        (a * k[1] * jnp.conj(u[1]) * u[2]) * 1j,
        axis=0
    )

    coupling = jnp.concatenate([
        coupling,
        jnp.expand_dims(
            (a * k[2] * jnp.conj(u[2]) * u[3]
             + b * k[1] * jnp.conj(u[0]) * u[2]) * 1j,
            axis=0
        )
    ], axis=0)

    for n in range(2, N - 2):

        term = (
            a * k[n + 1] * jnp.conj(u[n + 1]) * u[n + 2]
            + b * k[n] * jnp.conj(u[n - 1]) * u[n + 1]
            - c * k[n - 1] * u[n - 1] * u[n - 2]
        ) * 1j

        coupling = jnp.concatenate([
            coupling,
            jnp.expand_dims(term, axis=0)
        ], axis=0)

    coupling = jnp.concatenate([
        coupling,
        jnp.expand_dims(
            (b * k[N - 2] * jnp.conj(u[N - 3]) * u[N - 1]
             - c * k[N - 3] * u[N - 3] * u[N - 4]) * 1j,
            axis=0
        )
    ], axis=0)

    coupling = jnp.concatenate([
        coupling,
        jnp.expand_dims(
            (-c * k[N - 2] * u[N - 2] * u[N - 3]) * 1j,
            axis=0
        )
    ], axis=0)

    return coupling


# ==========================================================
# RK4 integrator
# ==========================================================

@jax.jit
def RK4(u):

    A1 = dt * (forcing + G(u))
    A2 = dt * (forcing + G(ek * (u + A1 / 2)))
    A3 = dt * (forcing + G(ek * u + A2 / 2))
    A4 = dt * (forcing + G(u * (ek**2) + ek * A3))

    u = (ek**2) * (u + A1 / 6) + ek * (A2 + A3) / 3 + A4 / 6

    return u


# ==========================================================
# Initial conditions
# ==========================================================

u = np.zeros((N, n_ics), dtype=np.complex128)

k1d = np.array([2**n for n in range(N)])

for i in range(n_ics):
    for n in range(6):

        r = np.random.rand()

        u[n, i] = (
            0.01
            * k1d[n] ** (-1/3)
            * (np.cos(2 * np.pi * r) + 1j * np.sin(2 * np.pi * r))
        )

aux = jnp.array(u)


# ==========================================================
# Time integration
# ==========================================================

print(f"Running simulation with {num_steps} steps...")

start_time = time.time()

for _ in tqdm(range(num_steps)):
    aux = RK4(aux)

end_time = time.time()

print("")
print(f"Simulation time: {end_time - start_time:.2f} s")


# ==========================================================
# Save final state
# ==========================================================

np.savez_compressed("dataset.npz", u=np.array(aux))
