"""Visualization of what happens to points in 3D when a 1-parameter subgroup of
O(3) is explored."""
from functools import partial
from jax import numpy as jnp, random as jrand, jit, vmap
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt

from datasets import spherical_uniform

# rng
SEED = 0
key = jrand.PRNGKey(SEED)
keys = jrand.split(key, num=100)

# params
n_pts = 5  # number of points to be drawn from the manifold
n_ts = 50  # number of times to be used for the orbits
t_max = 2*jnp.pi  # max time for the orbit
Rcnp = .05  # concept manifold radius

# draw points from a 2D subspace
U = jnp.array([[1, 0, 0], [0, 1, 0]])
dx_ = spherical_uniform(key=keys[0], d=2, shape=(n_pts,)) @ U
x0 = jrand.normal(key=keys[1], shape=(2,))
x0 = jnp.concatenate((x0, jnp.zeros(1)))
x_ = x0 + Rcnp*dx_

# this is the param that governs how "coherent" the stringy donut is
disp = Rcnp/((x0@x0)**.5)
print(f"Donut dispersion index: {disp:.3f}")

# create a group element using an element from the respective algebra and some
# time parameters
B = jrand.normal(key=keys[2], shape=(3, 3))
A = (B - B.T) / 2
t_ = jnp.linspace(0, t_max, n_ts)
exps = vmap(expm)(A*t_[:, None, None])

# apply the elements to all points to obtain their images
batchmm = jit(partial(jnp.einsum, 'tij,nj->nti'))
xg_ = batchmm(exps, x_)

# and then produce a plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
for pt_id, orbit in enumerate(xg_):
    plt.plot(*orbit.T, marker='o')
plt.scatter(*x0, marker='x', color='black')
plt.title(f'Stringy donut: {n_pts} orbits, Rcnp/||x0||: {disp:.3f}')
plt.show()
