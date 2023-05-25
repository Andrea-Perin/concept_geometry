"""Visualization of what happens when using the iterative step method for
finding the optimal time on an orthogonal group (in this case, O(3))."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import numpy as jnp, random as jrand, vmap
from jax.scipy.linalg import expm

from datasets import spherical_uniform

# folder for saving plots
OUT_DIR_PATH = 'orbit_gifs'
out_dir = Path() / OUT_DIR_PATH
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

# rng
SEED = int(input("Insert random seed: "))
key = jrand.PRNGKey(SEED)
keys = jrand.split(key, num=100)

# params
n_pts = 5  # number of points to be drawn from the manifold
Rcnp = .5  # concept manifold radius
Rtest = .01  # test point radius
n_iter = 100  # number of iterative steps for finding optimal t
Rx0 = 1.

# draw points from a 2D subspace
U = jnp.array([[1, 0, 0], [0, 1, 0]])
dx_ = spherical_uniform(key=keys[0], d=2, shape=(n_pts,)) @ U
x0 = jrand.normal(key=keys[1], shape=(2,)) * Rx0
x0 = jnp.concatenate((x0, jnp.zeros(1)))
x_ = x0 + Rcnp*dx_
# and then create a target point, sitting somewhere random
xi = jrand.normal(key=keys[5], shape=(3,))

# this is the param that governs how "coherent" the stringy donut is
disp = Rcnp/((x0@x0)**.5)
print(f"Donut dispersion index: {disp:.3f}")

# create a group element using an element from the respective algebra and some
# time parameters
B = jrand.normal(key=keys[3], shape=(3, 3))
A = (B - B.T) / 2


# this is the update function as specified in my silly computations
def x_update(x0, xi, A):
    t_hat = -((A@x0)@xi)/(((A@A)@x0)@xi)
    return expm(t_hat*A)@x0


def get_update_trajectory(x0, xi, A, n_iter=n_iter):
    traj = [x0]
    for n in range(n_iter):
        x0 = x_update(x0, xi, A)
        traj.append(x0)
    return jnp.stack(traj)


# first of all, let make sure whether or not we are dealing with a one shot thing
trajectories = [get_update_trajectory(x0_, xi, A) for x0_ in x_]


def is_fixed_point(trajectory):
    # the first diff is nonzero, and the last entry is not a diff
    diffs = jnp.diff(trajectory)[1:-1]
    return jnp.allclose(diffs, 0)


all_fixed = all(map(is_fixed_point, trajectories))
print(f"All the trajectories converge after a single iteration: {all_fixed}")


# since I am stupid, let's check if we just sample from the group a lot
t_ = jnp.linspace(-10, 10, 500)
tA = vmap(expm)(t_[:, None, None]*A)
all_imgs = jnp.einsum('tij,nj->nti', tA, x_)
diffs = all_imgs - xi
dists = (diffs*diffs).sum(axis=-1)
best_approx_idx = jnp.argmin(dists, axis=1)
naive_way = jnp.take_along_axis(all_imgs, best_approx_idx[:, None, None], 1)
naive_way = naive_way.squeeze()

# via the iterative method
iter_way = jnp.array([t[-1] for t in trajectories])

# and then via the smart way
gen_period = jnp.sqrt(jnp.linalg.eigh(A.T@A)[0])[-1]  # largest eigval
period = (jnp.pi)/gen_period
half_circle = expm(period*A)
complementary_ = jnp.einsum('ij,nj->ni', half_circle, iter_way)
complementary_traj = [get_update_trajectory(
    cx0_, xi, A)[-1] for cx0_ in complementary_]
comp_way = jnp.array(complementary_traj)

iter_dist = ((iter_way-xi)**2).sum(axis=1)
comp_dist = ((comp_way-xi)**2).sum(axis=1)
smart_way = jnp.where((iter_dist < comp_dist)[:, None], iter_way, comp_way)

# PLOTTING
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
orbit = all_imgs[0]
traj = trajectories[0]
bn = naive_way[0]
best = smart_way[0]


def plot_data(ax, all_pts, bn, best, traj, target):
    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
               alpha=.01, label='full orbit')
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], s=20,
               alpha=0.01, label='optim. trajectory')
    ax.scatter(xi[0], xi[1], xi[2], marker='x', s=20, label='target')
    ax.scatter(bn[0], bn[1], bn[2], marker='x', s=40, label='best naive')
    bs = traj[-1]
    ax.scatter(bs[0], bs[1], bs[2], marker='x', s=40, label='best optim')
    ax.scatter(best[0], best[1], best[2], marker='x', s=40, label='best smart')
    leg = plt.legend(bbox_to_anchor=(.85, 1.0), loc='upper left')
    for lh in leg.legend_handles:
        lh.set_alpha(.5)


def animate(i):
    ax.clear()
    if i <= 360:
        ax.view_init(elev=10., azim=i)
        plot_data(ax, orbit, bn, best, traj, xi)
    else:
        Az = i-(int(i/360)*360)
        ax.view_init(elev=10., azim=Az)
        plot_data(ax, orbit, bn, best, traj, xi)


anim = FuncAnimation(fig, animate, interval=1, repeat=True, frames=360)
anim.save(f'{out_dir}/animation_{SEED}.gif', fps=60)
plt.close()
