"""
Specifically training MLPs on the circle task.
Aiming for 0 error as a function of circle separation, we look at the required
number of samples needed.
"""
from itertools import cycle
from functools import partial
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from typing import Optional
from jax import numpy as jnp, vmap, jit, random as jrand, nn as jnn
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


def polygon(x):
    return jnp.ceil(jnp.pi/(jnp.arccos(1/x)))


def spherepack(x):
    return jnp.ceil(jnp.pi/(2*jnp.arcsin((x-1)/(2*(x+1)))))


class Batch2LP(eqx.Module):
    w1: jnp.ndarray
    w2: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray

    def __init__(self, in_size, out_size, width_size, n_mods, *, key):
        keys = jrand.split(key, num=4)
        lim = 1 / math.sqrt(in_size)
        self.w1 = jrand.uniform(keys[0], (n_mods, width_size, in_size), minval=-lim, maxval=lim)
        self.b1 = jrand.uniform(keys[1], (n_mods, width_size), minval=-lim, maxval=lim)
        lim = 1 / math.sqrt(width_size)
        self.w2 = jrand.uniform(keys[2], (n_mods, out_size, width_size), minval=-lim, maxval=lim)
        self.b2 = jrand.uniform(keys[3], (n_mods, out_size), minval=-lim, maxval=lim)

    def __call__(self, x):
        h = jnn.relu(jnp.einsum('nhi,i->nh', self.w1, x) + self.b1)
        return jnn.sigmoid(jnp.einsum('noh,nh->no', self.w2, h) + self.b2)


# sample points uniformly from a circle
def get_points(N, alpha):
    ts_ = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha * jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    pts = jnp.concatenate((inner, outer))
    labs = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return pts, labs


# TRAINING TOOLS
def loss(
    model: eqx.nn.MLP, x: Float[Array, "batch 2"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = vmap(model, out_axes=1)(x).squeeze()
    return -jnp.mean(y*jnp.log(pred_y) + (1-y)*jnp.log(1-pred_y))
    # return -jnp.mean(y*jnp.log(pred_y))


@eqx.filter_jit
def make_step(
    model: eqx.nn.MLP,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Float[Array, "batch 2"],
    y: Int[Array, " batch"],
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def train_b2lp(model, dataloader) -> eqx.nn.MLP:
    optim = optax.adam(3e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for (xs, ys) in dataloader:
        model, opt_state, _ = make_step(model, optim, opt_state, xs, ys)
    return model


def test_model(model, alpha, N_TEST=int(1e4)):
    """Does the model ensemble correctly classify the outer circle?"""
    ts_ = jnp.linspace(0, 2*jnp.pi, N_TEST)
    outer = alpha*jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    preds = vmap(model)(outer)
    return (preds < 0.5).any()


# dataloader
def dataloader(key, xs, ys, batch_size=64, n_epochs=-1):
    permkey, _ = jrand.split(key)
    perm = jrand.permutation(permkey, len(xs))
    xs, ys = xs[perm], ys[perm]
    total = len(xs) // batch_size + 1
    idxs = cycle(range(total+1))
    for _, i in zip(range(total*n_epochs), idxs):
        xout = xs[i*batch_size:(i+1)*batch_size]
        yout = ys[i*batch_size:(i+1)*batch_size]
        yield xout, yout


# plotting util
def plot_decision(model, xinner, xouter, a):
    # plot decision boundary
    pts = jnp.linspace(-2.5, 2.5, 100)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # prob of being in inner circle
    preds = vmap(vmap(model))(pts)[..., 0, 0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    contourf = ax.contourf(xv, yv, preds, levels=10, vmin=0, vmax=1)
    contour = ax.contour(xv, yv, preds, levels=[0, 0.5, 1], vmin=0, vmax=1, colors='red')
    # also plot the best polygon
    N = int(polygon(a))
    poly_points = a*jnp.exp(1j*jnp.linspace(0, 2*jnp.pi, N+1))
    ptsx, ptsy = -poly_points.real, poly_points.imag
    ax.plot(ptsx, ptsy, label=f"optimal polygon ({N})")
    # colormap
    ax.clabel(contour, inline=1, fontsize=10, zorder=6)
    ax.scatter(*(xouter.T), marker='x', color='orange', alpha=.5, label='outer')
    ax.scatter(*(xinner.T), marker='o', color='orange', alpha=.5, label='inner')
    ax.set_title(f"Decision boundary, 2-layer MLP")
    cbar = plt.colorbar(contourf)
    cbar.ax.set_ylabel("inner circle class. probability")
    cbar.add_lines(contour)
    plt.legend()
    return fig, ax


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    rng, shufkey, mkey = jrand.split(rng, num=3)
    # params
    ALPHA_ = jnp.linspace(0, ALPHA_MAX := jnp.log(1.1), (N_ALPHA := 20)+1)[1:]
    ALPHA_ = jnp.exp(ALPHA_)
    BS = 128
    EPOCHS = 50
    N_MODS = 5
    N_MAX = 5000
    N_ = []
    # arch params
    arch_kwargs = dict(
        in_size=2,
        out_size=1,
        width_size=4096,
        n_mods=N_MODS,
        )

    #
    hyps = [55, 55, 55] + [28]*17
    get_loader = partial(dataloader, key=shufkey, batch_size=BS, n_epochs=EPOCHS)
    for a, n in tqdm(zip(ALPHA_, hyps), total=N_ALPHA):
        is_some_error = True
        while (n < N_MAX) and (is_some_error):
            n += 100
            xs, ys = get_points(N=n, alpha=a)
            loader = get_loader(xs=xs, ys=ys)
            model = Batch2LP(**arch_kwargs, key=mkey)
            model = train_b2lp(model, loader)
            # perform test on trained model
            is_some_error = test_model(model, alpha=a)
            print(f"With {n} points: {is_some_error}")
        N_.append(n)
        # fig, ax = plot_decision(model, xs[:n], xs[n:], a)
        # plt.savefig(f"/Users/perina1/concept_geometry/imgs/nva/{a:.3f}_{n}.png")
        # plt.close()
    # take sum across the vmapped evaluate_ensemble
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(ALPHA_, N_, '-o', label='empirical')
    XS = jnp.linspace(1, jnp.exp(ALPHA_MAX), 100)
    plt.step(XS, polygon(XS), label='polygon bound')
    plt.step(XS, spherepack(XS), label='spack bound')
    plt.title(f"Number of points for perfect separation vs. ratio of the radii (w={arch_kwargs['width_size']})")
    plt.xlabel(r"$R_M/r_m$")
    plt.ylabel("N")
    plt.legend()
    plt.show()
