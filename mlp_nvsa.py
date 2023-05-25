"""
Specifically training MLPs on the circle task.
Only a single MLP is trained instead of an ensemble.
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
    """Smallest number of sides for a regular polygon separating the two circles."""
    return jnp.ceil(jnp.pi/(jnp.arccos(1/x)))


def spherepack(x):
    """Smallest number of spheres that cover the optimal separating circle."""
    return jnp.ceil(jnp.pi/(2*jnp.arcsin((x-1)/(2*(x+1)))))


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


def train_model(model, data, n_epochs=1000) -> eqx.nn.MLP:
    xs, ys = data
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for _ in range(n_epochs):
        model, opt_state, _ = make_step(model, optim, opt_state, xs, ys)
    return model


def test_model(model, alpha, N_TEST=int(1e5), inf=None):
    """Does the model ensemble correctly classify the outer circle?"""
    ts_ = jnp.linspace(0, 2*jnp.pi, N_TEST)
    outer = alpha*jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = jnp.einsum('ji,nj->ni', inf, outer)
    preds = vmap(model)(outer)
    return (preds < 0.5).any()


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    rng, shufkey, mkey, ikey = jrand.split(rng, num=4)
    # params
    ALPHA_ = jnp.linspace(0, ALPHA_MAX := jnp.log(1.1), (N_ALPHA := 100)+1)[1:]
    ALPHA_ = jnp.exp(ALPHA_)
    EPOCHS = 1000
    N_MAX = 5000
    D = 3
    N_ = []
    # mlp params
    mlp_kwargs = dict(
        in_size=D,
        out_size=1,
        width_size=4096,
        depth=1,
        final_activation=jnn.sigmoid
        )
    mlp = eqx.nn.MLP(**mlp_kwargs, key=mkey)

    # inflator to larger dim
    inflator = jrand.normal(ikey, shape=(2, D))

    # A bit of efficiency: let us start from the polygon hypothesis
    hyps = jnn.relu(polygon(ALPHA_)-15).astype(int)
    for a, n in tqdm(zip(ALPHA_, hyps), total=N_ALPHA):
        print(f"Doing {a}")
        is_some_error = True
        while (n < N_MAX) and (is_some_error):
            n += 1
            xs, ys = get_points(N=n, alpha=a)
            xs = jnp.einsum('ji,nj->ni', inflator, xs)
            model = eqx.nn.MLP(**mlp_kwargs, key=mkey)
            model = train_model(model, (xs, ys), n_epochs=EPOCHS)
            # perform test on trained model
            is_some_error = test_model(model, alpha=a, inf=inflator)
            print(f"With {n} points: {is_some_error}")
        N_.append(n)

    # take sum across the vmapped evaluate_ensemble
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(ALPHA_, N_, '-o', markersize=3, label='empirical')
    plt.step(ALPHA_, polygon(ALPHA_), label='polygon bound')
    plt.step(ALPHA_, spherepack(ALPHA_), label='spack bound')
    plt.title(f"N vs alpha, W={mlp_kwargs['width_size']}, 3D")
    plt.xlabel(r"$R_M/r_m$")
    plt.ylabel("N")
    plt.legend()
    plt.show()
