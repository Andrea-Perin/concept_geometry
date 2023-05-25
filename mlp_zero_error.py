"""
Specifically training MLPs on the circle task.
Aiming for 0 error as a function of circle separation, we look at the required
number of samples needed.
"""
from itertools import cycle
from functools import partial
from jax import numpy as jnp, random as jrand, vmap, jit
import jax.nn as jnn
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from typing import Optional







# sample points uniformly from a circle
def get_points(N, alpha):
    ts_ = jnp.linspace(0, 2*jnp.pi, N)
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha * jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    pts = jnp.concatenate((inner, outer))
    labs = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return pts, labs



# TRAINING TOOLS
def loss(
    model: eqx.nn.MLP, x: Float[Array, "batch 2"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = vmap(model)(x)
    return -jnp.mean(y*pred_y)


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


def train_ensemble(model_ensemble, dataloader) -> eqx.nn.MLP:
    # define the forward function
    # actual training
    optim = optax.adam(3e-3)
    opt_state = optim.init(eqx.filter(model_ensemble, eqx.is_array))
    return model_ensemble


# dataloader
def dataloader(key, xs, ys, batch_size=64, n_epochs=-1):
    permkey = jrand.split(key, num=1)
    perm = jrand.permutation(permkey, len(xs))
    xs, ys = xs[perm], ys[perm]
    idxs = cycle(range(len(xs)//batch_size + 2))
    n = 0
    while (n < n_epochs) or (n_epochs < 0):
        for i in idxs:
            xout = xs[i*batch_size:(i+1)*batch_size]
            yout = ys[i*batch_size:(i+1)*batch_size]
            yield xout, yout
        n += 1




if __name__ == "__main__":
    SEED = int(input("Insert seed: "))
    rng = jrand.PRNGKey(SEED)
    rng, shufkey, mkey = jrand.split(rng, num=3)
    # params
    ALPHA_ = jnp.linspace(1, ALPHA_MAX := 2, (N_ALPHA := 10)+1)[1:]
    BS = 64
    EPOCHS = 50
    N_MODS = 20
    N_MAX = 100
    N_ = []
    # arch params
    arch_kwargs = dict(
        in_size=2,
        out_size=1,
        width_size=16,
        depth=1,
        final_activation=jnn.sigmoid
        )

    # ensembler
    @eqx.filter_vmap
    def make_ensemble(key):
        return eqx.nn.MLP(**arch_kwargs, key=key)

    # some black magic
    class EMLP(eqx.Module):
        mlps: eqx.nn.MLP

        def __init__(self, keys):
            self.mlps = make_ensemble(keys)

        def __call__(self, x):
            return eqx.filter_vmap(in_axes=(eqx.if_array(0), None))(self.mlps, x) 

    # model ensembling
    @eqx.filter_vmap
    def make_ensemble(key):
        return eqx.nn.MLP(**arch_kwargs, key=key)

    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
    def evaluate_ensemble(model, x):
        return model(x)


    modkeys = jrand.split(mkey, num=N_MODS)
    #
    get_loader = partial(dataloader, key=shufkey, batch_size=BS, n_epochs=EPOCHS)
    for a in ALPHA_:
        n = 3
        while n < N_MAX:
            xs, ys = get_points(n)
            loader = get_loader(xs=xs, ys=ys)
            models = make_ensemble(modkeys)
            y_pred = vmap(evaluate_ensemble, in_axes=(None, 0))(models, xs)
            print(y_pred)
            break


# take sum across the vmapped evaluate_ensemble
