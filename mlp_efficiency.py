import numpy as np
from jax import numpy as jnp, random as jrand, vmap, jit, nn as jnn
from jaxtyping import Array, Float, Int, PyTree
import equinox as eqx
import optax
from itertools import cycle
from functools import partial
import matplotlib.pyplot as plt

import os
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1 "
                           "inter_op_parallelism_threads=1")


def dataloader(key, xs, ys, batch_size):
    _, perm_key = jrand.split(key)
    total = xs.shape[0]
    order = jrand.permutation(key=perm_key, x=total)
    pxs, pys = xs[order], ys[order]
    idxs = cycle(range(0, total//batch_size))
    for i in idxs:
        xout = pxs[i*batch_size: (i+1)*batch_size]
        yout = pys[i*batch_size: (i+1)*batch_size]
        yield xout, yout


def spherical_uniform(key, d, shape=(1,)):
    """
    Generate samples from the uniform N dimensional spherical distribution.
    """
    # generate from multivariate normal
    z = jrand.normal(key=key, shape=(*shape, d))
    norm = jnp.sqrt((z**2).sum(axis=-1, keepdims=True))
    return z/norm


def get_datasets(key, n, alpha):
    ki, ko = jrand.split(key)
    x_inner = spherical_uniform(key=ki, d=2, shape=(n,))
    x_outer = spherical_uniform(key=ko, d=2, shape=(n,))
    x_outer *= alpha
    y_inner = jnp.zeros((n,), dtype=int)
    y_outer = jnp.ones((n,), dtype=int)
    # and create an iterable dataset from them
    xs = jnp.concatenate((x_inner, x_outer))
    ys = jnp.concatenate((y_inner, y_outer))
    return xs, ys


# TRAINING TOOLS
def loss(
    model: eqx.nn.MLP, x: Float[Array, "batch 2"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 2"]
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


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


def train_model(loader, key, model_specs, lr, steps) -> eqx.nn.MLP:
    mlp = eqx.nn.MLP(key=key, **model_specs)
    # actual training
    optim = optax.adam(learning_rate=lr)
    opt_state = optim.init(eqx.filter(mlp, eqx.is_array))
    for step, (x, y) in zip(range(steps), loader):
        mlp, opt_state, train_loss = make_step(mlp, optim, opt_state, x, y)
    return mlp


# EVAL TOOLS
def error_prob(model, alpha, ntest=int(1e3)):
    ts = jnp.linspace(0, 2*jnp.pi, ntest)
    test_xs = alpha*jnp.stack((jnp.cos(ts), jnp.sin(ts)), axis=1)
    preds = jnp.exp(vmap(model)(test_xs))[:, 0]
    return jnp.mean(preds > .5)


# plotting util
def plot_decision(model, xinner, xouter, xmin=-2, xmax=2):
    # plot decision boundary
    mult = 1.2
    pts = jnp.linspace(mult*xmin, mult*xmax, 100)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # prob of being in inner circle
    preds = jnp.exp(vmap(vmap(model))(pts)[..., 0])
    fig, ax = plt.subplots()
    contourf = ax.contourf(xv, yv, preds, levels=10, vmin=0, vmax=1)
    contour = ax.contour(xv, yv, preds, levels=1, vmin=0, vmax=1, colors='red')
    ax.clabel(contour, inline=1, fontsize=10, zorder=6)
    ax.scatter(*(xouter.T), marker='x', color='orange', label='outer')
    ax.scatter(*(xinner.T), marker='o', color='orange', label='inner')
    ax.set_title(f"Decision boundary, 2-layer MLP, width={model.width_size}")
    cbar = plt.colorbar(contourf)
    cbar.ax.set_ylabel("inner circle class. probability")
    cbar.add_lines(contour)
    plt.legend()
    return fig, ax


# if __name__ == "__main__":
#     SEED = int(input("Insert seed: "))
#     rng = jrand.PRNGKey(SEED)
#     # PARAMS
#     ALPHA_ = jnp.linspace(1, 10, (N_ALPHAS := 20)+1)[1:]  # ratios between radii
#     N_PTS_ = jnp.arange(3, 30, 2)  # number of points per sphere
#     BATCH_SIZE = 10  # so few points that it is worth doing them all in one go
#     LEARNING_RATE = 3e-3
#     STEPS = int(50)
#     N_EXPS = 50
#     # ARCHITECTURE DETAILS
#     arch_kwargs = dict(
#         in_size=2,
#         out_size=2,
#         width_size=16,  # should be enough
#         depth=1,
#         final_activation=jnn.log_softmax)
#     # rng manip
#     rng, dkey, lkey, mkey = jrand.split(rng, num=4)
#     modkeys = jrand.split(mkey, num=N_EXPS)
#     datakeys = jrand.split(dkey, num=N_EXPS)
#     # some partials to keep the code clean
#     ploader = partial(dataloader, key=lkey, batch_size=BATCH_SIZE)
#     trainer = partial(train_model, model_specs=arch_kwargs, lr=LEARNING_RATE, steps=STEPS)
#     # loop over parameters
#     errors = np.empty((N_ALPHAS, len(N_PTS_)))
#     stds = np.empty((N_ALPHAS, len(N_PTS_)))
#     for ida, alpha in enumerate(ALPHA_):
#         for idn, n in enumerate(N_PTS_):
#             # generate datasets and dataloaders
#             xsys = [get_datasets(key=dk, n=n, alpha=alpha) for dk in datakeys]
#             dloaders = [ploader(xs=x, ys=y) for (x, y) in xsys]
#             # train an mlp on each experiment
#             mlps = [trainer(loader=l, key=k) for l, k in zip(dloaders, modkeys)]
#             # perform experiment to find error probability
#             ers = jnp.array([error_prob(m, alpha) for m in mlps])
#             errors[ida, idn] = jnp.mean(ers)
#             stds[ida, idn] = jnp.std(ers)
#
#     # VIZ AN MLP
#     # model = mlps[0]
#     # plot_decision(model,  xsys[0][0][:10], xsys[0][0][10:], xmin=-10, xmax=10)
#     # plt.show()
#
#     # VIZ THE ERROR AS FUNCTION OF ALPHA
#     # plt.plot(ALPHA_, errors.squeeze(), label='empirical')
#     # plt.title(f"Error probability on outer circle vs radius ratio (N={N_PTS_[0]}, width={arch_kwargs['width_size']})")
#     # plt.xlabel("alpha")
#     # plt.ylabel("error probablity on outer circle")
#     # plt.legend()
#     # plt.show()
#
#     # # VIZ THE ERROR AS A FUNCTION OF N
#     # ys = errors[0].squeeze()
#     # dys = stds[0].squeeze()
#     # plt.plot(N_PTS_, ys, label='empirical')
#     # plt.fill_between(N_PTS_, ys-dys, ys+dys, alpha=.2)
#     # plt.title(f"Error probability on outer circle vs radius ratio (alpha={ALPHA_[0]}, width={arch_kwargs['width_size']})")
#     # plt.xlabel("N")
#     # plt.ylabel("error probablity on outer circle")
#     # plt.legend()
#     # plt.show()


if __name__ == "__main__":
    SEED = int(input("Insert seed: "))
    rng = jrand.PRNGKey(SEED)
    # PARAMS
    ALPHA_ = jnp.linspace(1, 2, (N_ALPHAS := 10)+1)[1:]  # ratios between radii
    ERR_THRESH = .1
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-3
    STEPS = int(50)
    N_EXPS = 25
    N_MAX = 50
    # ARCHITECTURE DETAILS
    arch_kwargs = dict(
        in_size=2,
        out_size=2,
        width_size=16,  # should be enough
        depth=1,
        final_activation=jnn.log_softmax)
    # rng manip
    rng, dkey, lkey, mkey = jrand.split(rng, num=4)
    modkeys = jrand.split(mkey, num=N_EXPS)
    datakeys = jrand.split(dkey, num=N_EXPS)
    # some partials to keep the code clean
    ploader = partial(dataloader, key=lkey, batch_size=BATCH_SIZE)
    trainer = partial(train_model, model_specs=arch_kwargs, lr=LEARNING_RATE, steps=STEPS)
    # loop over parameters
    NS = np.empty_like(ALPHA_)
    for ida, alpha in enumerate(ALPHA_):
        print(f"Doing alpha = {alpha}")
        err = jnp.inf
        std = 1
        n = 3
        # while jnp.abs((err - ERR_THRESH) / std) > 1:
        while (err > ERR_THRESH) and (n<N_MAX):
            # generate datasets and dataloaders
            xsys = [get_datasets(key=dk, n=n, alpha=alpha) for dk in datakeys]
            dloaders = [ploader(xs=x, ys=y) for (x, y) in xsys]
            # train an mlp on each experiment
            mlps = [trainer(loader=l, key=k) for l, k in zip(dloaders, modkeys)]
            # perform experiment to find error probability
            ers = jnp.array([error_prob(m, alpha) for m in mlps])
            err = jnp.mean(ers)
            std = jnp.std(ers)
            n += 1
        NS[ida] = n
    # viz
    plt.plot(ALPHA_, NS)
    plt.xlabel("alpha")
    plt.ylabel("N")
    plt.title(f"Number of samples to get an average outer circle less than {ERR_THRESH}")
    plt.show()
