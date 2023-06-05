"""
Training an MLP on a pair of random curves.
"""
import json
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from jax import numpy as jnp, vmap, jit, random as jrand, nn as jnn
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pathlib import Path
from random_loop import get_polar_loop
from expman import ExpLogger
from numpy import empty


# sample points uniformly from a circle
def get_points(key, N, alpha, **kwargs):
    inner = get_polar_loop(key, N, **kwargs)
    outer = alpha * inner
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


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    rng, mkey, dkey = jrand.split(rng, num=3)
    # params
    ALPHA = [1.5, 1.2, 1.15, 1.1, 1.075, 1.05, 1.025, 1.01, 1.001]
    ERR_THRESHOLD = 0  #.05
    EPOCHS = 1000
    N_MAX = 1000
    N_TEST = int(1e4)
    N = empty(len(ALPHA))
    # mlp params
    mlp_kwargs = dict(
        out_size=1,
        width_size=2048,
        depth=1,
        final_activation=jnn.sigmoid
        )

    with ExpLogger() as experiment:
        # save experiment params
        mlp_act = {'final_activation': mlp_kwargs['final_activation'].__name__}
        PARAMS_DICT = {
                'seed': SEED,
                'alpha': ALPHA,
                'err_threshold': ERR_THRESHOLD,
                'epochs': EPOCHS,
                'n_max': N_MAX,
                'n_test': N_TEST,
                # mlp params
                'mlp_params': {**mlp_kwargs, **mlp_act},
                }
        experiment.save_dict(PARAMS_DICT, 'params.json')

        # run experiment
        EPS_ = {a: {} for a in ALPHA}
        for ida, a in enumerate(ALPHA):
            n_lo, n, n_hi = 0, N_MAX//2, N_MAX
            while True:
                n = (n_lo + n_hi) // 2
                # create dataset and train the model
                xs, ys = get_points(key=dkey, N=n, alpha=a)
                model = eqx.nn.MLP(**mlp_kwargs, in_size=2, key=mkey)
                model = train_model(model, (xs, ys), n_epochs=EPOCHS)
                # perform test on trained model, save errors
                xs_test, ys_test = get_points(key=dkey, N=N_TEST, alpha=a)
                eps_inn = (vmap(model)(xs_test[:N_TEST]) > .5).mean()
                eps_out = (vmap(model)(xs_test[N_TEST:]) < .5).mean()
                EPS_[a][n] = {'inn': float(eps_inn), 'out': float(eps_out)}
                print(f"n:{n},nlo:{n_lo},nhi:{n_hi},inn:{eps_inn:.3f},out:{eps_out:.3f}")
                n_is_sufficient = (eps_inn <= ERR_THRESHOLD) and (eps_out <= ERR_THRESHOLD)
                # exit condition
                if (n == n_lo) and (not n_is_sufficient):
                    n += 1
                    break
                if (n == n_hi) and (n_is_sufficient):
                    break
                # update n_lo and n_hi
                n_hi = n if n_is_sufficient else n_hi
                n_lo = n_lo if n_is_sufficient else n
            N[ida] = n

        # take sum across the vmapped evaluate_ensemble
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(ALPHA, N, label='experimental')
        plt.title(f"N vs alpha, eps={ERR_THRESHOLD}, W={mlp_kwargs['width_size']}")
        plt.xlabel(r"$R_M/r_m$")
        plt.ylabel("N")
        plt.legend()
        plt.savefig(experiment/'nvsa.png')

        # plot the errors
        def plot_error_fracs(err_dict):
            fig, ax = plt.subplots()
            ax.set_xscale('log')
            ax.set_xlabel('N')
            ax.set_ylabel('error fraction')
            ax.set_title('Errors vs. N')
            alphas = sorted(err_dict.keys())
            cmap = mpl.colormaps['viridis']
            for idx, a in enumerate(alphas):
                sorted_ns = sorted(err_dict[a].keys())
                inns = [err_dict[a][s]['inn'] for s in sorted_ns]
                outs = [err_dict[a][s]['out'] for s in sorted_ns]
                col = cmap((idx+1)/len(alphas))
                ax.plot(sorted_ns, inns, linestyle='-', color=col, alpha=0.5)
                ax.plot(sorted_ns, outs, linestyle='-.', color=col, alpha=0.5)
            norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            cb = fig.colorbar(sm, ax=ax)
            cb.set_label('alphas')
            lcargs = dict(linestyle='-', color='black')
            linner = mpl.lines.Line2D([0], [0], label='inner error', **lcargs)
            louter = mpl.lines.Line2D([0], [0], label='outer error', **lcargs)
            ax.legend(handles=[linner, louter])
            return fig, ax

        fig, ax = plot_error_fracs(EPS_)
        plt.savefig(experiment/'errors.png')
