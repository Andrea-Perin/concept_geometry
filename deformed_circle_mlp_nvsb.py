"""
Training MLPs on a deformed circle task.
"""
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from jax import numpy as jnp, vmap, jit, random as jrand, nn as jnn
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import subprocess
import json


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


def get_deformer(key, beta):
    """Create a linear transformation with determinant 1 and eigenvalues beta
    and 1/beta respectively (beta > 1)."""
    # eig_max = beta*eig_min, eig_max
    eigs = jnp.array([beta, 1/beta])
    P = jrand.normal(key, shape=(2, 2))
    return jnp.linalg.inv(P)@jnp.diag(eigs)@P


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


def error_fraction(model, alpha, N_TEST=int(1e4), transf=None):
    """What fraction of the inner/outer circle is misclassified?"""
    ts_ = jnp.linspace(0, 2*jnp.pi, N_TEST)
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha*jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = jnp.einsum('ji,nj->ni', transf, outer)
    inner = jnp.einsum('ji,nj->ni', transf, inner)
    preds_inn = vmap(model)(inner)
    preds_out = vmap(model)(outer)
    return (preds_inn > 0.5).mean(), (preds_out < 0.5).mean()


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    rng, shufkey, mkey, ikey = jrand.split(rng, num=4)
    # params
    ALPHA = 1.01
    BETAS = jnp.linspace(BETA_MIN := 1, BETA_MAX := 3, N_BETAS := 5)
    ERR_THRESH = 0  # 0.05
    EPOCHS = 100
    N_MAX = 10
    D = [2, ]  # 3, 8, 16]
    N_ = []
    # mlp params
    mlp_kwargs = dict(
        out_size=1,
        width_size=4096,
        depth=1,
        final_activation=jnn.sigmoid
    )

    # CREATE RESULTS FOLDER
    EXP_PARAMS = {
        'seed': SEED,
        'alpha': ALPHA,
        'betas': BETAS.tolist(),
        'err_thresold': ERR_THRESH,
        'epochs': EPOCHS,
        'n_max': N_MAX,
        'd': D,
        'mlp_kwargs': {**mlp_kwargs, 'final_activation': mlp_kwargs['final_activation'].__name__},
    }
    dt = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    process = subprocess.Popen(
        ['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip().decode('utf-8')
    METADATA = {
        'datetime': dt,
        'git_head_hash': git_head_hash,
    }
    JSON_DATA = {'exp_params': EXP_PARAMS, 'meta': METADATA}
    # create directory
    fname = f'{Path(__file__).stem}_{dt}'
    tmpname = 'TMP_' + fname
    experiment_dirname = Path('./experiments') / tmpname
    experiment_dirname.mkdir(exist_ok=True, parents=True)
    print(f"Saving files to {experiment_dirname}")
    # save experiment data on file
    with (experiment_dirname / "params.json").open("w", encoding="UTF-8") as target:
        json.dump(JSON_DATA, target, indent=4)

    # bisection scheme for finding N at every ALPHA_
    ND_ = []
    EPS_ = {d: {} for d in D}
    for d in tqdm(D):
        ND_.append([])
        for b in tqdm(BETAS):
            b = float(b)
            # create the deformer
            deformer_core = get_deformer(ikey, b)
            deformer = jnp.concatenate((deformer_core, jnp.zeros((2, d-2))), axis=1)
            EPS_[d][b] = {}
            n_lo, n, n_hi = 0, N_MAX//2, N_MAX
            while True:
                n = (n_lo + n_hi) // 2
                print(f"Currently trying n={n} (nlo={n_lo}, nhi={n_hi})")
                # create dataset and train the model
                xs, ys = get_points(N=n, alpha=ALPHA)
                xs = jnp.einsum('ji,nj->ni', deformer, xs)
                model = eqx.nn.MLP(**mlp_kwargs, in_size=d, key=mkey)
                model = train_model(model, (xs, ys), n_epochs=EPOCHS)
                # perform test on trained model
                eps_inn, eps_out = error_fraction(model, alpha=ALPHA, transf=deformer)
                EPS_[d][b][n] = {'inn': float(eps_inn), 'out': float(eps_out)}
                print(f"Error fraction: inn={eps_inn:.3f}, out={eps_out:.3f}")
                n_is_suff = (eps_inn <= ERR_THRESH) and (eps_out <= ERR_THRESH)
                # exit condition
                if (n == n_lo) and (not n_is_suff):
                    n += 1
                    break
                if (n == n_hi) and (n_is_suff):
                    break
                # update n_lo and n_hi
                n_hi = n if n_is_suff else n_hi
                n_lo = n_lo if n_is_suff else n
            ND_[-1].append(n)

    # save the relevant things
    with (experiment_dirname / "eps.json").open("w", encoding="UTF-8") as out:
        json.dump(EPS_, out, indent=4)
    with (experiment_dirname / "nd.json").open("w", encoding="UTF-8") as out:
        json.dump(ND_, out, indent=4)

    # take sum across the vmapped evaluate_ensemble
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    for d, nd in zip(D, ND_):
        ax.plot(BETAS, nd, '-o', markersize=3, label=f'empirical, dim={d}', alpha=.5)
    XS = jnp.linspace(BETA_MIN, BETA_MAX, 1000)
    ax.plot(XS, polygon(XS), label='polygon bound')
    ax.set_title(f"N vs beta, eps={ERR_THRESH}, W={mlp_kwargs['width_size']}")
    ax.set_xlabel(r"$\lambda_M/\lambda_m$")
    ax.set_ylabel("N")
    fig.legend()
    plt.savefig(experiment_dirname / 'nvsb.png')

    # plot the errors
    def plot_error_fracs(err_dict, D):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('error fraction')
        ax.set_title(f'Errors vs. N (dim={D})')
        betas = sorted(err_dict.keys())
        cmap = mpl.colormaps['viridis']
        for idx, b in enumerate(betas):
            sorted_ns = sorted(err_dict[b].keys())
            inns = [err_dict[b][s]['inn'] for s in sorted_ns]
            outs = [err_dict[b][s]['out'] for s in sorted_ns]
            col = cmap((idx+1)/len(betas))
            ax.plot(sorted_ns, inns, linestyle='-', color=col, alpha=0.5)
            ax.plot(sorted_ns, outs, linestyle='-.', color=col, alpha=0.5)
        norm_colors = mpl.colors.Normalize(vmin=betas[0], vmax=betas[-1])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_colors)
        cb = fig.colorbar(sm, cax=ax)
        cb.set_label('alphas')
        line_inner = mpl.lines.Line2D(
            [0], [0], label='inner circle error', linestyle='-', color='black')
        line_outer = mpl.lines.Line2D(
            [0], [0], label='outer circle error', linestyle='-.', color='black')
        ax.legend(handles=[line_inner, line_outer])
        return fig, ax

    for d in EPS_:
        fig, ax = plot_error_fracs(EPS_[d], d)
        plt.tight_layout()
        plt.savefig(experiment_dirname / f'errors_{d}.png')

    # LAST THING: SET THE EXPERIMENT AS COMPLETE
    experiment_dirname.rename(Path('./experiments') / fname)
