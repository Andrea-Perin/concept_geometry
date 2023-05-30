"""
Specifically training MLPs on the circle task.
Only a single MLP is trained instead of an ensemble.
Aiming for 0 error as a function of circle separation, we look at the required
number of samples needed.
"""
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from jax import numpy as jnp, vmap, jit, random as jrand, nn as jnn
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pathlib import Path


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


def error_fraction(model, alpha, N_TEST=int(1e4), inf=None):
    """What fraction of the inner/outer circle is misclassified?"""
    ts_ = jnp.linspace(0, 2*jnp.pi, N_TEST)
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha*jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = jnp.einsum('ji,nj->ni', inf, outer)
    inner = jnp.einsum('ji,nj->ni', inf, inner)
    preds_inn = vmap(model)(inner)
    preds_out = vmap(model)(outer)
    return (preds_inn > 0.5).mean(), (preds_out < 0.5).mean()


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    rng, shufkey, mkey, ikey = jrand.split(rng, num=4)
    # params
    RANDOM_PROJ = False
    ALPHA_ = jnp.linspace(0, LOG_ALPHA_MAX:=jnp.log(1.1), (N_ALPHAS:=15)+1)[1:]
    ALPHA_ = jnp.exp(ALPHA_)
    ERR_THRESHOLD = 0  #.05
    EPOCHS = 2000
    N_MAX = 500
    D = [2, 3, 8, 16]
    N_ = []
    # mlp params
    mlp_kwargs = dict(
        out_size=1,
        width_size=4096,
        depth=1,
        final_activation=jnn.sigmoid
        )

    # bisection scheme for finding N at every ALPHA_
    ND_ = []
    EPS_ = {d: {} for d in D}
    for d in tqdm(D):
        # inflator to larger dim
        if RANDOM_PROJ:
            # make projection noisy
            inflator = jrand.normal(ikey, shape=(2, d))
        else:
            # just add fake dimensions
            # inflator = jnp.concatenate((jnp.eye(2), jnp.zeros((2, d-2))), axis=1)
            inflator = jnp.concatenate((jrand.orthogonal(ikey, 2), jnp.zeros((2, d-2))), axis=1)
        ND_.append([])
        for a in tqdm(ALPHA_):
            EPS_[d][float(a)] = {}
            n_lo, n, n_hi = 0, N_MAX//2, N_MAX
            while True:
                n = (n_lo + n_hi) // 2
                print(f"Currently trying n={n} (nlo={n_lo}, nhi={n_hi})")
                # create dataset and train the model
                xs, ys = get_points(N=n, alpha=a)
                xs = jnp.einsum('ji,nj->ni', inflator, xs)
                model = eqx.nn.MLP(**mlp_kwargs, in_size=d, key=mkey)
                model = train_model(model, (xs, ys), n_epochs=EPOCHS)
                # perform test on trained model
                eps_inn, eps_out = error_fraction(model, alpha=a, inf=inflator)
                del model
                EPS_[d][float(a)][n] = {'inn': float(eps_inn), 'out': float(eps_out)}
                print(f"Error fraction: inn={eps_inn:.3f}, out={eps_out:.3f}")
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
            ND_[-1].append(n)

    # take sum across the vmapped evaluate_ensemble
    plt.yscale('log')
    plt.xscale('log')
    for d, nd in zip(D, ND_):
        plt.plot(ALPHA_, nd, '-o', markersize=3, label=f'empirical, dim={d}', alpha=.5)
    XS = jnp.linspace(ALPHA_[0], ALPHA_[-1], 1000)
    plt.plot(XS, polygon(XS), label='polygon bound')
    # plt.plot(XS, spherepack(XS), label='spack bound')
    plt.title(f"N vs alpha, eps={ERR_THRESHOLD}, W={mlp_kwargs['width_size']}")
    plt.xlabel(r"$R_M/r_m$")
    plt.ylabel("N")
    plt.legend()
    plt.show()

    # plot the errors
    def plot_error_fracs(err_dict, D):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel('error fraction')
        ax.set_title(f'Errors vs. N (dim={D})')
        alphas = sorted(err_dict.keys())
        cmap = mpl.colormaps['viridis']
        for idx, a in enumerate(alphas):
            sorted_ns = sorted(err_dict[a].keys())
            inns = [err_dict[a][s]['inn'] for s in sorted_ns]
            outs = [err_dict[a][s]['out'] for s in sorted_ns]
            col = cmap((idx+1)/len(alphas))
            ax.plot(sorted_ns, inns, linestyle='-', color=col, alpha=0.5)
            ax.plot(sorted_ns, outs, linestyle='-.', color=col, alpha=0.5)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1]))
        cb = fig.colorbar(sm)
        cb.set_label('alphas')
        line_inner = mpl.lines.Line2D([0], [0], label='inner circle error', linestyle='-', color='black')
        line_outer = mpl.lines.Line2D([0], [0], label='outer circle error', linestyle='-.', color='black')
        ax.legend(handles=[line_inner, line_outer])
        return fig, ax

    imgs_dir = Path(f'./imgs/test_ortho')
    imgs_dir.mkdir(parents=True)
    for d in EPS_:
        fig, ax = plot_error_fracs(EPS_[d], d)
        plt.tight_layout()
        plt.savefig(imgs_dir / f'nvsa_{d}.png')

