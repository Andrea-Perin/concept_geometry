"""Trying out learning curves on a sine curve on a circle."""
from tqdm import tqdm
import jax
from jax import numpy as jnp, random as jrand, vmap, nn as jnn
from numpy import zeros
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib as mpl

from dataset_utils import dataloader

from expman import ExpLogger


def plot_decision(model, pts_inn, pts_out, npts=100, mult=2):
    """plotting function for the decision boundary of a 2D MLP"""
    # plot decision boundary
    pts = jnp.linspace(lo := -mult, hi := mult, npts)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # prob of being in inner circle
    preds = vmap(vmap(model))(pts).squeeze()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    tit = f"Decision boundary, N={pts_inn.shape[0]}"
    ax.set_title(tit)
    # plot contours
    avicii = jnp.linspace(0, 1, 11)
    contourf = ax.contourf(xv, yv, preds, levels=avicii, vmin=0, vmax=1)
    contour = ax.contour(xv, yv, preds, levels=avicii,
                         vmin=0, vmax=1, colors='red', alpha=0.5)
    # plot scatter points
    ax.scatter(*pts_inn.T, marker='x', color='black')
    ax.scatter(*pts_out.T, marker='o', color='black')
    # colormap
    ax.clabel(contour, inline=True, fontsize=10, zorder=6)
    cbar = plt.colorbar(contourf)
    cbar.ax.set_ylabel("class. probability")
    cbar.add_lines(contour)
    return fig, ax


def sine_on_circle(N, f: int, a: float = .25, *, key=None):
    """It is not really a sine wave on a circle, but close enough"""
    if key is None:
        ts = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    else:
        ts = jrand.uniform(key, shape=(N,)) * (2*jnp.pi)
    r = 1 + a*jnp.sin(f*ts)
    return jnp.stack((r*jnp.cos(ts), r*jnp.sin(ts)), axis=1)


# training function
@eqx.filter_value_and_grad
def loss(model, x, y):
    pred_y = vmap(model)(x).squeeze()
    return -jnp.mean(y*jnp.log(pred_y) + (1-y)*jnp.log(1-pred_y))


@eqx.filter_jit
def make_step(model, opt_state, x, y):
    loss_value, grads = loss(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def train_model(model, dloader, optimizer):
    # record model's state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    for (xs, ys) in dloader:
        model, opt_state, loss = make_step(model, opt_state, xs, ys)
        losses.append(float(loss))
    return model, losses


# the whole thing
with ExpLogger() as experiment:
    # experiment parameters
    PARAMS = dict(
        seed=int(input("Insert random seed (default: 0): ") or "0"),
        # NETWORK ARCHITECTURE
        arch={'in_size': (D := 2),
              'out_size': 1,
              'width_size': 256,
              'depth': 2,
              'final_activation': jnn.sigmoid},
        # OPTIMIZER STUFF
        schedule=optax.warmup_cosine_decay_schedule,
        schedule_params={'init_value': 1e-4,
                         'peak_value': 5e-3,
                         'warmup_steps': 50,
                         'end_value': 1e-6,
                         'decay_steps': 1000, },
        clipper=optax.clip,
        clipper_params={'max_delta': 2.0},
        optimizer=optax.adam,
        optimizer_params={},
        # DLOADER PARAMS
        dloader_params={'batch_size': 2048,
                        'n_epochs': 5000, },
        # EXPERIMENTAL PARAMS
        # alphas=[1.0001, 1.0005, 1.001, 1.002, 1.003, 1.004, 1.005],
        alphas=[1.1,],
        freqs=[4, 5, 6, 7],
        ns=jnp.unique(jnp.logspace(NMIN:=.5, NMAX:=2, NN:=10).astype(int)).tolist(),
        # OTHER
        n_test=int(1e4),
    )

    # some aliases for shorter code
    N_TEST = PARAMS['n_test']
    FREQS = PARAMS['freqs']
    ALPHAS = PARAMS['alphas']
    ENN = PARAMS['ns']
    WIDTH = PARAMS['arch']['width_size']

    # prepare the storage of results
    res_shape = (len(ALPHAS), len(FREQS), len(ENN))
    RESULTS = dict(
        ERRORS=zeros((*res_shape, 4)),
        LOSSES=[],
    )

    # prepare the optimizer
    schedule = PARAMS['schedule'](**PARAMS['schedule_params'])
    opt = PARAMS['optimizer'](**PARAMS['optimizer_params'], learning_rate=schedule)
    clipper = PARAMS['clipper'](**PARAMS['clipper_params'])
    optimizer = optax.chain(clipper, opt)

    # ===============
    # EXPERIMENT CODE
    # ===============
    # start the RNG
    rng = jrand.PRNGKey(PARAMS['seed'])
    dkey, skey, mkey, tkey = jrand.split(rng, num=4)
    # start looping over alphas, freqs and Ns
    for ida, a in enumerate(tqdm(ALPHAS)):
        RESULTS['LOSSES'].append([])
        for idf, f in enumerate(tqdm(FREQS)):
            RESULTS['LOSSES'][-1].append([])
            for idn, n in enumerate(tqdm(ENN)):
                # create dataset
                inn = sine_on_circle(n, f, key=None)
                out = inn * a
                labs = jnp.repeat(jnp.arange(2), n)
                dset = (jnp.concatenate((inn, out)), labs)
                dload = dataloader(dset, **PARAMS['dloader_params'], skey=skey)
                # train the model on the data
                model = eqx.nn.MLP(**PARAMS['arch'], key=mkey)
                model, losses = train_model(model, dload, optimizer)
                # perform test on trained model, save errors
                test_inn = sine_on_circle(N_TEST, f)
                test_out = test_inn * a
                test_set = jnp.concatenate((test_inn, test_out))
                err_inn = float((vmap(model)(test_inn) > .5).mean())
                err_out = float((vmap(model)(test_out) < .5).mean())
                # store errors of training dataset
                train_err_inn = float((vmap(model)(inn) > .5).mean())
                train_err_out = float((vmap(model)(out) < .5).mean())
                # store results
                results = (err_inn, err_out, train_err_inn, train_err_out)
                RESULTS['ERRORS'][ida, idf, idn] = results
                RESULTS['LOSSES'][-1].append(losses)
                # plot decision boundary
                fig, ax = plot_decision(model, inn, out)
                plt.savefig(experiment / f'decision_{a:.2f}_{f}_{n}.png')
                plt.close()

    # some plotting
    def plot_errors(errors):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Error fraction')
        cmap = mpl.colormaps.get_cmap('cividis')
        colors = [cmap(i) for i in jnp.linspace(0, 1, len(errors))]
        inn, out = errors[..., 0], errors[..., 1]
        for e_in, e_out, c in zip(inn, out, colors):
            ax.plot(ENN, e_in, '-', c=c)
            ax.plot(ENN, e_out, '--', c=c)
        # for colorbar
        normer = plt.Normalize(vmin=FREQS[0], vmax=FREQS[-1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normer)
        fig.colorbar(sm, ax=ax, label='frequency')
        # for legend
        lcargs = dict(linestyle='-', color='black')
        linner = mpl.lines.Line2D([0], [0], label='inner error', **lcargs)
        lcargs = dict(linestyle='--', color='black')
        louter = mpl.lines.Line2D([0], [0], label='outer error', **lcargs)
        ax.legend(handles=[linner, louter])
        return fig, ax

    fig, ax = plot_errors(RESULTS['ERRORS'][0, ..., :2])
    plt.savefig(experiment / 'errors_test.png')
    plt.close()
    fig, ax = plot_errors(RESULTS['ERRORS'][0, ..., 2:])
    plt.savefig(experiment / 'errors_train.png')
    plt.close()

    # and conclude by saving the RESULTS
    RESULTS['ERRORS'] = RESULTS['ERRORS'].tolist()
    experiment.save_dict(RESULTS, 'results.json')
    # modify the PARAMS dict to make it saveable
    PARAMS['arch']['final_activation'] = PARAMS['arch']['final_activation'].__name__
    PARAMS['schedule'] = PARAMS['schedule'].__name__
    PARAMS['clipper'] = PARAMS['clipper'].__name__
    PARAMS['optimizer'] = PARAMS['optimizer'].__name__
    # NOW save it :3
    experiment.save_dict(PARAMS, 'params.json')
