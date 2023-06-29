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


class PMLP(eqx.Module):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array

    def __init__(self, in_size, out_size, width_size, n, key):
        width = width_size
        # w1k, b1k, w2k, b2k, w3k, b3k = jrand.split(key, 6)
        keys = jrand.split(key, 3)
        w1k, b1k = jrand.split(keys[0], 2)
        w2k, b2k = jrand.split(keys[1], 2)
        w3k, b3k = jrand.split(keys[2], 2)
        lim1 = 1 / jnp.sqrt(in_size)
        self.w1 = jrand.uniform(w1k, (n, width, in_size), minval=-lim1, maxval=lim1)
        self.b1 = jrand.uniform(b1k, (n, width), minval=-lim1, maxval=lim1)
        lim2 = 1 / jnp.sqrt(width)
        self.w2 = jrand.uniform(w2k, (n, width, width), minval=-lim2, maxval=lim2)
        self.b2 = jrand.uniform(b2k, (n, width), minval=-lim2, maxval=lim2)
        lim3 = 1 / jnp.sqrt(width)
        self.w3 = jrand.uniform(w3k, (n, out_size, width), minval=-lim3, maxval=lim3)
        self.b3 = jrand.uniform(b3k, (n, out_size), minval=-lim3, maxval=lim3)

    def __call__(self, x):
        x = self.b1 + jnp.einsum('nwi,i->nw', self.w1, x)
        x = jnn.relu(x)
        x = self.b2 + jnp.einsum('nwv,nv->nw', self.w2, x)
        x = jnn.relu(x)
        x = self.b3 + jnp.einsum('now,nw->no', self.w3, x)
        x = vmap(jnn.sigmoid)(x)
        return x


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
def ce(y, y_pred):
    return -jnp.mean(y*jnp.log(y_pred) + (1-y)*jnp.log(1-y_pred))


@eqx.filter_value_and_grad
def loss(model, x, y):
    y_preds = vmap(model)(x).squeeze().T  # (batch size, n, fakedim) -> (n, batch_size)
    return jnp.sum(vmap(ce, in_axes=(None, 0))(y, y_preds))


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
              'n': 10,
              # 'final_activation': jnn.sigmoid
              },
        # OPTIMIZER STUFF
        schedule=optax.warmup_cosine_decay_schedule,
        schedule_params={'init_value': 1e-3,
                         'peak_value': 5e-3,
                         'warmup_steps': 50,
                         'end_value': 1e-5,
                         'decay_steps': 1000, },
        clipper=optax.clip,
        clipper_params={'max_delta': 10.0},
        optimizer=optax.sgd,
        optimizer_params={},
        # DLOADER PARAMS
        dloader_params={'batch_size': 2048,
                        'n_epochs': 5000, },
        # EXPERIMENTAL PARAMS
        # alphas=[1.0001, 1.0005, 1.001, 1.002, 1.003, 1.004, 1.005],
        alphas=[1.1,],
        freqs=[4, 5, 6, 7, 8],
        ns=jnp.unique(jnp.logspace(NMIN:=.5, NMAX:=2, NN:=20).astype(int)).tolist(),
        # OTHER
        n_test=int(1e4),
    )

    # some aliases for shorter code
    N_TEST = PARAMS['n_test']
    FREQS = PARAMS['freqs']
    ALPHAS = PARAMS['alphas']
    ENN = PARAMS['ns']
    WIDTH = PARAMS['arch']['width_size']
    PPP = PARAMS['arch']['n']

    # prepare the storage of results
    res_shape = (len(ALPHAS), len(FREQS), len(ENN), PPP)
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
                model = PMLP(**PARAMS['arch'], key=mkey)
                # model = eqx.nn.MLP(**PARAMS['arch'], key=mkey)
                model, losses = train_model(model, dload, optimizer)
                # perform test on trained model, save errors
                test_inn = sine_on_circle(N_TEST, f)
                test_out = test_inn * a
                test_err_inn = (vmap(model)(test_inn) > .5).mean(axis=0)
                test_err_out = (vmap(model)(test_out) < .5).mean(axis=0)
                # store errors of training dataset
                train_err_inn = (vmap(model)(inn) > .5).mean(axis=0)
                train_err_out = (vmap(model)(out) < .5).mean(axis=0)
                # store results
                results = jnp.hstack((test_err_inn, test_err_out,
                                      train_err_inn, train_err_out))
                RESULTS['ERRORS'][ida, idf, idn] = results
                RESULTS['LOSSES'][-1].append(losses)
                # plot decision boundary
                # fig, ax = plot_decision(model, inn, out)
                # plt.savefig(experiment / f'decision_{a:.2f}_{f}_{n}.png')
                # plt.close()

    # some plotting
    def plot_errors(errors):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Error fraction')
        cmap = mpl.colormaps['cividis']
        colors = [cmap(i) for i in jnp.linspace(0, 1, len(errors))]
        errs = errors.mean(axis=-1)
        for err, c in zip(errs, colors):
            err_mean = err.mean(axis=-1)
            ax.plot(ENN, err_mean, '-o', c=c)
            # add errorbars
            err_sem = jnp.std(err, axis=-1) / jnp.sqrt(PPP)
            ax.fill_between(ENN, err_mean-err_sem, err_mean+err_sem, alpha=.25, color=c)
        # for colorbar
        normer = plt.Normalize(vmin=FREQS[0], vmax=FREQS[-1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normer)
        fig.colorbar(sm, ax=ax, label='frequency')
        # for legend
        return fig, ax

    # make plots and save them
    for ida, a in enumerate(ALPHAS):
        # select test errors
        test_errs = RESULTS['ERRORS'][ida, ..., :2]  # only tests
        fig, ax = plot_errors(test_errs)
        plt.savefig(experiment / f'errors_test_{a:.2f}.png')
        plt.close()
        # select train errors
        train_errs = RESULTS['ERRORS'][ida, ..., 2:]  # only train
        fig, ax = plot_errors(train_errs)
        plt.savefig(experiment / f'errors_train_{a:.2f}.png')
        plt.close()

    # and conclude by saving the RESULTS
    RESULTS['ERRORS'] = RESULTS['ERRORS'].tolist()
    experiment.save_dict(RESULTS, 'results.json')
    # modify the PARAMS dict to make it saveable
    # PARAMS['arch']['final_activation'] = PARAMS['arch']['final_activation'].__name__
    PARAMS['schedule'] = PARAMS['schedule'].__name__
    PARAMS['clipper'] = PARAMS['clipper'].__name__
    PARAMS['optimizer'] = PARAMS['optimizer'].__name__
    # NOW save it :3
    experiment.save_dict(PARAMS, 'params.json')
