"""Learning a flower-like 2D shape. Points drawn randomly. Averaging over
random datasets."""
from tqdm import tqdm
import jax
from jax import numpy as jnp, random as jrand, vmap, nn as jnn
from numpy import zeros
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier

from dataset_utils import dataloader

from expman import ExpLogger


class PMLP(eqx.Module):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array

    def _make_layer(self, in_size, out_size, n, key):
        wk, bk = jrand.split(key)
        lim1 = 1 / jnp.sqrt(in_size)
        w1 = jrand.uniform(wk, (n, out_size, in_size), minval=-lim1, maxval=lim1)
        b1 = jrand.uniform(bk, (n, out_size), minval=-lim1, maxval=lim1)
        return w1, b1

    def __init__(self, in_size, out_size, width_size, n, key):
        k1, k2, k3 = jrand.split(key, 3)
        self.w1, self.b1 = self._make_layer(in_size, width_size, n, k1)
        self.w2, self.b2 = self._make_layer(width_size, width_size, n, k2)
        self.w3, self.b3 = self._make_layer(width_size, out_size, n, k3)

    def __call__(self, x):
        x = self.b1 + jnp.einsum('nwi,ni->nw', self.w1, x)
        x = jnn.relu(x)
        x = self.b2 + jnp.einsum('nwi,ni->nw', self.w2, x)
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


def sine_on_circle(N, n, f: int, a: float = .25, *, key=None):
    """It is not really a sine wave on a circle, but close enough"""
    ts = jrand.uniform(key, shape=(N, n)) * (2*jnp.pi)
    r = 1 + a*jnp.sin(f*ts)
    return jnp.stack((r*jnp.cos(ts), r*jnp.sin(ts)), axis=-1)


# training function
def ce(y, y_pred):
    return -jnp.mean(y*jnp.log(y_pred) + (1-y)*jnp.log(1-y_pred))


@eqx.filter_value_and_grad
def loss(model, x, y):
    y_preds = vmap(model)(x)  # (batch_size, n, fakedim)
    y_preds = y_preds.squeeze().T  # (n, batch_size)
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
              'width_size': 32,
              'n': 5,
              # 'final_activation': jnn.sigmoid
              },
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
        freqs=[4, ],  # 5, 6, 7, 8],
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
    REPS = PARAMS['arch']['n']  # how many datasets/models?

    # prepare the storage of results
    res_shape_mlps = (len(ALPHAS), len(FREQS), len(ENN), REPS, 2)
    res_shape_knn = (len(ALPHAS), len(FREQS), len(ENN), REPS)
    RESULTS = dict(
        ERRORS_MLP=zeros(res_shape_mlps),
        ERRORS_KNN=zeros(res_shape_knn),
        LOSSES=[],
    )

    # prepare the optimizer
    schedule = PARAMS['schedule'](**PARAMS['schedule_params'])
    opt = PARAMS['optimizer'](**PARAMS['optimizer_params'], learning_rate=schedule)
    clipper = PARAMS['clipper'](**PARAMS['clipper_params'])
    optimizer = optax.chain(clipper, opt)

    # and prepare the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)

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
                inn = sine_on_circle(n, REPS, f, key=dkey)
                out = inn * a
                labs = jnp.repeat(jnp.arange(2), n)
                dset = (jnp.concatenate((inn, out)), labs)
                dload = dataloader(dset, **PARAMS['dloader_params'], skey=skey)
                # train the model on the data
                model = PMLP(**PARAMS['arch'], key=mkey)
                model, losses = train_model(model, dload, optimizer)
                # perform test on trained model, save errors
                test_inn = sine_on_circle(N_TEST, REPS, f, key=dkey)
                test_out = test_inn * a
                test_set = jnp.concatenate((test_inn, test_out))
                test_lab = jnp.repeat(jnp.arange(2), N_TEST)
                test_pred = jnp.heaviside(vmap(model)(test_set) - .5, 0).squeeze()
                test_err = (test_pred != test_lab[:, None]).mean(axis=0)
                # and on train
                train_pred = jnp.heaviside(vmap(model)(dset[0]) - .5, 0).squeeze()
                train_err = (train_pred != dset[1][:, None]).mean(axis=0)
                # store results
                results = jnp.stack((test_err, train_err)).T
                RESULTS['ERRORS_MLP'][ida, idf, idn] = results
                RESULTS['LOSSES'][-1].append(losses)
                # train and evaluate KNN on each dataset copy
                test_err_knn = []
                for draw in range(REPS):
                    knn.fit(dset[0][:, draw], dset[1])
                    test_draw = jnp.concatenate((test_inn[:, draw], test_out[:, draw]))
                    knn_preds = knn.predict(test_draw)
                    test_err_knn.append(jnp.mean(knn_preds != test_lab))
                RESULTS['ERRORS_KNN'][ida, idf, idn] = test_err_knn

    # some plotting
    def plot_errors(errors_mlp, errors_knn):
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_xlabel('Number of samples')
        ax.set_ylabel('Error fraction')
        cmap = mpl.colormaps['cividis']
        colors = [cmap(i) for i in jnp.linspace(0, 1, len(errors_mlp))]
        for err_mlp, err_knn, c in zip(errors_mlp, errors_knn, colors):
            # first for the mlp
            err_mlp_mean = err_mlp.mean(axis=-1)
            ax.plot(ENN, err_mlp_mean, '-o', c=c)
            err_mlp_sem = jnp.std(err_mlp, axis=-1) / jnp.sqrt(REPS)
            lo = err_mlp_mean-err_mlp_sem
            hi = err_mlp_mean+err_mlp_sem
            ax.fill_between(ENN, lo, hi, alpha=.25, color=c)
            # then for the knn
            err_knn_mean = err_knn.mean(axis=-1)
            ax.plot(ENN, err_knn_mean, '--o', c=c)
            err_knn_sem = jnp.std(err_knn, axis=-1) / jnp.sqrt(REPS)
            lo = err_knn_mean-err_knn_sem
            hi = err_knn_mean+err_knn_sem
            ax.fill_between(ENN, lo, hi, alpha=.25, color=c)
        # for colorbar
        normer = plt.Normalize(vmin=FREQS[0], vmax=FREQS[-1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normer)
        fig.colorbar(sm, ax=ax, label='frequency')
        # for legend
        return fig, ax

    # make plots and save them
    for ida, a in enumerate(ALPHAS):
        # select test errors
        test_errs_mlp = RESULTS['ERRORS_MLP'][ida, ..., 0]  # only tests
        test_errs_knn = RESULTS['ERRORS_KNN'][ida]  # only tests
        fig, ax = plot_errors(test_errs_mlp, test_errs_knn)
        # plt.savefig(experiment / f'errors_test_{a:.2f}.png')
        # plt.close()
        # # select train errors
        # train_errs = RESULTS['ERRORS_MLP'][ida, ..., 1]  # only train
        # fig, ax = plot_errors(train_errs)
        # plt.savefig(experiment / f'errors_train_{a:.2f}.png')
        # plt.close()

    # and conclude by saving the RESULTS
    RESULTS['ERRORS_MLP'] = RESULTS['ERRORS_MLP'].tolist()
    RESULTS['ERRORS_KNN'] = RESULTS['ERRORS_KNN'].tolist()
    experiment.save_dict(RESULTS, 'results.json')
    # modify the PARAMS dict to make it saveable
    # PARAMS['arch']['final_activation'] = PARAMS['arch']['final_activation'].__name__
    PARAMS['schedule'] = PARAMS['schedule'].__name__
    PARAMS['clipper'] = PARAMS['clipper'].__name__
    PARAMS['optimizer'] = PARAMS['optimizer'].__name__
    # NOW save it :3
    experiment.save_dict(PARAMS, 'params.json')
