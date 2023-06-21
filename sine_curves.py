"""Trying out learning curves on a sine curve on a circle."""
from jax import numpy as jnp, random as jrand, vmap, nn as jnn
from numpy import zeros
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from functools import partial

from dataset_utils import dataloader, get_dataset, get_shifted_signal, get_polar_loop
from model_utils import StupidMLP

from expman import ExpLogger


def sine_on_circle(N, f: int, a: float = .25):
    """It is not really a sine wave on a circle, but close enough"""
    ts = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
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
              'width_size': 2048,
              'depth': 1,
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
        dloader_params={'batch_size': 128,
                        'n_epochs': 5000, },
        # EXPERIMENTAL PARAMS
        # alphas=[1.0001, 1.0005, 1.001, 1.002, 1.003, 1.004, 1.005],
        alphas=[1.1, 1.5],
        freqs=list(range(2, 3)),
        ns=jnp.unique(jnp.logspace(NMIN:=.5, NMAX:=3, NN:=2).astype(int)).tolist(),
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
        ERRORS=zeros((*res_shape, 2)),
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
    for ida, a in enumerate(ALPHAS):
        RESULTS['LOSSES'].append([])
        for idf, f in enumerate(FREQS):
            RESULTS['LOSSES'][-1].append([])
            for idn, n in enumerate(ENN):
                # create dataset
                inn = sine_on_circle(n, f)
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
                err_out = float((vmap(model)(test_inn) > .5).mean())
                # store results
                RESULTS['ERRORS'][ida, idf, idn] = (err_inn, err_out)
                RESULTS['LOSSES'][-1].append(losses)

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
