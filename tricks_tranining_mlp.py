"""The usual training of an MLP. This time with added tricks.

Includes:
* sgd,
* batching,
* dropout,
* batchnorm,
* 3 layers,
* relu,
* error threshold,
* ensembling,
* learning rate,
* curriculum learning
"""
from jax import numpy as jnp, random as jrand, vmap, nn as jnn
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from functools import partial

from numpy import zeros

from dataset_utils import dataloader, get_dataset
from model_utils import FatMLP
from plotting_utils import plot_decision, plot_loss, plot_error_fracs, plot_n_alpha

from expman import ExpLogger


# training function
@partial(eqx.filter_value_and_grad, has_aux=True)
def loss(model, state, x, y, *, key):
    batch_mod = vmap(model, in_axes=(0, None, None), axis_name='batch', out_axes=(0, None))
    pred_y, state = batch_mod(x, state, key)
    pred_y = pred_y.squeeze()
    return -jnp.mean(y*jnp.log(pred_y) + (1-y)*jnp.log(1-pred_y)), state


@eqx.filter_jit
def make_step(model, mod_state, opt_state, x, y, *, key):
    ((loss_value, mod_state), grads) = loss(model, mod_state, x, y, key=key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, mod_state, opt_state, loss_value


def train_model(model, dloader, optimizer, *, key):
    # record model's state
    mod_state = eqx.nn.State(model)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    for (xs, ys) in dloader:
        key, tkey = jrand.split(key)
        model, mod_state, opt_state, loss = make_step(model, mod_state, opt_state, xs, ys, key=tkey)
        losses.append(loss)
    # put the trained model in eval mode
    model = eqx.tree_inference(model, value=True)
    model = eqx.Partial(model, state=mod_state, key=jrand.PRNGKey(0))
    mod_out = lambda x: model(x)[0]
    return mod_out, losses


# the whole thing
with ExpLogger() as experiment:
    # experiment parameters
    PARAMS = dict(
        seed=int(input("Insert random seed (default: 0): ") or "0"),
        # NETWORK ARCHITECTURE
        arch={'in_size': 2,
              'out_size': 1,
              'width_size': 2048,
              'depth': 1,
              'activation': jnn.relu,
              'final_activation': jnn.sigmoid,
              'use_bias': True,
              'use_final_bias': True,
              'dropout_pct': 0.1,
              'batch_norm': True},
        # OPTIMIZER STUFF
        schedule=optax.constant_schedule,
        # schedule_params={'init_value': 0,
        #                  'peak_value': 0.5,
        #                  'warmup_steps': 100,
        #                  'decay_steps': 1000,
        #                  'end_value': 0.0, },
        schedule_params={'value': 1e-3,},
        clipper=optax.clip,
        clipper_params={'max_delta': 2.0},
        optimizer=optax.sgd,
        optimizer_params={},
        # DLOADER PARAMS
        dloader_params={'batch_size': 512,
                        'n_epochs': 1000, },
        # DATASET PARAMS
        dset_params={'H': 5,
                     'rmin': -3,
                     'rmax': -2, },
        # EXPERIMENTAL PARAMS
        alpha=[1.01, 1.1, 1.5],
        err_threshold=0,
        # OTHER
        n_test=int(1e4),
        n_max=1000,
    )

    # some aliases for shorter code
    N_TEST = PARAMS['n_test']
    N_MAX = PARAMS['n_max']
    ALPHAS = PARAMS['alpha']
    ERR = PARAMS['err_threshold']
    WIDTH = PARAMS['arch']['width_size']

    # prepare the storage of results
    RESULTS = dict(
        N=zeros(len(ALPHAS), dtype=int),
        ERRORS={a: {} for a in ALPHAS},
    )

    # prepare the optimizer
    schedule = PARAMS['schedule'](**PARAMS['schedule_params'])
    opt = PARAMS['optimizer'](
        **PARAMS['optimizer_params'], learning_rate=schedule)
    clipper = PARAMS['clipper'](**PARAMS['clipper_params'])
    optimizer = optax.chain(clipper, opt)

    # ===============
    # EXPERIMENT CODE
    # ===============
    # start the RNG
    rng = jrand.PRNGKey(PARAMS['seed'])
    dkey, skey, mkey, tkey = jrand.split(rng, num=4)
    # start looping over alphas
    for ida, a in enumerate(ALPHAS):
        # random curve case
        n_lo, n, n_hi = 1, N_MAX//2, N_MAX
        while True:
            n = (n_lo + n_hi) // 2
            # create dataset and train the model
            dset = get_dataset(dkey, n, a, **PARAMS['dset_params'])
            dload = dataloader(dset, **PARAMS['dloader_params'], skey=skey)
            model = FatMLP(**PARAMS['arch'], key=mkey)
            model, losses = train_model(model, dload, optimizer, key=tkey)
            # perform test on trained model, save errors
            testset, _ = get_dataset(dkey, N_TEST, a, **PARAMS['dset_params'])
            inn, out = testset[:N_TEST], testset[N_TEST:]
            eps_inn = float((vmap(model)(inn) > .5).mean())
            eps_out = float((vmap(model)(out) < .5).mean())
            RESULTS['ERRORS'][a][n] = {'inn': eps_inn, 'out': eps_out}
            # msg
            msg = f"n:{n},nlo:{n_lo},nhi:{n_hi},inn:{eps_inn:.3f},out:{eps_out:.3f}"
            print(msg)
            # exit condition
            n_is_sufficient = (eps_inn <= ERR) and (eps_out <= ERR)
            # exit condition
            if (n == n_lo) and (not n_is_sufficient):
                n += 1
                break
            if (n == n_hi) and (n_is_sufficient):
                break
            # update n_lo and n_hi
            n_hi = n if n_is_sufficient else n_hi
            n_lo = n_lo if n_is_sufficient else n

        # collect results and produce output
        RESULTS['N'][ida] = n
        # create plot with trained model on given data and save it
        fig, ax = plot_decision(model, a, dset[0][:n], dset[0][n:])
        plt.savefig(experiment / f"decision_{a:.3f}.png")
        plt.close()
        # create plot with loss progression on latest model and save iter
        fig, ax = plot_loss(losses, a)
        plt.savefig(experiment / f"losses_{a:.3f}.png")
        plt.close()

    # create plot with errors vs Ns for various alphas
    fig, ax = plot_error_fracs(RESULTS['ERRORS'])
    plt.savefig(experiment / 'errors_behavior.png')
    plt.close()

    # plot the final N vs alpha
    fig, ax = plot_n_alpha(ALPHAS, RESULTS['N'], ERR, WIDTH)
    plt.savefig(experiment / 'n_vs_a.png')
    plt.close()

    # and conclude by saving the RESULTS
    RESULTS['N'] = RESULTS['N'].tolist()
    experiment.save_dict(RESULTS, 'results.json')

    # modify the PARAMS dict to make it saveable
    PARAMS['arch']['activation'] = PARAMS['arch']['activation'].__name__
    PARAMS['arch']['final_activation'] = PARAMS['arch']['final_activation'].__name__
    PARAMS['schedule'] = PARAMS['schedule'].__name__
    PARAMS['clipper'] = PARAMS['clipper'].__name__
    PARAMS['optimizer'] = PARAMS['optimizer'].__name__
    # NOW save it :3
    experiment.save_dict(PARAMS, 'params.json')
