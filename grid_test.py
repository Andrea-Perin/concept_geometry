from jax import numpy as jnp, random as jrand, jit, vmap
from functools import partial
from itertools import product
from tqdm import tqdm

from datasets import get_points_ortho

@jit
def is_class_a(xa, xb, xi):
    return -(xa-xi)@(xa-xi)+(xb-xi)@(xb-xi) > 0

def test_configuration(key, dset_kwargs):
    xas, xbs, xi = get_points_ortho(**dset_kwargs, key=key)
    # compute the average of xas and xbs
    xa = jnp.mean(xas, axis=0)
    xb = jnp.mean(xbs, axis=0)
    return is_class_a(xa, xb, xi.squeeze())

if __name__ == "__main__":
    SEED = 123
    master_key = jrand.PRNGKey(SEED)
    # generate many keys in order to average the results
    N_EXPS = 1000
    keys = jrand.split(key=master_key, num=N_EXPS)
    dset_kwargs = dict(
        n = 100,
        d = 30,  # must be < n//2
        m = 1,
        delta=.1,
        Ra = 100.,
        Rb = 10.,)
    run_exp = partial(test_configuration, dset_kwargs=dset_kwargs)
    eps = jnp.mean(vmap(run_exp)(keys))

    print("now doing the spicy part")
    # parameter grid
    n_ = (100,)  # let us keep this one fixed
    d_ = tuple(range(5, 30, 5))
    m_ = (1,)  # also this one fixed
    Ra_ = jnp.linspace(1, 10, 5)
    Rb_ = jnp.linspace(1, 10, 5)
    delta_ = jnp.linspace(0.1, 1, 5)
    prod = product(n_, d_, m_, delta_, Ra_, Rb_)
    names = ('n', 'd', 'm', 'delta', 'Ra', 'Rb')
    dictprod = [dict(zip(names, p)) for p in prod]
    # run the experiments
    run_test = vmap(test_configuration, in_axes=(0, None))
    all_eps = [run_test(keys, combo) for combo in tqdm(dictprod)]
