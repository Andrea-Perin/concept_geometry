from jax import numpy as jnp, random as jrand, jit, vmap
from functools import partial
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

    # test on a single variable
    fixed_kwargs=dict(
        n = 100,
        d = 30,
        m = 1,
        delta = 1.,
        Ra = 1,)
    Rb = jnp.linspace(1, 10, 10)
    dset_kwargs = [{**fixed_kwargs, 'Rb':r} for r in Rb]

    run_test = vmap(test_configuration, in_axes=(0,))
    results = [run_test(
    results = run_all_tests(keys, dset_kwargs)
