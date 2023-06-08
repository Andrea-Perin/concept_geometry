from jax import numpy as jnp, random as jrand


# dataloade  to perform batching
def batched(array, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    i = 0
    while i*n < len(array):
        yield array[i*n: (i+1)*n]
        i += 1


def dataloader(dataset, n_epochs, batch_size, *, skey):
    # at every epoch, shuffle the data
    xs, ys = dataset
    shkeys = jrand.split(skey, num=n_epochs)
    for k in shkeys:
        perm = jrand.permutation(k, len(xs))
        sxs, sys = xs[perm], ys[perm]
        xbatch = batched(sxs, batch_size)
        ybatch = batched(sys, batch_size)
        for bx, by in zip(xbatch, ybatch):
            yield bx, by


# DATASETS
def get_polar_loop(key, N, H=5, rmin=-2.5, rmax=-.5):
    """Sample H harmonics and add them to a fixed radius of 1. For H=0, the
    standard circle is recovered."""
    # sample radii and phases
    rp = jrand.uniform(key, (2*H, ))
    rho = (rp[:H] * jnp.logspace(rmin, rmax, H))[:, None]
    phi = (rp[H:] * 2 * jnp.pi)[:, None]
    # accumulation step
    ts = jnp.linspace(0, 2*jnp.pi, N)
    hs = jnp.arange(1, H+1)[:, None]
    thetas = jnp.sin(hs*ts + phi)
    r = 1 + (rho*thetas).sum(axis=0)
    # return xs and ys
    return jnp.stack((r*jnp.cos(ts), r*jnp.sin(ts)), axis=1)


def get_dataset(key, N, alpha, **kwargs):
    inner = get_polar_loop(key, N, **kwargs)
    outer = alpha*inner
    xs = jnp.concatenate((inner, outer))
    ys = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return xs, ys
