from jax import numpy as jnp, random as jrand, vmap
from functools import partial


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

# CIRCLES AND DEFORMED CIRCLES
def get_polar_loop(key, N, H=5, rmin=-2.5, rmax=-.5):
    """Sample H harmonics and add them to a fixed radius of 1. For H=0, the
    standard circle is recovered."""
    # sample radii and phases
    rp = jrand.uniform(key, (2*H, ))
    rho = (rp[:H] * jnp.logspace(rmin, rmax, H))[:, None]
    phi = (rp[H:] * 2 * jnp.pi)[:, None]
    # accumulation step
    ts = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    hs = jnp.arange(1, H+1)[:, None]
    thetas = jnp.sin(hs*ts + phi)
    r = 1 + (rho*thetas).sum(axis=0)
    # return xs and ys
    return jnp.stack((r*jnp.cos(ts), r*jnp.sin(ts)), axis=1)


# ORBIT OF SHIFT GROUP
def shifter(dx, d):
    return jnp.exp(dx*1j*jnp.arange((d+1)//2)*(2*jnp.pi))


def get_shifted_signal(key, N, d):
    # signal itself
    x0 = jrand.randint(key, shape=(d,), minval=0, maxval=2)
    x0_hat = jnp.fft.rfft(x0)
    shifts = vmap(partial(shifter, d=d))(jnp.linspace(0, 1, N))
    return vmap(partial(jnp.fft.irfft, n=d))(shifts*x0_hat)


# ACTUAL DATASETS
def get_dataset(inner, transf):
    N = len(inner)
    outer = transf(inner)
    xs = jnp.concatenate((inner, outer))
    ys = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return xs, ys


if __name__ == "__main__":
    rng = jrand.PRNGKey(0)
    print(get_polar_loop(rng, 2, H=0))
