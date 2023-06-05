"""Trying the method in https://scicomp.stackexchange.com/a/35744

Much faster than the GP method, probably best to use this one
"""
from jax import numpy as jnp, random as jrand


def get_polar_loop(key, N, H=5, rmin=-2.5, rmax=-.5):
    # sample radii and phases
    rp = jrand.uniform(key, (2*H, ))
    rho = (rp[:H] * jnp.logspace(rmin, rmax, H))[:, None]
    phi = (rp[H:] * 2 * jnp.pi)[:, None]
    # accumulation step
    ts = jnp.linspace(0, 2*jnp.pi, N)
    hs = jnp.arange(1, H+1)[:, None]
    r = jnp.ones_like(ts)
    thetas = jnp.sin(hs*ts + phi)
    r += (rho*thetas).sum(axis=0)
    # return xs and ys
    return r*jnp.cos(ts), r*jnp.sin(ts)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SEED = 0
    rng = jrand.PRNGKey(SEED)
    xs, ys = get_polar_loop(rng, 1000, H=12, rmin=-2, rmax=-1)
    plt.plot(xs, ys)
    plt.show()
