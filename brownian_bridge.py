from jax import numpy as jnp, random as jrand, vmap
from sklearn.gaussian_process import GaussianProcessRegressor as gpr


def moving_average(x, w):
    periodic_x = jnp.concatenate((x[-w//2+1:], x, x[:w//2]))
    return jnp.convolve(periodic_x, jnp.ones(w), 'valid') / w


def bbk(x):
    # compute brownian bridge kernel matrix
    return jnp.minimum(x[:, None], x[None, :]) - x[:, None]*x[None, :]


def rqk(x, l=1., alpha=100, sig=1.):
    return (sig**2)*(1+((x[:, None]-x)**2)/(2*alpha*l**2))**(-alpha)


def bbeqk(x, l=1, sig=1.):
    K_bb = bbk(x)
    K_eqk = eqk(x, l=l, sig=sig)
    return K_bb*K_eqk


def eqk(x, l=1., sig=1.):
    return (sig**2)*jnp.exp(-((x[:, None]-x)**2)/(2*l**2))


def get_points_flat(key, N, **kwargs):
    ts_all = jnp.linspace(0, 1, N)
    K = eqk(ts_all, **kwargs)
    r_mean = jnp.zeros(N)  # mean function
    xs = jrand.multivariate_normal(key, r_mean, K, method='svd')
    return ts_all, xs


def get_points(key, N_all, kerf=rqk, **kwargs):
    xk, yk = jrand.split(key)
    ts_all = jnp.linspace(0, 1, N_all)
    K = kerf(ts_all, **kwargs)
    r_mean = jnp.ones(N_all)  # mean function
    xs = jrand.multivariate_normal(xk, r_mean, K, method='svd')
    ys = jrand.multivariate_normal(yk, r_mean, K, method='svd')
    return xs, ys


def get_points_trick(key, N_all, w=50, kerf=rqk, **kwargs):
    ts_all = jnp.linspace(0, 1, N_all)
    K = kerf(ts_all, **kwargs)
    r_mean = jnp.ones(N_all)  # mean function
    drs = jrand.multivariate_normal(key, r_mean, K, method='svd')
    # smooth everything out locals
    drs_smooth = moving_average(drs, w=w)
    r = 1 + drs_smooth
    xs = r*jnp.cos(ts_all*jnp.pi*2)
    ys = r*jnp.sin(ts_all*jnp.pi*2)
    return xs, ys


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seed = 11
    rng = jrand.PRNGKey(seed)
    N = 5000
    W = int(.075*N)
    sig = 1
    l = int(1e0)
    xs, ys = get_points_trick(rng, N, w=W, kerf=bbeqk, sig=sig, l=l)
    plt.plot(xs, ys)
    plt.show()
