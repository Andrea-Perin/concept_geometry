"""
Specifies how to generate datasets with which to test Sorscher et al. results.
"""
# %% IMPORTS
from jax import numpy as jnp, random as jrand, jit


# %% DATA GENERATION FUNCTIONS
def spherical_uniform(key, d, shape=(1,)):
    """
    Generate samples from the uniform N dimensional spherical distribution.
    """
    # generate from multivariate normal
    z = jrand.multivariate_normal(key=key, mean=jnp.zeros(d), cov=jnp.eye(d), shape=shape)
    norm = jnp.sqrt(jnp.sum(z**2, axis=1, keepdims=True))
    return z/norm


def get_points_ortho(n, d, m=1, Ra=1., Rb=1., delta=1., *, key):
    # generate orthogonal subspaces of the same dimension, plus one for the
    # signal direction
    uk, xak, xbk, sak, sbk, sigmak = jrand.split(key=key, num=6)
    Uall = jrand.orthogonal(key=uk, n=n)[:, :2*d+1]
    Ua, Ub, delta_x = Uall[:, :d], Uall[:, d:-1], Uall[:, -1]
    # generate centroids for the two manifolds
    # the signal-noise overlap must be zero
    x0a = jrand.normal(key=xak, shape=(n,))
    x0b = delta*((x0a-delta_x)/((x0a-delta_x)@(x0a-delta_x)))
    # now let us sample from the sphere
    sa = spherical_uniform(key=sak, d=d, shape=(m,))
    sb = spherical_uniform(key=sbk, d=d, shape=(m,))
    sigma = spherical_uniform(sigmak, d=d)
    # and we can finally return all the results we want
    xas = x0a + Ra*(sa@Ua.T)
    xbs = x0b + Rb*(sb@Ub.T)
    xi = x0a + Ra*(sigma@Ua.T)
    return xas, xbs, xi


# %% TESTS
if __name__ == "__main__":
    SEED = 123
    key = jrand.PRNGKey(SEED)
    # dataset parameters
    dset_kwargs = dict(
        n = 100,
        d = 20,  # must be < n//2
        m = 130,
        Ra = 2.,
        Rb = 3.,)
    # generate the points
    xas, xbs, xi = get_points_ortho(**dset_kwargs, key=key)
    print(xas.shape)
    print(xbs.shape)
    print(xi.shape)
