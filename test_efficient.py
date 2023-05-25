from jax import numpy as jnp, random as jrand, vmap, jit
from functools import partial
from datasets import spherical_uniform
from itertools import product
import matplotlib.pyplot as plt


SEED = 1234
key = jrand.PRNGKey(SEED)

# params
N_EXPS = 13
keys = jrand.split(key, num=N_EXPS)

# generate random values just once
def get_random_params(n, d, m=1, *, key):
    # generate orthogonal subspaces of the same dimension, plus one for the
    # signal direction
    uk, xak, xbk, sak, sbk, sigmak = jrand.split(key=key, num=6)
    Uall = jrand.orthogonal(key=uk, n=n)[:, :2*d+1]
    Ua, Ub, delta_x = Uall[:, :d], Uall[:, d:-1], Uall[:, -1]
    # generate centroid for one manifold (the other can be inferred)
    x0a = jrand.normal(key=xak, shape=(n,))
    # now let us sample from the sphere
    sa = spherical_uniform(key=sak, d=d, shape=(m,))
    sb = spherical_uniform(key=sbk, d=d, shape=(m,))
    sigma = spherical_uniform(sigmak, d=d)
    return Ua, Ub, delta_x, x0a, sa, sb, sigma


@jit
def produce_data(Ua, Ub, delta_x, x0a, sa, sb, sigma, Ra, Rb, signal):
    # retrieve x0b
    x0b = signal*((x0a-delta_x)/((x0a-delta_x)@(x0a-delta_x)))
    # mix everything up
    xas = x0a + Ra*(sa@Ua.T)
    xbs = x0b + Rb*(sb@Ub.T)
    xi = x0a + Ra*(sigma@Ua.T)
    return xas, xbs, xi


@jit
def get_xas(Ua, x0a, sa, Ra):
    return x0a + Ra*(sa@Ua.T)

@jit
def get_xbs(Ub, x0b, sb, Rb):
    return x0b + Rb*(sb@Ub.T)

@jit
def get_x0b(x0a, delta_x, signal):
    return signal*((x0a-delta_x)/((x0a-delta_x)@(x0a-delta_x)))

@jit
def get_xi(Ua, x0a, sigma, Ra):
    return x0a + Ra*(sigma@Ua.T)

@jit
def is_class_a_flat(x):
    xa, xb, xi = x[0].mean(axis=0), x[1].mean(axis=0), x[2].squeeze()
    return -(xa-xi)@(xa-xi).T+(xb-xi)@(xb-xi).T > 0

# data generation parameters
n = 100
d = 30
m = 1
Ra = 1.
Rb = jnp.linspace(1, 20)
signal = jnp.linspace(0.1, 10)
# produce the fixed random variables only once
all_rands = [get_random_params(n=n, d=d, m=m, key=k) for k in keys]
print("done generating")
all_rands_transposed = list(zip(*all_rands))
Ua_, Ub_, delta_x_, x0a_, sa_, sb_, sigma_ = map(jnp.stack, all_rands_transposed)
print("done reshaping")

# getters
get_x0b_fixed = partial(vmap(get_x0b, in_axes=(0, 0, None)), x0a_, delta_x_)
get_xas_fixed = partial(vmap(get_xas, in_axes=(0, 0, 0, None)), Ua_, x0a_, sa_)
get_xbs_fixed = partial(vmap(vmap(get_xbs, in_axes=(None, 0, None, None)), in_axes=(0, 0, 0, None)), Ub_, x0b_, sb_)


grid = product(Rb, signal)
eps_out = []
for (rb, sig) in grid:
    xas_ = vmap(get_xas_fixed)(Ra)
    x0b_ = vmap(get_x0b_fixed)(Rb)
    xbs_ = vmap(get_xbs_fixed)(Rb)



# create the data points
# x0b_ = vmap(vmap(get_x0b, in_axes=(0, 0, None)), in_axes=(None, None, 0))(x0a_, delta_x_, signal)
# xis = vmap(get_xi, in_axes=(0, 0, 0, None))(Ua_, x0a_, sigma_, Ra)


# data = [[produce_data(*ar, Ra, Rb, s) for ar in all_rands] for s in signal]
# data = jnp.array(data)
# # run the test on all the data points
# eps = vmap(vmap(is_class_a_flat))(data)
# eps = eps.squeeze()
# # take average over the realizations (keys)
# avg_eps = 1 - jnp.mean(eps, axis=1)
# 
# # quick plot
# plt.plot(signal, avg_eps)
# plt.title("Error probability vs. signal")
# plt.xlabel("Signal")
# plt.ylabel("Error probability")
# plt.show()
