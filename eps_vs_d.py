import numpy as np
from scipy.special import erf
from functools import partial
from jax import numpy as jnp, random as jrand, vmap, jit
# from datasets import spherical_uniform
# from itertools import product
import matplotlib.pyplot as plt


def spherical_uniform(key, d, shape=(1,)):
    """
    Generate samples from the uniform N dimensional spherical distribution.
    """
    # generate from multivariate normal
    z = jrand.normal(key=key, shape=(*shape, d))
    norm = jnp.sqrt((z**2).sum(axis=-1, keepdims=True))
    return z/norm


# the analytic result
def H(x):
    return .5*(1-erf(x/np.sqrt(2)))


def formula_S17(dx, Ra, Rb, D):
    delta = .5*(dx**2+(Rb/Ra)**2-1)
    f1 = ((1-delta**2)**(D/2))
    f2 = np.exp((D/2)*(delta**2)/(1-delta**2))
    f3 = H(np.sqrt((D*delta**2)/(1-delta**2)))
    return f1*f2*f3


def formula_S18(dx, Ra, Rb, D):
    return H(.5*(dx**2+(Rb/Ra)**2-1)/(np.sqrt(1/D)))


# determine the class
def is_class_a(xa, xb, xi):
    return -(xa-xi)@(xa-xi).T+(xb-xi)@(xb-xi).T > 0
is_a = jit(vmap(is_class_a, in_axes=(0, 0, 0)))


N_EXP = int(1e4)
SEED = int(input("Insert seed: "))
key = jrand.PRNGKey(SEED)
# geometric parameters
n = 300
m = 1
Ra = 1.
Rb = 1.
S = 1.
d_ = jnp.arange(1, (ND:=49)+1)

# define the orthogonal basis, and make it orthonormal
uk, xak, xbk, sak, sbk, sigmak = jrand.split(key=key, num=6)
Uall = jrand.orthogonal(key=uk, n=n)
# store the subspaces and the related quantities
results = np.empty(ND)
for idx, d in enumerate(d_):
    Ua, Ub, dx = Uall[:, :d], Uall[:, d:2*d], Uall[:, 2*d+1]
    x0a = jrand.normal(key=xak, shape=(n,))
    # draw the coefficients
    sa_ = spherical_uniform(key=sak, d=d, shape=(N_EXP, m))
    sb_ = spherical_uniform(key=sbk, d=d, shape=(N_EXP, m))
    sigma_ = spherical_uniform(sigmak, d=d, shape=(N_EXP,))
    # get the offsets from the centers
    uxas = sa_@Ua.T
    uxbs = sb_@Ub.T
    uxis = sigma_@Ua.T
    # get the data
    x0b = (x0a-S*dx)
    xas = x0a + Ra*uxas
    xbs = x0b + Rb*uxbs
    xis = x0a + Ra*uxis
    xi_is_a = is_a(xas.mean(axis=1), xbs.mean(axis=1), xis)
    eps = 1 - jnp.mean(xi_is_a, axis=0)
    results[idx] = eps


# with fixed radii, show analytic result
f17 = partial(formula_S17, Ra=Ra, Rb=Rb, dx=S)
f18 = partial(formula_S18, Ra=Ra, Rb=Rb, dx=S)
ys17 = [f17(D=d) for d in d_]
ys18 = [f18(D=d) for d in d_]
# create plot
plt.plot(d_, results, label='empirical')
plt.plot(d_, ys17, label='theory - SI17', c='black', linestyle='--')
plt.plot(d_, ys18, label='theory - SI18', c='red', linestyle='--')
plt.title(f"Ra=1, Rb=1, S=1, m={m}, n={n}")
plt.ylabel("Error probability")
plt.xlabel("D")
plt.show()
