from functools import partial
import numpy as np
from scipy.special import erf
from jax import numpy as jnp, random as jrand, vmap, jit
# from datasets import spherical_uniform
import matplotlib.pyplot as plt


def spherical_uniform_fake(key, d, shape=(1,)):
    """
    Generate samples from the uniform N dimensional spherical distribution.
    """
    # generate from multivariate normal
    z = jrand.normal(key=key, shape=(*shape, d))
    norm = jnp.sqrt(d)
    return z/norm


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


# rng
N_EXP = int(1e4)
SEED = int(input("insert seed: "))
key = jrand.PRNGKey(SEED)
# geometric parameters
n = 200
d = 50
m = 1
Ra = 1.
Rb_ = jnp.linspace(1., 1.5, NRB := 1)
S_ = jnp.linspace(0.001, 2, NS := 100) / Ra

# define the orthogonal basis, and make it orthonormal
uk, xak, xbk, sak, sbk, sigmak = jrand.split(key=key, num=6)
Uall = jrand.orthogonal(key=uk, n=n)
# store the subspaces and the related quantities
Ua, Ub, x0a, x0b = Uall[:, :d], Uall[:, d:2*d], Uall[:, 2*d+1], Uall[:, 2*d+2]
x0a *= (jnp.sqrt(2)/2)
x0b *= (jnp.sqrt(2)/2)

# draw the coefficients
sa_ = spherical_uniform(key=sak, d=d, shape=(N_EXP, m))
sb_ = spherical_uniform(key=sbk, d=d, shape=(N_EXP, m))
sigma_ = spherical_uniform(sigmak, d=d, shape=(N_EXP,))

# get the thingies
uxas = sa_@Ua.T
uxbs = sb_@Ub.T
uxis = sigma_@Ua.T

results = np.empty((NRB, NS))
for ridx, rb in enumerate(Rb_):
    for sidx, s in enumerate(S_):
        xas = x0a*s + Ra*uxas
        xbs = x0b*s + rb*uxbs
        xis = x0a*s + Ra*uxis
        xi_is_a = is_a(xas.mean(axis=1), xbs.mean(axis=1), xis)
        eps = 1 - jnp.mean(xi_is_a, axis=0)
        results[ridx, sidx] = eps


# with fixed radii, show analytic result
f17 = partial(formula_S17, Ra=Ra, Rb=Rb_[0], D=d)
f18 = partial(formula_S18, Ra=Ra, Rb=Rb_[0], D=d)
ys17 = [f17(dx=dx) for dx in S_]
ys18 = [f18(dx=dx) for dx in S_]
# create plot
plt.plot(S_, results.squeeze(), label='empirical')
plt.plot(S_, ys17, label='theory - SI17', c='black', linestyle='--')
plt.plot(S_, ys18, label='theory - SI18', c='red', linestyle='--')
plt.title(f"Ra=1, Rb=1, m={m}, n={n}, d={d}")
plt.ylabel("Error probability")
plt.xlabel("Signal")
plt.legend()
plt.show()
# plt.savefig(f'./imgs/signal_n{n}.png')

