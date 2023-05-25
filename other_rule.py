from tqdm import tqdm
import numpy as np
from jax import numpy as jnp, random as jrand, vmap, jit
from jax.scipy.linalg import expm
from functools import wraps, partial
from datasets import spherical_uniform
import matplotlib.pyplot as plt


def euclidean_is_a(xa, xb, xi):
    """Takes the prototypes for the classes a and b plus the test point, and
    returns a boolean indicating if the test point belongs to class a according
    to the euclidean rule (i.e., whether or not the test point is closer to
    prototype a or b)."""
    return -(xa-xi)@(xa-xi) + (xb-xi)@(xb-xi) > 0


# def t_hat(x, xi, A):
#     return -((A@x)@xi)/(((A@A)@x)@xi)
#
#
# def x_hat(x, t, S, U, Uinv):
#     return U@jnp.diag(jnp.exp(t*S))@Uinv
#
#
# def x_update(x, xi, S, U, Uinv):
#     A = U @ jnp.diag(S) @ Uinv
#     t = t_hat(x, xi, A)
#     return x_hat(x, t, S, U, Uinv)


# # @jit
# def offset_is_a(xa, xb, xi, S, U, Uinv):
#     """Uses the iterative method to find the best approximations of the test
#     point that can be obtained by applying the group action to the
#     prototypes."""
#     print('before the loop: ', xa.shape)
#     print('before the loop: ', xb.shape)
#     N_ITER = 20  # just a random value
#     for n in range(N_ITER):
#         xa = x_update(xa, xi, S, U, Uinv)
#         xb = x_update(xb, xi, S, U, Uinv)
#     # now that the geometry is factored out, use the euclidean distance
#     return euclidean_is_a(xa, xb, xi)
def x_update(x, xi, D, Ul, Ur):
    A = Ul @ jnp.diag(D) @ Ur
    t_hat = -((A@x)@xi)/(((A@A)@x)@xi)
    return (Ul@jnp.diag(jnp.exp(t_hat*D))@Ur)@x


@jit
def offset_is_a(xa, xb, xi, D, Ul, Ur, n_iter=100):
    update = partial(x_update, xi=xi, D=D, Ul=Ul, Ur=Ur)
    for n in range(n_iter):
        xa = update(xa)
        xb = update(xb)
    return euclidean_is_a(xa, xb, xi)


def get_action_from(G):
    @wraps
    def exp_matr(t):
        return expm(t*G)
    return exp_matr


myexpm = jit(vmap(vmap(expm)))

# The idea now is that we have some group elements acting on the input
# points
SEED = 123
key = jrand.PRNGKey(SEED)
keys = jrand.split(key, num=100)

N_EXP = int(1e4)
n = 50

# generate the generator
B = jrand.normal(key=keys[0], shape=(n, n))
A = (B - B.T) / 2
D, Ul = jnp.linalg.eig(A)
D = 1j*(D.imag)
Ur = jnp.conj(Ul.T)

# g_t = get_action_from(A)

m = 1
Ra = 1.
Rb = 1.
S = 1.
d_ = jnp.arange(1, (ND := (n//2 - 1))+1)

# # # STUFF THAT DOES NOT CHANGE WITH D
# define the orthogonal basis, and make it orthonormal
uk, xak, tk, sak, sbk, sigmak = jrand.split(key=key, num=6)
Uall = jrand.orthogonal(key=uk, n=n)
# draw the times for the group action
allt_ = jrand.normal(key=tk, shape=(N_EXP, 2*m+1))
txa_, txb_, txi_ = allt_[:, :m], allt_[:, m:2*m], allt_[:, -1]
# and apply the transformation
print("creating the exponential matrices...")
# SMART METHOD: with the diagonalized form of the antisymmetric matrix,
# exponentiation is stupid quick
best_path = jnp.einsum_path('ij,nmj,jk->nmik', Ul, (txa_[..., None]*D), Ur)
get_exp = jit(partial(jnp.einsum, 'ij,nmj,jk->nmik', optimize=best_path[0]))
aexp_ = get_exp(Ul, txa_[..., None]*D, Ur).real
bexp_ = get_exp(Ul, txa_[..., None]*D, Ur).real
xexp_ = get_exp(Ul, txi_[..., None, None]*D, Ur).real
print("done creating the exponential matrices")

# Optimizing the most used function in the whole program
# batchmm_path = jnp.einsum_path('nmij,nmj->nmi', aexp_, jnp.ones((N_EXP, m, n)))
batchmm = jit(partial(jnp.einsum, 'nmij,nmj->nmi'))  # , optimize=batchmm_path[0]))

# # #
is_a_e = jit(vmap(euclidean_is_a, in_axes=(0, 0, 0)))
is_a_o = jit(vmap(offset_is_a, in_axes=(0, 0, 0, None, None, None)))
results = np.empty((ND, 2))
for idx, d in enumerate(tqdm(d_)):
    Ua, Ub, dx = Uall[:, :d], Uall[:, d:2*d], Uall[:, 2*d+1]
    x0a = jrand.normal(key=xak, shape=(n,))
    # draw the coefficients
    sa_ = spherical_uniform(key=sak, d=d, shape=(N_EXP, m))
    sb_ = spherical_uniform(key=sbk, d=d, shape=(N_EXP, m))
    sigma_ = spherical_uniform(key=sigmak, d=d, shape=(N_EXP,))
    # get the data (without group action)
    x0b = (x0a-S*dx)
    xas = x0a + Ra*(sa_@Ua.T)
    xbs = x0b + Rb*(sb_@Ub.T)
    xis = x0a + Ra*(sigma_@Ua.T)
    # and apply the group action + get the prototypes
    xa = batchmm(aexp_, xas).mean(axis=1)
    xb = batchmm(bexp_, xbs).mean(axis=1)
    xi = batchmm(xexp_, xis[:, None, :]).mean(axis=1)
    # now we have to perform the two distinct decision thresholds
    # on one hand, we use the simple euclidean distance
    # on the other, we use the group aware method
    is_a_euclidean = is_a_e(xa, xb, xi)
    is_a_offset = is_a_o(xa, xb, xi, D, Ul, Ur)
    euclidean_eps = 1 - jnp.mean(is_a_euclidean, axis=0)
    offset_eps = 1 - jnp.mean(is_a_offset, axis=0)
    # store results
    results[idx] = (euclidean_eps, offset_eps)

# create plot
plt.plot(d_, results[:, 0], label='euclidean')
plt.plot(d_, results[:, 1], label='offset')
plt.title(f"Ra=1, Rb=1, S=1, m={m}, n={n}")
plt.ylabel("Error probability")
plt.xlabel("D")
plt.legend()
plt.show()
