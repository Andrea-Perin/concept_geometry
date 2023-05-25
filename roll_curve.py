from functools import partial, wraps
from jax import numpy as jnp, grad, lax, random as jrand, nn as jnn, jit, value_and_grad
import equinox as eqx
import optax
from tqdm import tqdm
from tabulate import tabulate


# UTILS
def archimedean_spiral(theta, a, b, c=1):
    r = a + b * theta**(1/c)
    return r*jnp.stack([jnp.cos(theta), jnp.sin(theta)])


def arclen(theta, a, b):
    z = a+b*theta
    return ((z/(2*b))*jnp.sqrt(b**2+z**2)) - (b/2)*jnp.log(-z+jnp.sqrt(b**2+z**2))
    # return (b/2)*(theta*jnp.sqrt(1+theta**2)+jnp.arcsinh(theta))


def f_inv(f, y, x0, *fargs, **fkwargs):
    """Compute the inverse of a function at a point using Newton's method."""
    N_ITER = 20
    x = x0
    for _ in range(N_ITER):
        x -= (f(x, *fargs, **fkwargs) - y)/grad(f)(x, *fargs, **fkwargs)
    return x


grad_arclen = grad(arclen)


def inv_arclen(theta_0, a, b, dl):
    N_ITER = 5
    x = jnp.array(theta_0 + 1, float)  # just an initial guess
    y = dl + arclen(theta_0, a, b)
    for _ in range(N_ITER):
        x -= (arclen(x, a, b) - y)/grad_arclen(x, a, b)
    return x


def get_next_theta(t_prev, abdl):
    t = inv_arclen(t_prev, *abdl)
    return t, t


# ACTUAL FUNCTIONS TO BE USED
def get_spiral_points_og(n, a, b, theta_0=0., theta_1=2*jnp.pi, arclen_norm=True):
    spiral = partial(archimedean_spiral, a=a, b=b, c=1)
    if arclen_norm:
        total_len = arclen(theta_1, a, b) - arclen(theta_0, a, b)
        dl = total_len / n
        abdl = jnp.ones((n, 3))*jnp.array([a, b, dl])
        last_theta, thetas = lax.scan(get_next_theta, init=theta_0, xs=abdl)
        assert jnp.allclose(last_theta, theta_1)
    else:
        thetas = jnp.linspace(theta_0, theta_1, n)
    return spiral(thetas)


def get_spiral_points(n, a, b, theta_0=0., theta_1=2*jnp.pi):
    spiral = partial(archimedean_spiral, a=a, b=b, c=1)
    total_len = arclen(theta_1, a, b) - arclen(theta_0, a, b)
    dl = total_len / n
    abdl = jnp.ones((n, 3))*jnp.array([a, b, dl])
    last_theta, thetas = lax.scan(get_next_theta, init=theta_0, xs=abdl)
    return spiral(thetas)


def deformer(key, in_size, out_size, width, dist=1., as_noise=True):
    keys = jrand.split(key, num=4)
    lim = dist / jnp.sqrt(in_size)
    w1 = jrand.uniform(keys[0], (width, in_size), minval=-lim, maxval=lim)
    b1 = jrand.uniform(keys[1], (width,), minval=-lim, maxval=lim)
    lim = dist / jnp.sqrt(width)
    w2 = jrand.uniform(keys[2], (out_size, width), minval=-lim, maxval=lim)
    b2 = jrand.uniform(keys[3], (out_size,), minval=-lim, maxval=lim)

    def deform(x):
        h = jnn.relu(w1@x + b1)
        return (w2@h + b2) + (x if as_noise else 0)
    return deform


def menger_curvature(x, y, z):
    """Curvature in 2D"""
    xy = jnp.linalg.norm((x-y))
    yz = jnp.linalg.norm((y-z))
    zx = jnp.linalg.norm((z-x))
    s = (xy+yz+zx)/2
    A = jnp.sqrt(s*(s-xy)*(s-yz)*(s-zx))
    return (4*A)/(xy*yz*zx)


def tuple_bundle(x, n=3):
    lll = len(x)
    return tuple(x[i:lll-(n-i-1)] for i in range(n))


def avg_curvature(pts):
    return jnp.mean(vmap(menger_curvature)(*tuple_bundle(pts, n=3)))


def avg_class_distance(pts1, pts2):
    return jnp.sqrt(jnp.sum((pts1-pts2)**2, axis=0)).mean()


# SHOW THE CURVES
def plot_curves(pts1, pts2, pts1_d, pts2_d):
    n_pts = len(pts1.T)
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax1.set_aspect('equal')
    ax1.set_title(f"Normal, N={n_pts}")
    ax1.scatter(*pts1, label='spiral 1', alpha=.2)
    ax1.scatter(*pts2, label='spiral 2', alpha=.2)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax2.set_aspect('equal')
    ax2.set_title(f"Deformed, N={n_pts}")
    ax2.scatter(*pts1_d, label='deformed 1', alpha=.2)
    ax2.scatter(*pts2_d, label='deformed 2', alpha=.2)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    # plt.tight_layout()
    return fig, (ax1, ax2)


def generate_datasets(key, n_pts, dist_level: float = 1.0, shake_width: int = 1024):
    k1, k2, k3 = jrand.split(key, num=3)
    shake = deformer(k1, 2, 2, shake_width, dist=dist_level, as_noise=True)
    # step 1: create the spirals to be deformer
    a1 = jrand.uniform(key=k2)
    b1, b2 = 1., 1.1  # these are best left untouched
    a2 = a1 + 1  # introduce initial separation
    t11, t21 = 3*jnp.pi, 3*jnp.pi
    pts1 = get_spiral_points(n=n_pts, a=a1, b=b1, theta_0=1., theta_1=t11)
    pts2 = get_spiral_points(n=n_pts, a=a2, b=b2, theta_0=1., theta_1=t21)
    # get the deformed points
    pts1_shaken = vmap(shake)(pts1.T).T
    pts2_shaken = vmap(shake)(pts2.T).T
    # get the statistics
    R1_dist = avg_curvature(pts1_shaken.T)
    R2_dist = avg_curvature(pts2_shaken.T)
    S_dist = avg_class_distance(pts1_shaken, pts2_shaken)

    # define a loss on curvature and separation
    def curve_sep_loss(params):
        a1, b1, t10, t11, a2, b2, t20, t21 = params
        # we already know the transformation for spiral 1
        pts1 = get_spiral_points(n=n_pts, a=a1, b=b1, theta_0=t10, theta_1=t11)
        pts2 = get_spiral_points(n=n_pts, a=a2, b=b2, theta_0=t20, theta_1=t21)
        R1 = avg_curvature(pts1.T)
        R2 = avg_curvature(pts2.T)
        S = avg_class_distance(pts1, pts2)
        return (R2-R2_dist)**2 + (R1-R1_dist)**2 + (S-S_dist)**2

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(curve_sep_loss)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        ab = optax.apply_updates(params, updates)
        return ab, opt_state, loss

    t10, t20 = 0., 0.
    params = jnp.array([a1, b1, t10, t11, a2, b2, t20, t21], dtype=float)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    for n in tqdm(range(int(1e4))):
        params, opt_state, loss = step(params, opt_state)
    # after the iterations, lets see how better the estimates ArithmeticError
    a1, b1, t10, t11, a2, b2, t20, t21 = params
    if t10 < 0 or t20 < 0:
        print("BAD DRAW")
    pts1_opt = get_spiral_points(n=n_pts, a=a1, b=b1, theta_0=t10, theta_1=t11)
    pts2_opt = get_spiral_points(n=n_pts, a=a2, b=b2, theta_0=t20, theta_1=t21)
    # return the spirals and the deformed manifolds
    return pts1_opt, pts2_opt, pts1_shaken, pts2_shaken


def print_datasets_stats(pts1_opt, pts2_opt, pts1_shaken, pts2_shaken):
    R1_dist = avg_curvature(pts1_shaken.T)
    R2_dist = avg_curvature(pts2_shaken.T)
    S_dist = avg_class_distance(pts1_shaken, pts2_shaken)
    R1_opt = avg_curvature(pts1_opt.T)
    R2_opt = avg_curvature(pts2_opt.T)
    S_opt = avg_class_distance(pts1_opt, pts2_opt)
    results_table = [
            ["Optimal spirals", S_opt, R1_opt, R2_opt],
            ["Deformed manif.", S_dist, R1_dist, R2_dist]]
    headers = ["Separation", "Curvature 1", "Curvature 2"]
    print(tabulate(results_table, headers=headers))



def generate_manifolds_and_spirals(key, n_pts, dist_level: float = 1.0, shake_width: int = 1024):
    k1, k2, k3 = jrand.split(key, num=3)
    shake = deformer(k1, 2, 2, shake_width, dist=dist_level, as_noise=True)
    # step 1: create the distorted lines
    z0 = jrand.uniform(key=k1, shape=(2, ), minval=-5, maxval=5)
    dz = jrand.normal(key=k2, shape=(2, ))
    offset = jrand.uniform(key=k3, minval=.25, maxval=.5)
    l1 = z0 + jnp.linspace(0, 1, n_pts)[:, None]*dz
    l2 = l1 + jnp.array([0, offset])
    pts1_shaken = vmap(shake)(l1)
    pts2_shaken = vmap(shake)(l2)
    # step 2: compute the statistics
    R1_dist = avg_curvature(pts1_shaken)
    R2_dist = avg_curvature(pts2_shaken)
    S_dist = avg_class_distance(pts1_shaken, pts2_shaken)
    targets = (R1_dist, R2_dist, S_dist)
    # step 3: optimize

    # define a loss on curvature and separation
    def curve_sep_loss(params, n_pts, targets):
        a1, b1, t10, t11, a2, b2, t20, t21 = params
        R1_dist, R2_dist, S_dist = targets
        # we already know the transformation for spiral 1
        pts1 = get_spiral_points(n=n_pts, a=a1, b=b1, theta_0=t10, theta_1=t11)
        pts2 = get_spiral_points(n=n_pts, a=a2, b=b2, theta_0=t20, theta_1=t21)
        R1 = avg_curvature(pts1.T)
        R2 = avg_curvature(pts2.T)
        S = avg_class_distance(pts1, pts2)
        return (R2-R2_dist)**2 + (R1-R1_dist)**2 + (S-S_dist)**2

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(curve_sep_loss)(params, n_pts, targets)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    a1, b1, a2, b2 = 0., 1., 1., 1.
    t10, t11, t20, t21 = 0., 3*jnp.pi, 0., 3*jnp.pi
    params = jnp.array([a1, b1, t10, t11, a2, b2, t20, t21], dtype=float)
    opt = optax.adam(1e-1)
    opt_state = opt.init(params)
    for n in tqdm(range(int(1e4))):
        params, opt_state, loss = step(params, opt_state)
        print(loss)
    # after the iterations, lets see how better the estimates ArithmeticError
    a1, b1, t10, t11, a2, b2, t20, t21 = params
    if t10 < 0 or t20 < 0:
        print("BAD DRAW")
    pts1_opt = get_spiral_points(n=n_pts, a=a1, b=b1, theta_0=t10, theta_1=t11)
    pts2_opt = get_spiral_points(n=n_pts, a=a2, b=b2, theta_0=t20, theta_1=t21)
    # return the spirals and the deformed manifolds
    return pts1_opt, pts2_opt, pts1_shaken, pts2_shaken


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax import vmap
    # rng
    SEED = int(input("Insert seed: ") or "0")
    rng = jrand.PRNGKey(SEED)
    DIST = float(input("Insert distortion level: ") or "1")
    # generate data clouds
    sp1, sp2, dsp1, dsp2 = generate_datasets(key=rng, n_pts=50, shake_width=16, dist_level=DIST)
    # print some stats just to be sure
    print_datasets_stats(sp1, sp2, dsp1, dsp2)
    # and plot to be even more sure
    fig, ax = plot_curves(sp1, sp2, dsp1, dsp2)
    plt.show()
