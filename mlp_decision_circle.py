"""Showing the effects of a single point on the function defined by the MLP"""
from jax import numpy as jnp, random as jrand, vmap, nn as jnn, grad, hessian, jit
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import trange


def get_points(N, alpha):
    ts_ = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha * jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    pts = jnp.concatenate((inner, outer))
    labs = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return pts, labs


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)
    mkey, dkey = jrand.split(rng)
    WIDTH = 1024
    N_ITER = int(1e4)
    # train and test params
    N = 4
    ALPHA = 1.2
    N_TEST_PTS = int(1e4)
    ALPHA_TEST = (1 + ALPHA)/2
    # plotting params
    lo, hi, npts = -5, 5, 100

    # Instantiate model
    mlp_kwargs = dict(
        in_size=2,
        out_size=1,
        width_size=WIDTH,
        depth=1,
        final_activation=lambda x: jnn.sigmoid(x).squeeze(),
        key=mkey)
    mlp = eqx.nn.MLP(**mlp_kwargs)

    # Select the training point and the test points
    x, y = get_points(N, ALPHA)
    pts = jnp.linspace(lo, hi, npts)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # Select the "optimal separator" test points
    sep_pts = get_points(N_TEST_PTS, alpha=ALPHA_TEST)[0][N_TEST_PTS:]
    # Start the optimizer
    opt = optax.adam(learning_rate=1e-5)
    opt_state = opt.init(eqx.filter(mlp, eqx.is_array))

    # Training functions
    def loss(model: eqx.nn.MLP, x, y):
        pred_y = vmap(model)(x).squeeze()
        return -jnp.mean(y*jnp.log(pred_y) + (1-y)*jnp.log(1-pred_y))

    @eqx.filter_jit
    def make_step(model: eqx.nn.MLP, opt_state, x, y,):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for n in trange(N_ITER):
        mlp, opt_state, _ = make_step(mlp, opt_state, x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(vmap(mlp)(sep_pts))
    plt.show()

    # ATTEMPT WITH INVERSE FUNCTION THEOREM
    # myf = lambda x: mlp(x) - 0.5
    # NNN = lambda x: (DG := grad(myf)(x))/(jnp.sqrt(jnp.sum(DG**2)))
    # LLL = lambda v: lambda x: (hessian(myf)(x)@v)/(jnp.sqrt(jnp.sum(grad(myf)(x)**2)))

    def polygon_n(alpha):
        return jnp.ceil(jnp.pi/(jnp.arccos(1/alpha)))

    def spherepack_n(alpha):
        return jnp.ceil(jnp.pi/(2*jnp.arcsin((alpha-1)/(2*(alpha+1)))))

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.xscale('log')
    plt.yscale('log')
    alphas = jnp.linspace(1, 10, 1000)
    ax.step(alphas, polygon_n(alphas), label='polygon')
    ax.step(alphas, spherepack_n(alphas), label='sphere packing')
    ax.set_xlabel("alpha")
    ax.set_ylabel("N")
    plt.legend()
    plt.show()
