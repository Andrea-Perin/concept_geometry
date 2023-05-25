"""Just checking whether learning a symmetric datasets gives the weights some symmetries."""
from jax import numpy as jnp, random as jrand, vmap, jit
from jax.nn import sigmoid
import equinox as eqx
import optax
from itertools import cycle


# sample points uniformly from a circle
def get_points(N, alpha):
    ts_ = jnp.linspace(0, 2*jnp.pi, N+1)[:-1]
    inner = jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    outer = alpha * jnp.stack((jnp.cos(ts_), jnp.sin(ts_)), axis=1)
    pts = jnp.concatenate((inner, outer))
    labs = jnp.concatenate((jnp.zeros(N), jnp.ones(N)))
    return pts, labs


# dataloader
def dataloader(key, xs, ys, batch_size=128, n_epochs=-1):
    permkey, _ = jrand.split(key)
    perm = jrand.permutation(permkey, len(xs))
    xs, ys = xs[perm], ys[perm]
    total = len(xs) // batch_size + 1
    idxs = cycle(range(total+1))
    for _, i in zip(range(total*n_epochs), idxs):
        xout = xs[i*batch_size:(i+1)*batch_size]
        yout = ys[i*batch_size:(i+1)*batch_size]
        yield xout, yout


if __name__ == "__main__":
    SEED = int(input("Insert seed (default: 0): ") or "0")
    rng = jrand.PRNGKey(SEED)

    dkey, mkey = jrand.split(rng)
    mod = eqx.nn.MLP(in_size=2, out_size=1, width_size=1024,
                     depth=1, final_activation=sigmoid, key=mkey)
    optim = optax.adam(learning_rate=1e-3)
    opt_state = optim.init(eqx.filter(mod, eqx.is_array))
    xs, ys = get_points(100, alpha=1.25)
    loader = dataloader(dkey, xs, ys, batch_size=128)

    # train the model
    def loss(mod: eqx.nn.MLP, x, y):
        pred_y = vmap(mod)(x)
        return -jnp.mean(y*jnp.log(pred_y) + (1-y)*jnp.log(1-pred_y))

    @eqx.filter_jit
    def make_step(mod: eqx.nn.MLP, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(loss)(mod, x, y)
        updates, opt_state = optim.update(grads, opt_state, mod)
        mod = eqx.apply_updates(mod, updates)
        return mod, opt_state, loss_value

    # actual training
    for (xs, ys) in loader:
        mod, opt_state, _ = make_step(mod, opt_state, xs, ys)

    w3d = jnp.concatenate((mod.layers[0].weight, mod.layers[0].bias[:, None]), axis=1)
    # and now let us check the weights
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*w3d.T)
    plt.show()
