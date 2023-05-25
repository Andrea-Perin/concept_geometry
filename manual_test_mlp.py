from jax import numpy as jnp, random as jrand, nn as jnn, vmap
import equinox as eqx
import matplotlib.pyplot as plt


def plot_decision(model):
    # plot decision boundary
    pts = jnp.linspace(-5, 5, 100)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # prob of being in inner circle
    preds = vmap(vmap(model))(pts).squeeze()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    contourf = ax.contourf(xv, yv, preds, levels=10, vmin=0, vmax=1)
    contour = ax.contour(xv, yv, preds, levels=[0, 0.5, 1], vmin=0, vmax=1, colors='red')
    ax.set_title(f"Decision boundary, 2-layer MLP")
    cbar = plt.colorbar(contourf)
    cbar.ax.set_ylabel("inner circle class. probability")
    cbar.add_lines(contour)
    return fig, ax


SEED = int(input("Insert seed: "))
rng = jrand.PRNGKey(SEED)

mlp = eqx.nn.MLP(2, 1, 4, 1, final_activation=jnn.sigmoid, key=rng)


w1 = jnp.arange(8, dtype=float).reshape(4, 2) / 12
b1 = jnp.ones(4, dtype=float) / 2
w2 = jnp.ones(4, dtype=float) / 2
b2 = jnp.ones(1, dtype=float) / 2

w1hat = w1[jnp.array([0, 2, 1, 3])]

# two shuffled networks
net = lambda x: jnn.sigmoid(w2@jnn.relu(w1@x+b1) + b2)
het = lambda x: jnn.sigmoid(w2@jnn.relu(w1hat@x+b1) + b2)
