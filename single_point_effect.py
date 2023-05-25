"""Showing the effects of a single point on the function defined by the MLP"""
from jax import numpy as jnp, random as jrand, vmap, nn as jnn, grad, hessian, jit
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    N_ITER = 300
    N = 10
    ALPHA = 1.2
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
    # x, y = jnp.array([[1, 1], [-1, -1]], dtype=float), jnp.array([1., 0.])
    # x, y = jnp.array([[1, 1]], dtype=float), jnp.array([1.])
    x, y = get_points(N, ALPHA)
    pts = jnp.linspace(lo, hi, npts)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # Start the optimizer
    opt = optax.adam(learning_rate=1e-4)
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

    # plotting function for the animation
    def plot_decision(n):
        # dirty globals trick
        globals()['mlp'], globals()['opt_state'], _ = make_step(
            globals()['mlp'], globals()['opt_state'], x, y)
        # plot decision boundary
        # prob of being in inner circle
        preds = vmap(vmap(globals()['mlp']))(pts).squeeze()
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
        title = ax.set_title(
            f"Decision boundary, width={WIDTH}. Iter n. {n+1}")
        avicii = jnp.linspace(0, 1, 11)
        contourf = ax.contourf(xv, yv, preds, levels=avicii, vmin=0, vmax=1)
        contour = ax.contour(xv, yv, preds, levels=avicii,
                             vmin=0, vmax=1, colors='red')
        clabel = ax.clabel(contour, inline=True, fontsize=10, zorder=6)
        pt1 = ax.scatter(*x[:N].T)
        pt2 = ax.scatter(*x[N:].T)
        # colormap
        return pt1, pt2, contourf, contour, clabel, title

    # plotting function for the animation
    def plot_decision_init():
        # plot decision boundary
        pts = jnp.linspace(lo, hi, npts)
        xv, yv = jnp.meshgrid(pts, pts)
        pts = jnp.stack((xv, yv), axis=-1)
        # prob of being in inner circle
        preds = vmap(vmap(mlp))(pts).squeeze()
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
        title = ax.set_title(f"Decision boundary, width={WIDTH}. Iter n. 0")
        avicii = jnp.linspace(0, 1, 11)
        contourf = ax.contourf(xv, yv, preds, levels=avicii, vmin=0, vmax=1)
        contour = ax.contour(xv, yv, preds, levels=avicii,
                             vmin=0, vmax=1, colors='red')
        pt1 = ax.scatter(*x[:N].T)
        pt2 = ax.scatter(*x[N:].T)
        # colormap
        clabel = ax.clabel(contour, inline=True, fontsize=10, zorder=6)
        cbar = plt.colorbar(contourf)
        cbar.ax.set_ylabel("class. probability")
        cbar.add_lines(contour)
        return pt1, pt2, contourf, contour, clabel, title, cbar

    # do the animation
    fig, ax = plt.subplots(figsize=(10, 10))
    anim = animation.FuncAnimation(
        fig, plot_decision, init_func=plot_decision_init, interval=50, blit=False, frames=N_ITER)
    anim.save(f'circles_MLP_w{WIDTH}_s{SEED}.gif')



    # ATTEMPT WITH INVERSE FUNCTION THEOREM
    myf = lambda x: mlp(x) - 0.5
    NNN = lambda x: (DG := grad(myf)(x))/(jnp.sqrt(jnp.sum(DG**2)))
    LLL = lambda v: lambda x: (hessian(myf)(x)@v)/(jnp.sqrt(jnp.sum(grad(myf)(x)**2)))

    N_TEST_PTS = int(1e4)
    ALPHA_TEST = (1 + ALPHA)/2
    optimal_pts = get_points(N_TEST_PTS, alpha=ALPHA_TEST)[0][N_TEST_PTS:]

    def plot_func(f):
        # plot decision boundary
        pts = jnp.linspace(lo, hi, npts)
        xv, yv = jnp.meshgrid(pts, pts)
        pts = jnp.stack((xv, yv), axis=-1)
        # prob of being in inner circle
        preds = vmap(vmap(f))(pts).squeeze()
        gf = grad(f)(jnp.ones(2))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
        title = ax.set_title(f"Decision boundary, width={WIDTH}. Iter n. 0")
        avicii = jnp.linspace(0, 1, 11)
        contourf = ax.contourf(xv, yv, preds, levels=avicii, vmin=0, vmax=1)
        contour = ax.contour(xv, yv, preds, levels=avicii,
                             vmin=0, vmax=1, colors='red')
        # vector field of gradient
        quiv = ax.quiver(1., 1., *gf)
        # colormap
        clabel = ax.clabel(contour, inline=True, fontsize=10, zorder=6)
        cbar = plt.colorbar(contourf)
        cbar.ax.set_ylabel("class. probability")
        cbar.add_lines(contour)
        return fig, ax
