from jax import numpy as jnp, vmap
import matplotlib.pyplot as plt
import matplotlib as mpl


def polygon(x):
    """Smallest number of sides for a regular polygon separating the two circles."""
    return jnp.ceil(jnp.pi/(jnp.arccos(1/x)))


def plot_decision(model, alpha, pts_inn, pts_out, npts=100, mult=1.5):
    """plotting function for the decision boundary of a 2D MLP"""
    # plot decision boundary
    pts = jnp.linspace(lo:=pts_out.min()*mult, hi:=pts_out.max()*mult, npts)
    xv, yv = jnp.meshgrid(pts, pts)
    pts = jnp.stack((xv, yv), axis=-1)
    # prob of being in inner circle
    preds = vmap(vmap(model))(pts).squeeze()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    title = ax.set_title(f"Decision boundary, alpha={alpha}, N={pts_inn.shape[0]}")
    # plot contours
    avicii = jnp.linspace(0, 1, 11)
    contourf = ax.contourf(xv, yv, preds, levels=avicii, vmin=0, vmax=1)
    contour = ax.contour(xv, yv, preds, levels=avicii,
                         vmin=0, vmax=1, colors='red', alpha=0.5)
    # plot scatter points
    inn = ax.scatter(*pts_inn.T, marker='x', color='black')
    out = ax.scatter(*pts_out.T, marker='o', color='black')
    # colormap
    clabel = ax.clabel(contour, inline=True, fontsize=10, zorder=6)
    cbar = plt.colorbar(contourf)
    cbar.ax.set_ylabel("class. probability")
    cbar.add_lines(contour)
    return fig, ax


def plot_loss(losses, a):
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(jnp.arange(len(losses)), losses, '-o')
    ax.set_xlabel('Batch number')
    ax.set_ylabel('Loss')
    ax.set_title(f"Loss trajectory, alpha={a:.3f}")
    return fig, ax


def plot_error_fracs(err_dict):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('error fraction')
    ax.set_title('Errors vs. N')
    alphas = sorted(err_dict.keys())
    cmap = mpl.colormaps['viridis']
    for idx, a in enumerate(alphas):
        sorted_ns = sorted(err_dict[a].keys())
        inns = [err_dict[a][s]['inn'] for s in sorted_ns]
        outs = [err_dict[a][s]['out'] for s in sorted_ns]
        col = cmap((idx+1)/len(alphas))
        ax.plot(sorted_ns, inns, linestyle='-', color=col, alpha=0.5)
        ax.plot(sorted_ns, outs, linestyle='-.', color=col, alpha=0.5)
    norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label('alphas')
    lcargs = dict(linestyle='-', color='black')
    linner = mpl.lines.Line2D([0], [0], label='inner error', **lcargs)
    louter = mpl.lines.Line2D([0], [0], label='outer error', **lcargs)
    ax.legend(handles=[linner, louter])
    return fig, ax


def plot_n_alpha(alphas, ns, err, width):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('N')
    ax.set_title('N vs alpha')
    # take sum across the vmapped evaluate_ensemble
    ax.plot(alphas, ns, label='empirical')
    ax.plot(XS:=jnp.linspace(1, 1.5, 1000), polygon(XS), label='theory')
    ax.set_title(f"N vs alpha, eps={err}, W={width}")
    ax.set_xlabel(r"$R_M/r_m$")
    ax.set_ylabel("N")
    ax.legend()
    return fig, ax
