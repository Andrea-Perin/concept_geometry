from typing import (
    Callable,
    Literal,
    Union,
)
import jax.nn as jnn
import jax.random as jrandom
from equinox.nn import Linear, Dropout, BatchNorm, Lambda, Sequential, State
from equinox import Module


class StupidMLP(Module):
    lin1: Linear
    drop: Dropout
    lin2: Linear

    def __init__(self, in_size, out_size, width_size, pdrop, *, key):
        keys = jrandom.split(key)
        self.lin1 = Linear(in_size, width_size, use_bias=True, key=keys[0])
        self.drop = Dropout(pdrop)
        self.lin2 = Linear(width_size, out_size, use_bias=True, key=keys[1])

    def __call__(self, x, key):
        x = self.lin1(x)
        x = jnn.relu(x)
        x = self.drop(x, key=key)
        x = self.lin2(x)
        return jnn.sigmoid(x)


class FatLinear(Linear):
    """A linear layer with some padding

    that is

    batchnorms and dropouts"""
    #drop: Dropout
    bnorm: BatchNorm
    act: Callable

    def __init__(
            self,
            dropout_pct: float,
            batch_norm: bool,
            activation=jnn.relu,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.drop = Dropout(p=dropout_pct)
        self.bnorm = BatchNorm(
            input_size=kwargs['out_features'], axis_name='batch')
        self.act = activation

    def __call__(self, x, state, key):
        x = super().__call__(x)
        x, state = self.bnorm(x, state)
        x = self.act(x)
        # x = self.drop(x, key=key)
        return x, state


class FatMLP(Module):
    layers: list
    last_lin: Linear
    last_act: Callable

    def __init__(
            self,
            in_size: Union[int, Literal["scalar"]],
            out_size: Union[int, Literal["scalar"]],
            width_size: int,
            depth: int,
            activation: Callable = jnn.relu,
            final_activation: Callable = jnn.sigmoid,
            use_bias: bool = True,
            use_final_bias: bool = True,
            dropout_pct: float = 0.,
            batch_norm: bool = False,
            *,
            key: jrandom.PRNGKey,
            **kwargs,):
        keys = jrandom.split(key, depth + 1)
        self.layers = []
        if depth < 1:
            raise ValueError("Depth must be at least 1.")
        # first layer
        fl_kwargs = dict(use_bias=use_bias,
                         dropout_pct=dropout_pct,
                         batch_norm=batch_norm,
                         activation=activation)
        first_sizes = dict(in_features=in_size, out_features=width_size)
        self.layers.append(FatLinear(**first_sizes, **fl_kwargs, key=keys[0]))
        # other layers
        other_sizes = dict(in_features=width_size, out_features=width_size)
        for d in range(depth-1):
            self.layers.append(
                FatLinear(**other_sizes, **fl_kwargs, key=keys[d+1]))
        # last one
        last_sizes = dict(in_features=width_size, out_features=out_size)
        self.last_lin = Linear(
            **last_sizes, use_bias=use_final_bias, key=keys[-1])
        self.last_act = final_activation

    def __call__(self, x, state, key):
        act_keys = jrandom.split(key, num=len(self.layers))
        for ak, layer in zip(act_keys, self.layers):
            x, state = layer(x, state, ak)
        x = self.last_lin(x)
        return self.last_act(x), state


if __name__ == "__main__":
    from jax import vmap
    kee = jrandom.PRNGKey(0)
    fml = FatMLP(2, 1, 1024, 1, dropout_pct=0.1, batch_norm=True, key=kee)
    state = State(fml)
    print(state)

    inp = jrandom.normal(kee, (123, 2))
    batch_mod = vmap(fml, axis_name='batch', in_axes=(
        0, None, None), out_axes=(0, None))
    out, state = batch_mod(inp, state, kee)
    print(out)
