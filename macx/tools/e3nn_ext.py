from functools import wraps
from itertools import chain, repeat
from typing import Callable, List, Optional, Sequence, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

__all__ = [
    "convert_irreps_array",
    "EquivariantLinear",
    "linear_weight_shapes",
    "WeightedTensorProduct",
]


def linear_weight_shapes(
    irreps_in: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
):
    r"""
    Calculate the shapes of the weights used by :class:EquivariantLinear.

    Return a list of tuples giving the shapes of the weights necessary to perform
    the equivariant linear layer specified by the arguments.

    Args:
        irreps_in (e3nn_jax.Irreps): the irreps of the input tensor.
        irreps_out (e3nn_jax.Irreps): the irreps of the output tensor.
    """
    shapes = []
    for mul_in, ir_in in irreps_in:
        for mul_out, ir_out in irreps_out:
            if ir_in == ir_out:
                shapes.append((mul_in, mul_out))
    return shapes


class FunctionalLinear(e3nn.FunctionalLinear):
    def __call__(
        self, ws: Union[List[jnp.ndarray], jnp.ndarray], input: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        input = input.convert(self.irreps_in)
        if input.ndim != 1:
            raise ValueError(
                "FunctionalLinear does not support broadcasting, "
                f"input shape is {input.shape}"
            )

        if not isinstance(ws, list):
            ws = self.split_weights(ws)

        paths = [
            ins.path_weight * w
            if ins.i_in == -1
            else (
                None
                if input.list[ins.i_in] is None
                # kfac_jax doesn't recognize this einsum as a proper linear layer:
                # jnp.einsum("uw,ui->wi", w, input.list[ins.i_in]),
                # use low level dot_general instead
                else ins.path_weight
                * jax.lax.dot_general(w, input.list[ins.i_in], (((0,), (0,)), ((), ())))
            )
            for ins, w in zip(self.instructions, ws)
        ]
        return self.aggregate_paths(paths, input.shape[:-1])


class FunctionalEquivariantLinear:
    def __init__(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: Sequence[e3nn.Irrep],
        channels_in: int,
        channels_out: Optional[int] = None,
        mix_channels: bool = False,
    ):
        if channels_out and not mix_channels:
            raise ValueError("mix_channels has to be True if channels_out is given")
        channels_out = channels_out or channels_in

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.channels_in = channels_in
        self.mix_channels = mix_channels
        self.channels_out = channels_out

        self.in_irreps_lin = e3nn.Irreps(
            (mul * channels_in if mix_channels else mul, ir) for mul, ir in irreps_in
        )
        self.out_irreps_lin = e3nn.Irreps(
            (channels_out if mix_channels else 1, ir) for ir in irreps_out
        )

    def __call__(self, weights, x):
        if self.mix_channels:
            x = x.axis_to_mul()
        lin = FunctionalLinear(self.in_irreps_lin, self.out_irreps_lin)

        def linear(x):
            return lin(weights, x)

        for _ in range(x.ndim - 1):
            linear = jax.vmap(linear)

        out = linear(x)
        if self.mix_channels:
            out = out.mul_to_axis()
        return out


class EquivariantLinear(hk.Module):
    r"""
    General equivariant linear layer.

    The input is assumed to be of shape
    [:math:`...`, :math:`N_\text{channels in}`, :math:`\sum_i m_i(2l_i+1)`] where
    :math:`i` runs over the input irreps, and :math:`m_i` is the multiplicity of the
    :math:`i`th irrep.

    Args:
        irreps_out (Sequence[e3nn_jax.Irrep]): sequence of output irreps.
        mix_channels (bool): default True, whether to mix the different input
            channels. Must be true if either :data:`channels_out`.
        channels_out (int): optional, the number of output channels, mixed from
            the input channels. If :data:`None`, it is set to
            :math:`N_\text{channels in}.
        weight_factory (Callable): optional, subnetwork that creates the linear
            weights.
    """

    def __init__(
        self,
        irreps_out: Sequence[e3nn.Irrep],
        mix_channels: bool = True,
        channels_out: Optional[int] = None,
        weight_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.irreps_out = irreps_out
        self.mix_channels = mix_channels
        self.channels_out = channels_out
        self.weight_factory = weight_factory

    def __call__(self, x, weight_factory_input=None):
        fel = FunctionalEquivariantLinear(
            x.irreps,
            self.irreps_out,
            x.shape[-2] if x.ndim > 1 else 0,
            self.channels_out,
            self.mix_channels,
        )
        weight_shapes = linear_weight_shapes(fel.in_irreps_lin, fel.out_irreps_lin)
        if self.weight_factory:
            weights = self.weight_factory(weight_shapes)(weight_factory_input)
        else:
            weights = []
            for i, shape in enumerate(weight_shapes):
                weights.append(
                    hk.get_parameter(
                        f"weight_{i}", shape, init=hk.initializers.VarianceScaling()
                    )
                )
        return fel(weights, x)


class WeightedTensorProduct(hk.Module):
    r"""
    Computed a weighted fully-connected tensor product.

    Performs an :data:`e3nn_jax.tensor_product`, and applies a
    :class:`EquivariantLinear` layer on the resulting tensors.

    Args:
        irreps_out (Sequence[e3nn.Irrep]): the irreps of the output tensor.
        weight_factory (Callabel): optional, constructor for a subnetwork that returns
            the linear weights.
        mix_channels (bool): default :data:`False`, whether to mix the channels
            (second to last dimension) of the output of the tensor product in the linear
            layer.
    """

    def __init__(
        self,
        irreps_out: Sequence[e3nn.Irrep],
        weight_factory: Optional[Callable] = None,
        mix_channels: bool = False,
    ):
        super().__init__()
        self.irreps_out = irreps_out
        self.weighted_sum = EquivariantLinear(
            irreps_out, mix_channels, weight_factory=weight_factory
        )

    def __call__(self, x, y, weight_factory_input=None):
        ia_out = e3nn.tensor_product(x, y, filter_ir_out=self.irreps_out).simplify()
        weighted_ia_out = self.weighted_sum(ia_out, weight_factory_input)
        return weighted_ia_out


def convert_irreps_array(*input_irreps: Sequence[e3nn.Irrep]):
    r"""
    Decorator to automatically convert :data:`jnp.ndarray`s to
    :data:`e3nn_jax.IrrepsArray`s.

    When applied to a function, it will automatically convert the passed arrays
    to :data:`IrrepsArray`, using the provided :data:`input_irreps`. The return values
    of the function array converted to arrays using :data:`IrrepsArray.array`.

    Args:
        input_irreps (Sequence[e3nn.Irrep]): the irreps of the respective positional
            arguments.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args = (
                e3nn.IrrepsArray(irrep, arg) if irrep else arg
                for irrep, arg in zip(chain(input_irreps, repeat(None)), args)
            )
            outs = func(*args, **kwargs)
            return (
                tuple(out.array for out in outs)
                if isinstance(outs, tuple)
                else outs.array
            )

        return wrapper

    return decorator
