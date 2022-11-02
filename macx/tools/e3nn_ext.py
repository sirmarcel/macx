from functools import wraps
from typing import Optional, Sequence

import e3nn_jax as e3nn
import haiku as hk


class GeneralLinear(hk.Module):
    r"""
    General equivariant linear layer.

    The input is assumed to be of shape
    [:math:`...`, :math:`N_\text{channels in}`, :math:`\sum_i m_i(2l_i+1)`] where
    :math:`i` runs over the input irreps, and :math:`m_i` is the multiplicity of the
    :math:`i`th irrep.

    Args:
        irreps_out (Sequence[e3nn_jax.Irrep]): sequence of output irreps.
        mix_channels (bool): default False, whether to mix the different input channels.
            Must be true if either :data:`channels_out` or :data:`new_channel_dim`
            is given.
        channels_out (int): optional, the number of output channels, mixed from
            the input channels. If :data:`None`, it is set to :math:`N_\text{channels in}.
        new_channel_dim (int): optional, the dimension of a new channel axes inserted
            before the input channels axis. If :data:`None`, no new channel axis
            is inserted. The new channels are mixed from the input channels.
    """

    def __init__(
        self,
        irreps_out: Sequence[e3nn.Irrep],
        mix_channels: bool = False,
        channels_out: Optional[int] = None,
        new_channel_dim: Optional[int] = None,
    ):
        if (channels_out or new_channel_dim) and not mix_channels:
            raise ValueError(
                "mix_channels has to be True if "
                "channels_out or new_channel_dim is given"
            )
        super().__init__()
        self.irreps_out = irreps_out
        self.mix_channels = mix_channels
        self.channels_out = channels_out
        self.new_channel_dim = new_channel_dim

    def __call__(self, x):
        emb_dim = x.array.shape[-2]
        channels_out = self.channels_out or emb_dim
        linear = (
            e3nn.Linear(self.irreps_out)
            if not self.mix_channels
            else e3nn.Linear(
                self.irreps_out, channels_out * (self.new_channel_dim or 1)
            )
        )
        out = linear(x)
        if self.new_channel_dim:
            out = out.reshape((*out.shape[:-2], self.new_channel_dim, channels_out, -1))
        return out


class WeightedTensorProduct(hk.Module):
    def __init__(
        self,
        irreps_x: Sequence[e3nn.Irrep],
        irreps_y: Sequence[e3nn.Irrep],
        irreps_out: Sequence[e3nn.Irrep],
    ):
        super().__init__()
        self.irreps_x = irreps_x
        self.irreps_y = irreps_y
        self.irreps_out = irreps_out
        self.weighted_sum = GeneralLinear(irreps_out)

    def __call__(self, x, y):
        ia_x = e3nn.IrrepsArray(self.irreps_x, x)
        ia_y = e3nn.IrrepsArray(self.irreps_y, y)
        ia_out = e3nn.tensor_product(ia_x, ia_y, filter_ir_out=self.irreps_out)
        weighted_ia_out = self.weighted_sum(ia_out)
        return weighted_ia_out.array


def convert_irreps_array(*irreps: Sequence[e3nn.Irrep]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert len(args) == len(irreps)
            args = (e3nn.IrrepsArray(irrep, arg) for irrep, arg in zip(irreps, args))
            outs = func(*args, **kwargs)
            return (
                tuple(out.array for out in outs)
                if isinstance(outs, tuple)
                else outs.array
            )

        return wrapper

    return decorator
