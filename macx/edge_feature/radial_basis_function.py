import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import e3nn_jax as e3nn
from typing import Callable, Union

Array = Union[np.ndarray, jnp.ndarray]


def safe_mask(mask, fn: Callable, operand: Array, placeholder: float = 0.) -> Array:
    """
    Safe mask which ensures that gradients flow nicely. See also
    https://github.com/google/jax-md/blob/b4bce7ab9b37b6b9b2d0a5f02c143aeeb4e2a560/jax_md/util.py#L67
    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.
    Returns: New array with values either being the output of fn or the placeholder value.
    """
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


class BesselBasis(hk.Module):
    def __init__(self, n_rbf: int, r_cut: float):
        super().__init__()

        self.n_rbf = jnp.float32(n_rbf)
        self.r_cut = jnp.float32(r_cut)
        self.offsets = jnp.arange(1, self.n_rbf + 1, 1)  # shape: (n_rbf)

        self.cutoff_fn = CosineCutoff(r_cut)

    def __call__(self, r: Array) -> Array:
        """
        Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)
        Args:
                r (Array): Distances, shape: (P)
        Returns: The expanded distances, shape: (P,n_rbf)
        """
        _r = r[..., None]  # shape: (P,1)
        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x

        basis = safe_mask(mask=_r != 0,
                         fn=f,
                         operand=_r,
                         placeholder=0.)  # shape: (P, n_rbf)

        fc = self.cutoff_fn(r)  # shape: (P)

        return fc[..., None] * basis  # shape: (P, n_rbf)

class CosineCutoff(hk.Module):

    def __init__(self, r_cut: float):
        super().__init__()
        self.r_cut = jnp.float32(r_cut)

    def __call__(self, dR: Array) -> Array:
        cutoff_fn = lambda x: 0.5 * (jnp.cos(x * jnp.pi / self.r_cut) + 1.0)
        return safe_mask(mask=(dR < self.r_cut),
                         fn=cutoff_fn,
                         operand=dR,
                         placeholder=jnp.float32(0.))