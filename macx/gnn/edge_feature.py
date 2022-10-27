from functools import partial
from typing import Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import sh
from typing import Union
from functools import partial
from itertools import product

from .radial_basis_function import BesselBasis

Array = Union[np.ndarray, jnp.ndarray]


def generate_irreps_string(l_max, n_rbf):
    degrees = np.arange(l_max + 1)
    s = []
    for n, y, d in zip([n_rbf] * len(degrees), "y" * len(degrees), degrees):
        s += [str(n) + "x" + str(d) + y]
    s = "+".join(s)
    return s


class EdgeFeature(hk.Module):
    def __init__(
        self, radial_fn: str, n_rbf: int, r_cut: float, l_max: int, z_max: int = 100
    ):

        super().__init__()

        self.irreps_out = generate_irreps_string(l_max=l_max, n_rbf=n_rbf)

        self.embed_fn = hk.Embed(vocab_size=z_max, embed_dim=1)

        if radial_fn == "bessel":
            self.radial_fn = BesselBasis(n_rbf=n_rbf, r_cut=r_cut)

        self.spherical_fn = partial(
            sh,
            irreps_out=np.arange(l_max + 1).tolist(),
            normalize=True,
            normalization="integral",
        )

    def __call__(
        self, z: Array, idx_i: Array, idx_j: Array, r_ij: Array, *args, **kwargs
    ):
        """

        Args:
            z (Array): shape: (P)
            idx_i (Array): shape: (P) central atoms
            idx_j (Array): shape: (P) neighbour atoms
            r_ij (Array): shape: (P,3) atom-pair vectors
            *args ():
            **kwargs ():

        Returns:
            A_ij (Array): shape: (P,n_rbf,m_tot)
        """

        x = self.embed_fn(z)  # shape: (P,1)
        x_i = x[idx_i][..., None]  # shape: (P,1,1)
        x_j = x[idx_j][..., None]  # shape: (P,1,1)

        d_ij = jnp.linalg.norm(r_ij, axis=-1)  # shape: (P)
        rbf_ij = self.radial_fn(d_ij)  # shape: (P,n_rbf)
        sph_ij = self.spherical_fn(input=r_ij)  # shape: (P,m_tot)  m_tot = (l_max+1)^2
        A_ij = (
            rbf_ij[:, None, :] * sph_ij[:, :, None] * x_i * x_j
        )  # shape: (P,m_tot,n_rbf)

        return e3nn.IrrepsArray(self.irreps_out, A_ij.reshape(A_ij.shape[0], -1))



# implemented elsewhere
#
# class Aggregation(hk.Module):
#     def __init__(self, z_max=100):
#         super().__init__()
#
#     def __call__(self,
#                  z,
#                  idx_i: Array,
#                  idx_j: Array,
#                  *args,
#                  **kwargs) -> Array:
#         """
#
#         Args:
#             z (Array): atomic types, shape: (n)
#             idx_i (Array): central index, shape: (P)
#             idx_j (Array): neighboring index, shape: (P)
#             *args (Array):
#             **kwargs (Array):
#
#         Returns:
#
#         """
#         A_i = jax.ops.segment_sum(A_ij, segment_ids=idx_j, num_segments=len(z))  # shape: (n,m_tot,n_rbf)
#         return A_i
