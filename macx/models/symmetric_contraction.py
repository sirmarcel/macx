from collections.abc import Sequence
from typing import Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from ..tools.cg import U_matrix_real


class SymmetricContraction(hk.Module):
    r"""
    Create higher body-order tensors transformig according to some irreps.

    This class is a wrapper of
    :class:`~macx.tools.symmetric_contraction.ContractionToIrrep` to create
    multiple concatenated higher body-order tensors each transforming according
    to different irreps. For more details see the documentation of
    :class:`~macx.tools.symmetric_contraction.ContractionToIrrep`.

    The output array has shape
    [:math:`N_\text{batch}`, :math:`N_\text{feature}, :math:`\sum_{i}(2 l_i + 1)`],
    where :math:`i` runs over :data:`irreps_in`.

    Args:
        irreps_out (Sequence[e3nn_jax.Irrep]): the irreps of the concatenated output
            tensors.
        irreps_in (Sequence[e3nn_jax.Irrep]): the irreps of the concatenated input
            tensors.
        max_body_order (int): output tensors up to body-order :data:`max_body_order`
            are calculated and their sum is returned.
    """

    def __init__(
        self,
        irreps_out: Sequence[e3nn.Irrep],
        irreps_in: Sequence[e3nn.Irrep],
        n_feature: int,
        max_body_order: int,
        n_node_type: int,
    ):
        super().__init__()
        self.irreps_out = irreps_out
        contractions = {}
        for irrep_out in irreps_out:
            contractions[str(irrep_out)] = ContractionToIrrep(
                irrep_out, irreps_in, n_feature, max_body_order, n_node_type
            )
        self.contractions = contractions

    def __call__(self, A, node_attrs=None):
        Bs = []
        for irrep in self.irreps_out:
            Bs.append(self.contractions[str(irrep)](A, node_attrs))
        return jnp.concatenate(Bs, axis=-1)


class ContractionToIrrep(hk.Module):
    r"""
    Create higher body-order tensors transforming according to some irrep.

    Taking as input concatenated 2-body tensors that transform according to
    :data:`irreps_in`, it calculates their tensor products using the generalized
    Clebsch--Gordan coefficients, to return a sum of higher body-order tensors
    that transforms as :data:`irrep_out`.

    Input array must have the shape
    [:math:`N_\text{batch}`, :math:`N_\text{feature}, :math:`\sum_{i}(2 l_i + 1)`],
    where :math:`i`, runs over :data:`irreps_in`. The output array has the shape
    [:math:`N_\text{batch}`, :math:`N_\text{feature}`, :math:`2 l_\text{out} + 1`].

    Args:
        irrep_out (e3nn_jax.Irrep): the irrep of the output tensor
        irreps_in (Sequence[e3nn_jax.Irrep]): the irreps of the concatenated input
            tensors.
        n_feature (int): the number of features of the input tensors.
        max_body_order (int): output tensors up to body-order :data:`max_body_order`
            are calculated and their sum is returned.
    """

    def __init__(
        self,
        irrep_out: e3nn.Irrep,
        irreps_in: Sequence[e3nn.Irrep],
        n_feature: int,
        max_body_order: int,
        n_node_type: int,
    ):
        super().__init__(f"contraction_to_irrep_{irrep_out}")
        if max_body_order < 2:
            raise ValueError(
                "Input has body-order 2, therefore max_body_order "
                f"has to be at least 2, got {max_body_order}"
            )
        self.correlation = max_body_order - 2
        # U matrix for scalar irrep_out is missing its first dimension (size 1),
        # we need to add it manually:
        self.scalar_out = irrep_out.is_scalar()

        U_matrices = []
        for nu in range(1, max_body_order):
            U = U_matrix_real(
                irreps_in=e3nn.Irreps(irreps_in),
                irreps_out=irrep_out,
                correlation=nu,
            )[-1]
            if irreps_in == [e3nn.Irrep("0e")]:
                # U matrix for single scalar input is missing all but its
                # last dimension (all size 1), we need to add it manually
                for _ in range(nu):
                    U = U[None]
            U_matrices.append(U)
        self.U_matrices = U_matrices

        self.equation_init = "...ik,ekc,bci,be -> bc..."
        self.equation_weighting = "...k,ekc,be->bc..."
        self.equation_contract = "bc...i,bci->bc..."
        weights = []
        for nu in range(1, max_body_order):
            # number of ways irrep_out can be created from irreps_in at body order nu:
            n_coupling = self.U_matrices[nu - 1].shape[-1]
            weights.append(
                hk.get_parameter(
                    f"coupling_weights_{nu}",
                    [n_node_type, n_coupling, n_feature],
                    init=hk.initializers.VarianceScaling(),
                )
            )
        self.weights = weights

    def __call__(self, A, node_types):
        # node_types is onehot encoded, it selects the index of weights,
        # and is usually faster than indexing
        B = jnp.einsum(
            self.equation_init,
            self.U_matrices[self.correlation],
            self.weights[self.correlation],
            A,
            node_types,
        )
        for corr in reversed(range(self.correlation)):
            c_tensor = jnp.einsum(
                self.equation_weighting,
                self.U_matrices[corr],
                self.weights[corr],
                node_types,
            )
            c_tensor = c_tensor + B
            B = jnp.einsum(self.equation_contract, c_tensor, A)
        return B[..., None] if self.scalar_out else B
