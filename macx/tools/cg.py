import jax.numpy as jnp
import mace
from e3nn import o3
from e3nn_jax import Irrep, Irreps
from mace.tools import cg as mace_cg

from macx.tools import cg


def U_matrix_real(  # easymode
    irreps_in,  #: Union[str, o3.Irreps],
    irreps_out,  #: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_in_o3 = o3.Irreps(str(irreps_in))
    irreps_out_o3 = o3.Irreps(str(irreps_out))
    U_matrix_torch = mace_cg.U_matrix_real(
        irreps_in=irreps_in_o3,
        irreps_out=irreps_out_o3,
        correlation=correlation,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
    )
    U_matrix_jax = [
        Irrep(str(U_mat))
        if isinstance(U_mat, o3.Irrep)
        else jnp.array(U_mat.cpu().detach().numpy(), dtype=dtype)
        for U_mat in U_matrix_torch
    ]
    return U_matrix_jax
