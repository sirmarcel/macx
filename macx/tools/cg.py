import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irrep, Irreps, clebsch_gordan

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


def U_matrix_real_forreal(
    irreps_in,  #: Union[str, o3.Irreps],
    irrep_out,  #: Union[str, o3.Irreps],
    dtype=None,
):

    if len(irreps_in) == 2:
        return clebsch_gordan(irreps_in[0].l, irreps_in[1].l, irrep_out.l) 
    for lint in range(np.abs(irreps_in[-1].l - irrep_out.l), irreps_in[-1].l + irrep_out.l):
        u_mat += U_matrix_real_forreal(irreps_in[:-1], Irrep(str(lint)+"y")) * clebsch_gordan(lint, irreps_in[2].l, irrep_out.l)
    return u_mat
