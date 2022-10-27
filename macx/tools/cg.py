import mace
from e3nn_jax import Irreps
from macx.tools import cg
from mace.tools import cg as mace_cg
from e3nn import o3
import jax.numpy as jnp

def U_matrix_real_easymode(
    irreps_in, #: Union[str, o3.Irreps],
    irreps_out, #: Union[str, o3.Irreps],
    correlation, #: int,
    normalization, #: str = "component",
    filter_ir_mid=None,
    dtype=None,
	):
	irreps_in_o3 = o3.Irreps(str(irreps_in))
	irreps_out_o3 = o3.Irreps(str(irreps_out))
	U_matrix_torch = mace_cd.U_matrix_real(irreps_in=irreps_in_o3, irreps_out=irreps_out_o3,
						  correlation=correlation, normalization=normalization,
						  filter_ir_mid=filter_ir_mid)
	U_matrix_jax = jnp.array(U_matrix_torch.cpu().detach().numpy(), dype=dtype) 
	
	
