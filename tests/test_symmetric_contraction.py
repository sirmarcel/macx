import e3nn_jax
import haiku as hk
import jax
import jax.numpy as jnp
import torch
from e3nn import o3
from mace.modules.symmetric_contraction import SymmetricContraction
from macx.models.symmetric_contraction import SymmetricContraction as SymContr_jax


def test_symmetric_contraction():

    irreps_in = o3.Irreps("7x0y+7x1y+7x2y")
    irreps_out = o3.Irreps("0y")
    correlation = 2
    A_torch = torch.rand(3, 7, 9)
    sym_con = SymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        internal_weights=True,
        element_dependent=False,
        shared_weights=True,
    )
    torch_B = sym_con(A_torch, None)

    irreps_in_jax = e3nn_jax.Irreps(str(irreps_in))
    irreps_out_jax = e3nn_jax.Irreps(str(irreps_out))
    A_jax = jnp.array(A_torch)

    @hk.without_apply_rng
    @hk.transform
    def sym_con_jax(A):
        return SymContr_jax(irreps_in_jax, irreps_out_jax, correlation)(A)

    params = {
        "symmetric_contraction/~/Contraction_0e": {
            "weights_1": sym_con.contractions["1x0e"].weights["1"].detach().numpy(),
            "weights_2": sym_con.contractions["1x0e"].weights["2"].detach().numpy(),
        }
    }
    params = jax.tree_util.tree_map(jnp.array, params)
    jax_B = sym_con_jax.apply(params, A_jax).squeeze()

    assert jnp.allclose(jnp.array(torch_B.detach()), jax_B, atol=1e-6)
