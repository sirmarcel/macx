import macx
import mace
from e3nn_jax import Irreps
from macx.tools import cg
from mace.tools import cg as mace_cg
from e3nn import o3
import jax.numpy as jnp
import numpy as np

# TODO can test CG orthogonality relations
# https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients

def _sorted_difference(x, y):
    return np.sort(x.reshape(-1)) - np.sort(y.reshape(-1))

def test_U_matrix_shape():
    irreps_in = Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = Irreps("1x0e + 1x1o")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3
    )[-1]
    assert u_matrix.shape == (3, 9, 9, 9, 21)

def test_U_matrix_mace_compare():
    irreps_in = Irreps("1x0e + 1x1o + 1x2e")
    irreps_out = Irreps("1x0e + 1x1o")
    irreps_in_mace = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out_mace = o3.Irreps("1x0e + 1x1o")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3
    )[-1]
    u_matrix_mace = mace_cg.U_matrix_real(
        irreps_in=irreps_in_mace, irreps_out=irreps_out_mace, correlation=3
    )[-1].numpy()
    assert np.max(np.abs(_sorted_difference(u_matrix, u_matrix_mace))) < 1e-6
    # TODO numpy code is correct in values BUT order is somehow off
    # TODO Issue 1: sorting of wigners by irrep_out in mace by out irrep is not unique
    # TODO Issue 2: Sometimes tensor product factors inside a wigner seem permuted
    #               Should they be treated as equivalent?
    # assert jnp.allclose(u_matrix, u_matrix_mace[-1])
