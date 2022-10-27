import macx
import mace
from e3nn_jax import Irreps
from macx.tools import cg
from mace.tools import cg as mace_cg
from e3nn import o3
import jax.numpy as jnp

# TODO can test CG orthogonality relations
# https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients


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
    print("initialized irreps jax")
    irreps_in_mace = o3.Irreps("1x0e + 1x1o + 1x2e")
    irreps_out_mace = o3.Irreps("1x0e + 1x1o")
    print("initialized irreps e3nn")
    u_matrix = cg.U_matrix_real(
        irreps_in=irreps_in, irreps_out=irreps_out, correlation=3
    )[-1]
    print("initialized matrix jax")
    u_matrix_mace = mace_cg.U_matrix_real(
        irreps_in=irreps_in_mace, irreps_out=irreps_out_mace, correlation=3
    )[-1]
    print("initialized matrix e3nn")
    assert jnp.allclose(u_matrix, u_matrix_mace.cpu().detach().numpy())
