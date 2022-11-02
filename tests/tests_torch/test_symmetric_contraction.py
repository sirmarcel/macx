import e3nn_jax
import haiku as hk
import jax
import jax.numpy as jnp
import torch
from e3nn import o3
from mace.modules.symmetric_contraction import \
    SymmetricContraction as SymmetricContraction_torch
from macx.models.symmetric_contraction import SymmetricContraction

torch.manual_seed(42)


def test_symmetric_contraction():
    hyperparameters_torch = {
        "irreps_in": o3.Irreps("3x0y+3x1y+3x2y"),
        "irreps_out": o3.Irreps("3x0y+3x1y"),
        "correlation": 2,
        "element_dependent": True,
        "num_elements": 2,
        "internal_weights": True,
        "shared_weights": True,
    }
    hyperparameters_jax = {
        "irreps_in": [e3nn_jax.Irrep("0y"), e3nn_jax.Irrep("1y"), e3nn_jax.Irrep("2y")],
        "irreps_out": [e3nn_jax.Irrep("0y"), e3nn_jax.Irrep("1y")],
        "max_body_order": 3,
        "n_node_type": 2,
        "n_feature": 3,
    }

    sc_torch = SymmetricContraction_torch(**hyperparameters_torch)

    @hk.without_apply_rng
    @hk.transform
    def sc_jax(x, y):
        return SymmetricContraction(**hyperparameters_jax)(x, y)

    n_nodes = 10
    n_features = 3
    irreps_in_dim = 9
    n_elements = 2

    x_test_torch = torch.randn(n_nodes, n_features, irreps_in_dim)
    y_test_torch = torch.randn(n_nodes, n_elements)

    x_test_jax = jnp.array(x_test_torch.numpy())
    y_test_jax = jnp.array(y_test_torch.numpy())

    key = jax.random.PRNGKey(42)

    params = sc_jax.init(key, x_test_jax, y_test_jax)
    params["symmetric_contraction/~/contraction_to_irrep_0e"][
        "coupling_weights_1"
    ] = sc_torch.contractions["3x0e"].weights["1"]
    params["symmetric_contraction/~/contraction_to_irrep_0e"][
        "coupling_weights_2"
    ] = sc_torch.contractions["3x0e"].weights["2"]
    params["symmetric_contraction/~/contraction_to_irrep_1o"][
        "coupling_weights_1"
    ] = sc_torch.contractions["3x1o"].weights["1"]
    params["symmetric_contraction/~/contraction_to_irrep_1o"][
        "coupling_weights_2"
    ] = sc_torch.contractions["3x1o"].weights["2"]

    params = jax.tree_util.tree_map(lambda x: jnp.array(x.detach().numpy()), params)

    result_jax = sc_jax.apply(params, x_test_jax, y_test_jax)
    result_jax = jnp.concatenate(
        (result_jax[:, :, 0], result_jax[:, :, 1:].reshape(n_nodes, -1)), axis=1
    )
    result_torch = sc_torch(x_test_torch, y_test_torch)
    assert jnp.isclose(result_torch.detach().numpy(), result_jax, atol=1e-3).all()
