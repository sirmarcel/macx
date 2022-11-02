import jax.numpy as jnp
import torch
from mace.modules.radial import BesselBasis as BesselBasis_torch


from macx.gnn.edge_features import BesselBasis

torch.manual_seed(42)


def tset_basis():
    hyperparameters_torch = {"BesselBasis": {"r_max": 10, "num_basis": 8}}
    hyperparameters_jax = {"BesselBasis": {"r_cut": 10, "n_rbf": 8}}

    r_test = [[1.0], [2.0], [3.0]]

    basis_torch = BesselBasis_torch(**hyperparameters_torch["BesselBasis"])
    basis_jax = BesselBasis(**hyperparameters_jax["BesselBasis"])

    result_torch = basis_torch(torch.tensor(r_test))
    result_jax = basis_jax(jnp.array(r_test))[:, 0]

    assert jnp.isclose(result_torch.numpy(), result_jax).all()
