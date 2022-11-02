def test_ace_equivariance():
    import e3nn_jax as e3nn
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from macx.gnn.edge_features import (BesselBasis, CombinedRadialBases,
                                        GaussianBasis)
    from macx.models import ACE
    from macx.tools.state_callback import state_callback
    from scipy.spatial.transform import Rotation

    rng = jax.random.PRNGKey(23)
    R = jnp.array(Rotation.from_euler("x", 52, degrees=True).as_matrix())

    @hk.without_apply_rng
    @hk.transform_with_state
    def ace(r, r_type):
        return ACE(
            2,
            4,
            10.0,
            3,
            embedding_irreps=[e3nn.Irrep("0e"), e3nn.Irrep("1o")],
            edge_feat_irreps=[e3nn.Irrep("0e"), e3nn.Irrep("1o")],
            node_types=[0, 1],
            edge_feat_kwargs={
                "radial_basis_factory": CombinedRadialBases,
                "radial_basis_kwargs": {
                    "n_rbfs": [2, 2],
                    "factories": [BesselBasis, GaussianBasis],
                },
            },
        )(r, r_type)

    jitted = jax.jit(jax.vmap(ace.apply, (None, 0, 0, 0)))

    r = jax.random.normal(rng, (1000, 2, 3))
    rotated_r = jnp.einsum("ij,baj->bai", R, r)
    r_type = jnp.tile(jnp.arange(2)[None], (1000, 1))

    params, state = jax.vmap(ace.init, (None, 0, 0), (None, 0))(rng, r, r_type)
    _, state = jitted(params, state, r, r_type)
    state, _ = state_callback(state, batch_dim=True)

    B, state = jitted(params, state, r, r_type)
    B_rot_r, _ = jitted(params, state, rotated_r, r_type)
    rot_B = B.at[:, :, :, 1:].set(jnp.einsum("ij,baej->baei", R, B[:, :, :, 1:]))
    diff = B_rot_r - rot_B
    assert jnp.abs(diff).max() < 1.0e-5
