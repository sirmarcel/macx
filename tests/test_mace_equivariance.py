def test_mace_equivariance():
    import e3nn_jax as e3nn
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from macx.models.mace import MACE
    from macx.tools.state_callback import state_callback
    from scipy.spatial.transform import Rotation

    rng = jax.random.PRNGKey(23)
    R = jnp.array(Rotation.from_euler("x", 52, degrees=True).as_matrix())

    @hk.without_apply_rng
    @hk.transform_with_state
    def mace(r, r_type):
        return MACE(
            5,
            5,
            10.0,
            4,
            embedding_irreps=[
                [e3nn.Irrep("0e"), e3nn.Irrep("1o")],
                [e3nn.Irrep("0e"), e3nn.Irrep("1o")],
            ],
            edge_feat_irreps=[e3nn.Irrep("0e"), e3nn.Irrep("1o")],
            node_types=[0, 1, 2, 3, 4],
        )(r, r_type)

    r = jax.random.normal(rng, (1000, 5, 3))
    rotated_r = jnp.einsum("ij,baj->bai", R, r)
    r_type = jnp.tile(jnp.arange(5)[None], (1000, 1))

    jitted = jax.jit(jax.vmap(mace.apply, (None, 0, 0, 0)))

    params, state = jax.vmap(mace.init, (None, 0, 0), (None, 0))(rng, r, r_type)
    _, state = jitted(params, state, r, r_type)
    state, _ = state_callback(state, batch_dim=True)

    B, state = jitted(params, state, r, r_type)
    B_rot, _ = jitted(params, state, rotated_r, r_type)
    rot_B = B.at[:, :, :, 1:].set(jnp.einsum("ij,baej->baei", R, B[:, :, :, 1:]))
    diff = B_rot - rot_B
    assert jnp.abs(diff).max() < 2.0e-5
