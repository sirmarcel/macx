import haiku as hk
from macx import EdgeFeature
import jax
import jax.numpy as jnp


def test_fire():
    n_rbf = 10
    l_max = 3

    @hk.without_apply_rng
    @hk.transform
    def edge_feature_fn(z, idx_i, idx_j, r_ij):
        return EdgeFeature(radial_fn="bessel", n_rbf=n_rbf, r_cut=5.0, l_max=l_max)(
            z=z, idx_i=idx_i, idx_j=idx_j, r_ij=r_ij
        )

    n = 5
    z = jnp.ones(n, dtype=int)
    idx_i = jnp.array([0, 0, 1, 1, 2, 2, 3, 4])
    idx_j = jnp.array([1, 2, 0, 2, 1, 0, 4, 3])

    R = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

    r_ij = jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j)

    inputs = {"z": z, "idx_i": idx_i, "idx_j": idx_j, "r_ij": r_ij}
    params = edge_feature_fn.init(jax.random.PRNGKey(0), **inputs)

    out = edge_feature_fn.apply(params, **inputs)

    assert out.shape == (len(idx_i), int(n_rbf * (l_max + 1) ** 2))
