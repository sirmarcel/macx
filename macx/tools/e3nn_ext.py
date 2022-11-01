from e3nn_jax import IrrepsArray, Linear


def get_irreps_str(irreps, embedding_dim):
    return "+".join(map(lambda x: f"{embedding_dim}x{x}", irreps))


class ArrayLinear(Linear):
    def __init__(self, irreps_out, irreps_in, embedding_dim, channel_out=1):
        mult_irreps_out = get_irreps_str(irreps_out, embedding_dim)
        mult_irreps_in = get_irreps_str(irreps_in, embedding_dim)
        super().__init__(mult_irreps_out, channel_out, irreps_in=mult_irreps_in)
        self.mult_irreps_in = mult_irreps_in
        self.channel_out = channel_out

    def __call__(self, x):
        *leading_dims, embedding_dim, irreps_dim = x.shape
        x = IrrepsArray(self.mult_irreps_in, x.reshape(*leading_dims, 1, -1))
        out = super().__call__(x)
        out = out.array.reshape(*leading_dims, self.channel_out, embedding_dim, -1)
        return jnp.squeeze(out, axis=-3) # remove singular channel dimension
