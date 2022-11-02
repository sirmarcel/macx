from e3nn_jax import IrrepsArray, Linear


def add_mul_to_irrep_str(irreps, mul):
    return "+".join(map(lambda x: f"{mul}x{x}", irreps))


class ArrayLinear(Linear):
    def __init__(self, irreps_out, irreps_in, embedding_dim, channel_out=1):
        # base class has attribute irreps_in, don't overwrite it
        self.input_irreps = irreps_in
        mult_irreps_out = add_mul_to_irrep_str(irreps_out, embedding_dim)
        super().__init__(mult_irreps_out, channel_out)

    def __call__(self, x):
        *leading_dims, embedding_dim, _ = x.shape
        x = IrrepsArray(self.input_irreps, x).axis_to_mul()[..., None, :]  # channel dim
        out = super().__call__(x)
        out = out.mul_to_axis()
        return out.array if self.channel_out > 1 else out.array.squeeze(axis=-3)
