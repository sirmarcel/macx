import e3nn_jax as e3nn
import jax.numpy as jnp
from opt_einsum import contract


def contract_with_irrep_array(
    instructions, *inputs: jnp.ndarray, irrep_array: e3nn.IrrepsArray
):
    if not isinstance(irrep_array, e3nn.IrrepsArray):
        raise TypeError
    for inp in inputs:
        if not isinstance(inp, jnp.ndarray):
            raise TypeError
    out = contract(instructions, *inputs, irrep_array.array)
    return e3nn.IrrepsArray(irrep_array.irreps, out)
