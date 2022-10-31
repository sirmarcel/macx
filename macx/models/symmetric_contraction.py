import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from ..tools.cg import U_matrix_real


class SymmetricContraction(hk.Module):
    def __init__(self, irreps_in, irreps_out, max_body_order, num_elements):
        super().__init__()
        self.irreps_out = irreps_out
        contractions = {}
        for _, irrep_out in irreps_out:
            contractions[str(irrep_out)] = Contraction(
                irreps_in, irrep_out, max_body_order, num_elements
            )
        self.contractions = contractions

    def __call__(self, A, node_attrs=None):
        Bs = []
        irrep_strs = []
        for _, irrep in self.irreps_out:
            irrep_str = str(irrep)
            Bs.append(self.contractions[irrep_str](A, node_attrs))
            irrep_strs.append(irrep_str)
        return jnp.concatenate(Bs, axis=-1)


class Contraction(hk.Module):
    def __init__(self, irreps_in, irrep_out, max_body_order, num_elements):
        super().__init__(f"Contraction_{irrep_out}")
        self.element_dependent = num_elements is not None
        self.max_body_order = max_body_order
        self.scalar_out = irrep_out.is_scalar()
        num_features = irreps_in.count("0e")
        coupling_irreps = e3nn.Irreps([irrep.ir for irrep in irreps_in])
        with jax.ensure_compile_time_eval():
            U_matrices = []
            for nu in range(1, max_body_order + 1):
                U_matrices.append(
                    U_matrix_real(
                        irreps_in=coupling_irreps,
                        irreps_out=irrep_out,
                        correlation=nu,
                    )[-1]
                )

            self.U_matrices = U_matrices

        if self.element_dependent:
            self.equation_init = "...ik,ekc,bci,be -> bc..."
            self.equation_weighting = "...k,kec,be->bc..."
        else:
            self.equation_init = "...ik,kc,bci -> bc..."
            self.equation_weighting = "...k,kc->c..."
        self.equation_contract = "bc...i,bci->bc..."
        weights = []
        for i in range(1, max_body_order + 1):
            num_params = self.U_matrices[i - 1].shape[-1]
            weight_shape = ()
            weights.append(
                hk.get_parameter(
                    f"weights_{i}",
                    (
                        [num_elements, num_params, num_features]
                        if self.element_dependent
                        else [num_params, num_features]
                    ),
                    init=hk.initializers.VarianceScaling(),
                )
            )
        self.weights = weights

    def __call__(self, A, node_attrs):
        B = jnp.einsum(
            self.equation_init,
            self.U_matrices[self.max_body_order - 1],
            self.weights[self.max_body_order - 1],
            *((A,) if node_attrs is None else (A, node_attrs)),
        )
        for corr in range(self.max_body_order - 1, 0, -1):
            c_tensor = jnp.einsum(
                self.equation_weighting,
                self.U_matrices[corr - 1],
                self.weights[corr - 1],
                *(() if node_attrs is None else (node_attrs,)),
            )
            c_tensor = c_tensor + B
            B = jnp.einsum(self.equation_contract, c_tensor, A)
        return B[..., None] if self.scalar_out else B
