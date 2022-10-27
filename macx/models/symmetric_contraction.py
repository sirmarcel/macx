import haiku as hk
import jax
from opt_einsum import contract

from ..jaxext import contract_with_irrep_array


class SymmetricContraction(hk.Module):
    def __init__(self, irreps_in, irreps_out, max_body_order):
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        contractions = {}
        for irrep_out in irreps_out:
            contractions[str(irrep_out)] = Contraction(
                self.irreps_in, irrep_out, max_body_order
            )

    def __call__(self, A):
        Bs = []
        irrep_strs = []
        for irrep in self.irreps_out:
            irrep_str = str(irrep)
            Bs.append(self.contractions[irrep_str](A).array)
            irrep_strs.append(irrep_str)
        return e3nn.IrrepsArray("+".join(irrep_strs), jnp.concatenate(Bs, axis=-1))


class Contraction(hk.Module):
    def __init__(self, irreps_in, irrep_out, max_body_order):
        super().__init__(f"Contraction_{irrep_out}")
        self.num_feauters = irreps_in.count("0e")
        self.coupling_irreps = e3nn.Irreps([irrep.ir for irrep in irreps_in])
        self.max_body_order = max_body_order
        with jax.ensure_compile_time_eval():
            U_matrices = []
            for nu in range(1, max_body_order + 1):
                U_matrices.append(
                    U_matrix_real(
                        irreps_in=self.coupling_irreps,
                        irreps_out=irrep_out,
                        correlation=nu,
                    )
                )
            self.U_matrices = U_matrices
        self.equation_init = "...ik,kc,bci -> bc..."
        self.equation_weighting = "...k,kc->c..."
        self.equation_contract = "bc...i,bci->bc..."
        weights = []
        for i in range(1, max_body_order + 1):
            num_params = self.U_matrices[i].shape[-1]
            weights.append(
                hk.get_parameter(f"weights_{i}"),
                [num_params, self.num_features],
                init=hk.initializers.VarianceScaling(),
            )
        self.weights = weights

    def __call__(self, A):
        B = contract_with_irrep_array(
            self.equation_init,
            self.U_matrices[self.max_body_order],
            self.weights[self.max_body_order],
            A,
        )
        for corr in range(self.max_body_order - 1, 0, -1):
            c_tensor = contract(
                self.equation_weighting,
                self.U_matrices[corr],
                self.weights[corr],
            )
            c_tensor = c_tensor + out
            B = contract_with_irrep_array(self.equation_contract, c_tensor, A)
        # [num_atoms, emb_dim * irrep_dim]
        return B
