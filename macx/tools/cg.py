###########################################################################################
# Modified from github.com/ACEsuit/mace code of Ilyes Batatia.
# We replace torch by jax. The original ASL license is kept below:
###########################################################################################

###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

import collections
from typing import List, Union

import e3nn_jax
from e3nn_jax import Irreps, Irrep
import jax.numpy as jnp
import numpy as np

# Based on e3nn

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = np.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = e3nn_jax.clebsch_gordan(ir_out.l, ir_left.l, ir.l)  # TODO dtype ?
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = np.einsum("jk,ijl->ikl", C_left.reshape(C_left.shape[0], -1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = np.zeros(
                        (
                            ir_out.dim,
                            *(irreps.dim for irreps in irrepss_left),
                            irreps_right.dim,  # TODO dtype?
                        )
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: str(x[:2]))  # TODO is this problem?


def U_matrix_real(
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = Irreps(irreps_out)
    irrepss = [Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = None

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            if stack is None:
                stack = base_o3.squeeze()[..., np.newaxis]
            else:
                stack = np.concatenate(
                    (stack, base_o3.squeeze()[..., np.newaxis]), axis=-1
                )
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze()[..., np.newaxis]
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, jnp.array(stack)]
    return out
