from typing import Sequence

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
from jax import ops

from ..gnn import GraphNeuralNetwork, MessagePassingLayer
from ..gnn.edge_features import EdgeFeatures
from .symmetric_contraction import SymmetricContraction


def to_onehot(features, node_types):
    ones = []
    for i, e in enumerate(node_types):
        ones.append(jnp.where(features == e, jnp.ones(1), jnp.zeros(1))[..., None])
    return jnp.concatenate(ones, axis=-1)


class ACELayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        shared,
        *,
        max_body_order: int,
        embedding_irreps: Sequence[e3nn.Irrep],
        mix_atomic_basis: bool = True,
    ):
        super().__init__(ilayer, shared)
        self.mix_atomic_basis = mix_atomic_basis
        self.symmetrize = SymmetricContraction(
            embedding_irreps,
            self.edge_feat_irreps,
            self.embedding_dim,
            max_body_order,
            self.n_node_type,
        )
        if mix_atomic_basis:
            self.atomic_basis_weights = hk.get_parameter(
                "atomic_basis_weights",
                [len(self.edge_feat_irreps), self.embedding_dim, self.embedding_dim],
                init=hk.initializers.VarianceScaling(),
            )
            acc = 0
            self.edge_split_idxs = []
            for ir in self.edge_feat_irreps[:-1]:
                acc += 2 * ir.l + 1
                self.edge_split_idxs.append(acc)

    def get_update_edges_fn(self):
        return None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            A = ops.segment_sum(
                data=edges.features,
                segment_ids=edges.receivers,
                num_segments=self.n_nodes,
            )
            if self.mix_atomic_basis:
                As = jnp.split(A, self.edge_split_idxs, axis=-1)
                A = jnp.concatenate(
                    [
                        jnp.einsum("kj,bji->bki", weight, A)
                        for weight, A in zip(self.atomic_basis_weights, As)
                    ],
                    axis=-1,
                )
            return A

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, A):
            B = self.symmetrize(A, nodes["node_type"])
            return B

        return update_nodes


class ACE(GraphNeuralNetwork):
    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        cutoff: float,
        max_body_order: int,
        embedding_irreps: Sequence[e3nn.Irrep],
        edge_feat_irreps: Sequence[e3nn.Irrep],
        node_types: Sequence[int],
        *,
        edge_feat_factory=None,
        edge_feat_kwargs=None,
        layer_kwargs=None,
    ):
        layer_kwargs = layer_kwargs or {}
        layer_kwargs.setdefault("max_body_order", max_body_order)
        layer_kwargs.setdefault("embedding_irreps", embedding_irreps)
        share = {
            "edge_feat_irreps": edge_feat_irreps,
            "n_node_type": len(node_types),
        }
        super().__init__(
            n_nodes,
            embedding_dim,
            cutoff,
            1,
            layer_kwargs,
            share_with_layers=share,
        )
        if edge_feat_factory is None:
            edge_feat_factory = EdgeFeatures
        self.edge_features = edge_feat_factory(
            embedding_dim, cutoff, edge_feat_irreps, **(edge_feat_kwargs or {})
        )
        self.node_types = node_types

    @classmethod
    @property
    def layer_factory(cls):
        r"""Return the class of the interaction layer to be used."""
        return ACELayer

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return zeros

    def node_factory(self, node_attrs):
        r"""Return the onehot encoded node types"""
        return {"node_type": to_onehot(node_attrs, self.node_types)}

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r_ij = pos_receiver[receiver_idx] - pos_sender[sender_idx]
        features = self.edge_features(r_ij)

        return features
