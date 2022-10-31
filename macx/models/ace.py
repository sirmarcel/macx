from functools import partial
from typing import Optional, Sequence

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
from jax import ops

from ..gnn import GraphNeuralNetwork, MessagePassingLayer
from ..gnn.edge_features import EdgeFeatures
from .symmetric_contraction import SymmetricContraction


class ACELayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        shared,
        *,
        max_body_order: int,
        embedding_irreps: Sequence[e3nn.Irrep],
        mix_atomic_basis: bool = True,
        num_elements: Optional[int] = None,
    ):
        super().__init__(ilayer, shared)
        self.mix_atomic_basis = mix_atomic_basis
        self.symmetrize = SymmetricContraction(
            embedding_irreps,
            self.edge_feat_irreps,
            self.embedding_dim,
            max_body_order,
            num_elements,
        )
        if mix_atomic_basis:
            self.atomic_basis_weights = hk.get_parameter(
                "atomic_basis_weights",
                [len(self.edge_feat_irreps), self.embedding_dim, self.embedding_dim],
                init=hk.initializers.VarianceScaling(),
            )
            acc = 0
            self.split_idxs = []
            for ir in self.edge_feat_irreps[:-1]:
                acc += 2 * ir.l + 1
                self.split_idxs.append(acc)

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
                As = jnp.split(A, self.split_idxs, axis=-1)
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
            B = self.symmetrize(A, nodes["node_types"])
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

    @classmethod
    @property
    def layer_factory(cls):
        r"""Return the class of the interaction layer to be used."""
        return ACELayer

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return zeros

    def node_factory(self, node_attrs):
        r"""Return the initial embeddings as a :class:`GraphNodes` instance."""
        return {
            "initial_embeddings": jnp.ones((self.n_nodes, 1)),
            "node_types": node_attrs,
        }

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r_ij = pos_receiver[receiver_idx] - pos_sender[sender_idx]
        features = self.edge_features(r_ij)

        return features
