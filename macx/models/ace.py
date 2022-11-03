from typing import Sequence

import e3nn_jax as e3nn
import jax.numpy as jnp
from jax import ops

from ..gnn import GraphNeuralNetwork, MessagePassingLayer
from ..gnn.edge_features import EdgeFeatures
from ..tools.e3nn_ext import EquivariantLinear, convert_irreps_array
from ..tools.symmetric_contraction import SymmetricContraction


def to_onehot(features, node_types):
    r"""
    Create onehot encoded vectors from :data:`features`.

    Args:
        features (int, jnp.ndarray): type of the nodes
        node_types (Sequence[int]): list of possible node types.
    """
    ones = []
    for i, e in enumerate(node_types):
        ones.append(jnp.where(features == e, jnp.ones(1), jnp.zeros(1))[..., None])
    return jnp.concatenate(ones, axis=-1)


class ACELayer(MessagePassingLayer):
    r"""
    Compute the ACE interaction.

    The ACE interaction is composed of constructing the edge features, summing them
    on each receiver node, and creating the symmetrized, many-body node embeddings.

    Args:
        max_body_order (int): the maximum body order up to which node embeddings are
            constructed.
        embedding_irreps (Sequence[e3nn.Irrep]): the irreps of the node embeddings.
        mix_atomic_basis (bool): default :data:`True`, whether to apply a linear layer
            on the initial node embeddings, before symmetrizing them.
    """

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
            self.atomic_basis_layer = convert_irreps_array(embedding_irreps)(
                EquivariantLinear(embedding_irreps, mix_channels=True)
            )

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
                A = self.atomic_basis_layer(A)

            return A

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, A):
            B = self.symmetrize(A, nodes["node_type"])
            return B

        return update_nodes


class ACE(GraphNeuralNetwork):
    r"""
    The ACE model.

    Args:
        n_nodes (int): the maximum number of nodes in the graph.
        embedding_dim (int): the embedding dimension, should be equal to the number of
            radial basis functions.
        cutoff (float): distance cutoff, beyond which interactions are not considered.
        max_body_order (int): the maximum body order up to which node embeddings are
            constructed.
        embedding_irreps (Sequence[e3nn.Irrep]): the irreps of the node embeddings.
        edge_feat_irreps (Sequence[e3nn.Irrep]): the irreps of the edge features.
        node_types (Sequence[int]): the list of possible node types.
        edge_feat_factory (Optional[Callable]): the edge feature constructing class,
            defaults to :class:`~gnn.edge_features.EdgeFeatures`.
        edge_feat_kwargs (Optional[dict]): extra arguments to be passed to
            :data:`edge_feat_factory`.
        layer_kwargs (dict): optional, kwargs to be passed to the layers.
    """

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
