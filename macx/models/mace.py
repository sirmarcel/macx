from typing import Optional, Sequence

import e3nn_jax as e3nn
import jax.numpy as jnp
from jax import ops

from ..gnn import GraphNeuralNetwork
from ..gnn.edge_features import EdgeFeatures
from ..tools.e3nn_ext import GeneralLinear, WeightedTensorProduct, convert_irreps_array
from .ace import ACELayer, to_onehot


class MACELayer(ACELayer):
    def __init__(
        self,
        *ace_args,
        prev_embed_irreps: Sequence[e3nn.Irrep],
        **ace_kwargs,
    ):
        super().__init__(*ace_args, **ace_kwargs)
        embedding_irreps = ace_kwargs["embedding_irreps"]
        self.prev_embed_mixing_layer = convert_irreps_array(prev_embed_irreps)(
            GeneralLinear(prev_embed_irreps, mix_channels=True)
        )

        self.message_mixing_layer = convert_irreps_array(embedding_irreps)(
            GeneralLinear(embedding_irreps, mix_channels=True)
        )
        if not self.first_layer:
            self.embed_mixing_layer = convert_irreps_array(embedding_irreps)(
                GeneralLinear(
                    embedding_irreps,
                    mix_channels=True,
                    new_channel_dim=self.n_node_type,
                )
            )
        self.wtp = WeightedTensorProduct(
            self.edge_feat_irreps,
            prev_embed_irreps,
            self.edge_feat_irreps,
        )

    def get_update_edges_fn(self):
        return None

    def get_aggregate_edges_for_nodes_fn(self):
        ace_aggregate = super().get_aggregate_edges_for_nodes_fn()

        def aggregate_edges_for_nodes(nodes, edges):
            embedding = self.prev_embed_mixing_layer(nodes["embedding"])
            updated_features = self.wtp(edges.features, embedding[edges.senders])
            updated_edges = edges._replace(features=updated_features)
            return ace_aggregate(nodes, updated_edges)

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        ace_update_nodes = super().get_update_nodes_fn()

        def update_nodes(nodes, A):
            messages = ace_update_nodes(nodes, A)
            # TODO: implement tensor product residual,
            # so different layers can have different irreps
            residual = (
                0
                if self.first_layer
                else jnp.einsum(
                    "ijkl,ij->ikl",
                    self.embed_mixing_layer(nodes["embedding"]),
                    nodes["node_type"],
                )
            )
            update = self.message_mixing_layer(messages)
            nodes["embedding"] = residual + update
            return nodes["embedding"] if self.last_layer else nodes

        return update_nodes


class MACE(GraphNeuralNetwork):
    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        cutoff: float,
        max_body_order: int,
        embedding_irreps: Sequence[Sequence[e3nn.Irrep]],
        edge_feat_irreps: Sequence[e3nn.Irrep],
        node_types: Sequence[int],
        *,
        edge_feat_factory=None,
        edge_feat_kwargs=None,
        layer_kwargs: Optional[Sequence[dict]] = None,
    ):
        n_interactions = len(embedding_irreps)
        layer_kwargs = layer_kwargs or [{} for _ in embedding_irreps]
        for i, emb_irrep in enumerate(embedding_irreps):
            layer_kwargs[i].setdefault("max_body_order", max_body_order)
            layer_kwargs[i].setdefault("embedding_irreps", emb_irrep)
            layer_kwargs[i].setdefault(
                "prev_embed_irreps",
                [e3nn.Irrep("0y")] if i == 0 else embedding_irreps[i - 1],
            )
        share = {
            "edge_feat_irreps": edge_feat_irreps,
            "n_node_type": len(node_types),
        }
        super().__init__(
            n_nodes,
            embedding_dim,
            cutoff,
            n_interactions,
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
        return MACELayer

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return zeros

    def node_factory(self, node_attrs):
        r"""Return initial embeddings and onehot encoded node types"""
        init_emb = jnp.ones((len(node_attrs), self.embedding_dim, 1))
        return {
            "embedding": init_emb,
            "node_type": to_onehot(node_attrs, self.node_types),
        }

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r_ij = pos_receiver[receiver_idx] - pos_sender[sender_idx]
        features = self.edge_features(r_ij)

        return features
