import e3nn_jax as e3nn
import jax.numpy as jnp
from jax import ops

from ..gnn import EdgeFeature, GraphNeuralNetwork, MessagePassingLayer
from .symmetric_contraction import SymmetricContraction


class ACELayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        shared,
        *,
        node_feats_irreps="0y+1y",
        target_irreps="0y",
        max_body_order=2,
    ):
        super().__init__(ilayer, shared)
        self.embedding_dim
        node_feats_irreps = "+".join(
            [f"{self.embedding_dim}x{y}" for y in node_feats_irreps.split("+")]
        )
        node_feats_irreps = e3nn.Irreps(node_feats_irreps)
        target_irreps = "+".join(
            [f"{self.embedding_dim}x{y}" for y in target_irreps.split("+")]
        )
        target_irreps = e3nn.Irreps(target_irreps)
        self.symmetrize = SymmetricContraction(
            node_feats_irreps, target_irreps, max_body_order
        )

    def get_update_edges_fn(self):
        return None

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            n_nodes = nodes.shape[-2]
            A = ops.segment_sum(
                data=edges.features, segment_ids=edges.receivers, num_segments=n_nodes
            )
            return A

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, A):
            B = self.symmetrize(A)
            return B

        return update_nodes


class ACE(GraphNeuralNetwork):
    def __init__(
        self,
        n_nodes,
        embedding_dim,
        cutoff,
        layer_kwargs=None,
        share_with_layers=None,
        *,
        radial_fn: str = "bessel",
        l_max: int = 0,
        **gnn_kwargs,
    ):
        super().__init__(n_nodes, embedding_dim, cutoff, n_interactions=1, **gnn_kwargs)
        self.edge_feature_factory = EdgeFeature(radial_fn, embedding_dim, cutoff, l_max)

    @classmethod
    @property
    def layer_factory(cls):
        r"""Return the class of the interaction layer to be used."""
        return ACELayer

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return zeros

    def initial_embeddings(self):
        r"""Return the initial embeddings as a :class:`GraphNodes` instance."""
        #  return e3nn.IrrepsArray("1x0e", jnp.ones((self.n_nodes, 1)))
        return jnp.ones((self.n_nodes, 1))

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r_ij = pos_receiver[receiver_idx] - pos_sender[sender_idx]
        features = self.edge_feature_factory(
            jnp.ones(1), sender_idx, receiver_idx, r_ij
        )

        return features
