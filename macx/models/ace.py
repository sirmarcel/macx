from jax import ops

from ..gnn import GraphNeuralNetwork, MessagePassingLayer
from .symmetric_contraction import SymmetricContraction


class ACELayer(MessagePassingLayer):
    def __init__(
        self,
        ilayer,
        shared,
        *,
        node_feats_irreps="0y+1y",
        target_irreps="0y",
        max_body_order,
    ):
        super().__init__(ilayer, shared)
        self.embedding_dim
        node_feats_irreps = "+".join(
            [f"{embedding_dim}x{y}" for y in node_feats_irreps.split("+")]
        )
        node_feats_irreps = e3nn.Irreps(node_feats_irreps)
        target_irreps = "+".join(
            [f"{embedding_dim}x{y}" for y in target_irreps.split("+")]
        )
        target_irreps = e3nn.Irreps(target_irreps)
        self.symmetrize = SymmetricContraction(
            node_feats_irreps, target_irreps, max_body_order
        )

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
        embedding_dim,
        cutoff,
        layer_kwargs=None,
        share_with_layers=None,
        **gnn_kwargs,
    ):
        super().__init__(embedding_dim, cutoff, n_interactions=1, **gnn_kwargs)
