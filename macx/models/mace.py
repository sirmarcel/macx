import e3nn_jax as e3nn
import jax.numpy as jnp
from jax import ops

from ..gnn import EdgeFeature, GraphNeuralNetwork
from .ace import ACELayer
from .symmetric_contraction import SymmetricContraction


class MACELayer(ACELayer):
    def __init__(
        self,
        *ace_args,
        mix_prev_embed: bool = True,
        **ace_kwargs,
    ):
        super().__init__(*ace_args, **ace_kwargs)
        self.mix_prev_embed = mix_prev_embed
        if mix_prev_embed:
            self.prev_embed_weights = hk.get_parameter(
                "prev_embed_weights",
                [len(self.embedding_irreps), self.embedding_dim, self.embedding_dim],
                init=hk.initializers.VarianceScaling(),
            )
            acc = 0
            self.embed_split_idxs = []
            for ir in self.embedding_irreps[:-1]:
                acc += 2 * ir.l + 1
                self.embed_split_idxs.append(acc)

    def get_update_edges_fn(self):
        return None

    def get_aggregate_edges_for_nodes_fn(self):
        ace_aggregate = super().get_aggregate_edges_for_nodes_fn()

        def aggregate_edges_for_nodes(nodes, edges):
            prev_embed = nodes["embedding"]
            if self.mix_prev_embed:
                prev_embeds = jnp.split(prev_embed, self.embed_split_idxs, axis=-1)
                prev_embed = jnp.concatenate(
                    [
                        jnp.einsum("kj,bji->bki", weight, pe)
                        for weight, pe in zip(self.prev_embed_weights, prev_embeds)
                    ],
                    axis=-1,
                )
            e3nn.tensor_product(
                e3nn.IrrepsArray(self.edge_feat_irreps, A_unsymm),
                e3nn.IrrepsArray(
                    e3nn.Irrep("0y") if self.first_layer else self.embedding_irreps,
                    prev_embed[edges.senders],
                ),
                filter_ir_out=self.embedding_irreps,
            )
            return ace_aggregate(nodes, updated_edges)

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        ace_update_nodes = super().get_update_nodes_fn()

        def update_nodes(nodes, A):
            B = self.symmetrize(A, nodes["types"])
            return jnp.einsum("kj,bji->bki", self.mix_Bs, B)
            return ace_update_nodes(nodes, A)

        return update_nodes


class MACE(GraphNeuralNetwork):
    def __init__(
        self,
        n_nodes: int,
        embedding_dim: int,
        cutoff: float,
        n_interactions: int,
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
            "n_node_type": len(elements),
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
        self.initial_embedding = (
            hk.Embed(self.n_node_type, self.embedding_dim)
            if initial_embedding
            else None
        )

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
        if self.initial_embedding:
            init_emb = self.initial_embedding(jnp.arange(len(node_attrs)))
        else:
            init_emb = jnp.ones((len(node_attrs), self.embedding_dim))
        return {"embedding": init_emb, "type": node_attrs}

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r_ij = pos_receiver[receiver_idx] - pos_sender[sender_idx]
        features = self.edge_features(r_ij)

        return features
