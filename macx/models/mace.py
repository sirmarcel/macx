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


class MACE(ACE):
    def __init__(
        self,
        *ace_args,
        initial_embedding: bool = True,
        **ace_kwargs,
    ):
        super().__init__(*ace_args, **ace_kwargs)
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
