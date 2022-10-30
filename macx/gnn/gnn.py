from functools import partial

import haiku as hk
import jax.numpy as jnp

from .graph import Graph, GraphEdgeBuilder, GraphUpdate


class MessagePassingLayer(hk.Module):
    r"""
    Base class for all message passing layers.

    Args:
        ilayer (int): the index of the current layer in the list of all layers
        shared (dict): attribute names and values which are shared between the
            layers and the :class:`GraphNeuralNetwork` instance.
    """

    def __init__(self, ilayer, shared):
        super().__init__()
        for k, v in shared.items():
            setattr(self, k, v)
        self.first_layer = ilayer == 0
        self.last_layer = ilayer == self.n_interactions - 1
        self.update_graph = GraphUpdate(
            update_nodes_fn=self.get_update_nodes_fn(),
            update_edges_fn=self.get_update_edges_fn(),
            aggregate_edges_for_nodes_fn=self.get_aggregate_edges_for_nodes_fn(),
        )

    def __call__(self, graph):
        r"""
        Execute the message passing layer.

        Args:
            graph (:class:`Graph`)

        Returns:
            :class:`Graph`: updated graph
        """
        return self.update_graph(graph)

    def get_update_edges_fn(self):
        r"""
        Create a function that updates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the updated edges as a :class:`GraphEdges` instance.
        """
        raise NotImplementedError

    def get_update_nodes_fn(self):
        r"""
        Create a function that updates the graph nodes.

        Returns:
            :data:`Callable[GraphNodes,*]`: a function
            that outputs the updated nodes as a :class:`GraphNodes` instance.
            The second argument will be the aggregated graph edges.
        """
        raise NotImplementedError

    def get_aggregate_edges_for_nodes_fn(self):
        r"""
        Create a function that aggregates the graph edges.

        Returns:
            :data:`Callable[GraphNodes,GraphEdges]`: a function
            that outputs the aggregated edges.
        """
        raise NotImplementedError


class GraphNeuralNetwork(hk.Module):
    r"""
    Base class for all graph neural networks on molecules.

    Args:
        n_nodes (int): the number of nodes in the graph
        embedding_dim (int): the size of the electron embeddings to be returned.
        cutoff (float): cutoff distance above which graph edges are discarded.
        n_interactions (int): the number of interaction layers in the GNN.
        layer_kwargs (dict): optional, kwargs to be passed to the layers.
        share_with_layers (dict): optional, attribute names and values to share
            with the interaction layers.
    """

    def __init__(
        self,
        n_nodes,
        embedding_dim,
        cutoff,
        n_interactions,
        layer_kwargs=None,
        share_with_layers=None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.cutoff = cutoff
        share_with_layers = share_with_layers or {}
        share_with_layers.setdefault("embedding_dim", embedding_dim)
        for k, v in share_with_layers.items():
            setattr(self, k, v)
        share_with_layers.setdefault("n_interactions", n_interactions)
        self.layers = [
            self.layer_factory(
                i,
                share_with_layers,
                **(layer_kwargs or {}),
            )
            for i in range(n_interactions)
        ]

    def init_state(self, shape, dtype):
        r"""Initialize the haiku state that communicates the sizes of edge lists."""
        raise NotImplementedError

    def initial_embeddings(self):
        r"""Return the initial embeddings as a :class:`GraphNodes` instance."""
        raise NotImplementedError

    def edge_feature_callback(self, pos_sender, pos_receiver, sender_idx, receiver_idx):
        r"""
        Define the :func:`feature_callback` for the different types of edges.

        Args:
            pos_sender (float, (:math:`N_\text{nodes}`, 3)): coordinates of the
                sender nodes.
            pos_receiver (float, (:math:`M_\text{nodes}`, 3]): coordinates of the
                receiver nodes.
            sender_idx (int, (:data:`occupancy_limit`)): indeces of the sender nodes.
            receiver_idx (int, (:data:`occupancy_limit`)): indeces of the receiver
                nodes.

        Returns:
            the features for the given edges
        """
        raise NotImplementedError

    def edge_factory(self, r, occupancies):
        r"""Return a function that builds all the edges used in the GNN."""
        mask_val = r.shape[0] + 1
        edge_factory = GraphEdgeBuilder(
            self.cutoff,
            True,
            (0, 0),
            (mask_val, mask_val),
            self.edge_feature_callback,
        )
        return edge_factory(r, r, occupancies)

    @classmethod
    @property
    def layer_factory(cls):
        r"""Return the class of the interaction layer to be used."""
        return MessagePassingLayer

    def __call__(self, r):
        r"""
        Execute the graph neural network.

        Args:
            r (float, (:math:`N_\text{elec}`, 3)): electron coordinates.

        Returns:
            float, (:math:`N_\text{elec}`, :data:`embedding_dim`):
            the final embeddings of the electrons.
        """
        if r.shape[0] != self.n_nodes:
            raise ValueError
        occupancies = hk.get_state(
            "occupancies",
            shape=1,
            dtype=jnp.int32,
            init=self.init_state,
        )
        graph_edges, occupancies = self.edge_factory(r, occupancies)
        hk.set_state("occupancies", occupancies)
        graph_nodes = self.initial_embeddings()
        graph = Graph(
            graph_nodes,
            graph_edges,
        )

        for layer in self.layers:
            graph = layer(graph)

        return graph.nodes
