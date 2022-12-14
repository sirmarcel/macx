from collections import namedtuple

import jax.numpy as jnp

GraphEdges = namedtuple("GraphEdges", "senders receivers features")
Graph = namedtuple("Graph", "nodes edges")


def all_graph_edges(pos_sender, pos_receiver):
    r"""Create all graph edges.

    Args:
        pos_sender (float, (:math:`N_\text{nodes}`, 3)): coordinates of graph
            nodes that send edges.
        pos_receiver (float, (:math:`M_\text{nodes}`, 3)): coordinates of graph
            nodes that receive edges.

    Returns:
        int, (:math:`N`, :math:`M`): matrix of node indeces, indicating
        the receiver node of the :math:`N \cdot M` possible edges.
    """
    idx = jnp.arange(pos_receiver.shape[0])
    return jnp.broadcast_to(idx[None, :], (pos_sender.shape[0], pos_receiver.shape[0]))


def mask_self_edges(idx):
    r"""Mask the edges where sender and receiver nodes have the same index.

    Args:
        idx (int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)):
            index of receiving nodes, assumed to be square since sets of sender
            and receiver nodes should be identical.

    Returns:
        int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`): matrix of
        receiving node indeces, the appropriate entries masked with
        :math:`N_\text{nodes}`.
    """
    self_mask = idx == jnp.reshape(
        jnp.arange(idx.shape[0], dtype=jnp.int32), (idx.shape[0], 1)
    )
    return jnp.where(self_mask, idx.shape[0], idx)


def mask_custom_edges(idx, mask):
    r"""
    Mask the edges according to :data:`mask`.

    Args:
        idx (int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)): matrix of
            receiving node indeces.
        mask (bool, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)): mask definition,
            entries that contain :data:`False` will be masked out.
    """
    return jnp.where(mask, idx.shape[1], idx)


def prune_graph_edges(
    pos_sender,
    pos_receiver,
    cutoff,
    idx,
    occupancy_limit,
    offsets,
    mask_vals,
    feature_callback,
):
    r"""Discards graph edges which have a distance larger than :data:`cutoff`.

    Args:
        pos_sender (float, (:math:`N_{nodes}`, 3)): coordinates of graph nodes
            that send edges.
        pos_receiver (float, (:math:`M_{nodes}`, 3)): coordinates of graph nodes
            that receive edges.
        cutoff (float): cutoff distance above which edges are discarded.
        idx (int, (:math:`N_\text{nodes}`, :math:`N_\text{nodes}`)): matrix of
            receiving node indices as created by :func:`all_graph_edges`
            (or :func:`mask_self_edges`).
        occupancy_limit (int): the number of edges that can be considered
            without overflow. The arrays describing the edges will have
            a last dimension of size :data:`occupancy_limit`.
        offsets ((int, int)): node index offset to be added to the returned
            sender and receiver node indeces respectively.
        mask_vals ((int, int)): if :data:`occupancy_limit` is larger than the number
            of valid edges, the remaining node indices will be filled with these
            values for the sender and receiver nodes respectively
            (i.e. the value to pad the node index arrays with).
        feature_callback (Callable): a function that takes the sender positions,
            receiver positions, sender node indeces and receiver node indeces and
            returns some data (features) computed for the edges.

    Returns:
        GraphEdges: object containing the indeces of the edge
        sending and edge receiving nodes, along with the features associated
        with the edges.
    """

    def apply_callback(pos_sender, pos2, sender_idx, receiver_idx):
        r"""Apply the feature_callback function, or return no features."""
        return (
            feature_callback(pos_sender, pos2, sender_idx, receiver_idx)
            if feature_callback
            else {}
        )

    if pos_sender.shape[0] == 0 or pos_receiver.shape[0] == 0:
        ones = jnp.ones(occupancy_limit, idx.dtype)
        sender_idx = offsets[0] * ones
        receiver_idx = offsets[1] * ones
        return (
            GraphEdges(
                sender_idx,
                receiver_idx,
                apply_callback(pos_sender, pos_receiver, sender_idx, receiver_idx),
            ),
            jnp.array(0),
        )

    def dist(sender, receiver):
        r"""Compute pairwise distances between inputs."""
        return jnp.sqrt(((receiver - sender) ** 2).sum(axis=-1))

    N_sender, N_receiver = pos_sender.shape[0], pos_receiver.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N_sender)[:, None], idx.shape)
    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))

    distances = dist(pos_sender[sender_idx], pos_receiver[receiver_idx])
    mask = (distances < cutoff) & (receiver_idx < N_receiver)
    cumsum = jnp.cumsum(mask)
    occupancy = cumsum[-1]

    # edge buffer is one larger than occupancy_limit:
    # masked edges assigned to last position and discarded
    out_sender_idx, out_receiver_idx = (
        (mask_val - offset) * jnp.ones(occupancy_limit + 1, jnp.int32)
        for mask_val, offset in zip(mask_vals, offsets)
    )
    index = jnp.where(mask, cumsum - 1, occupancy_limit)

    sender_idx = out_sender_idx.at[index].set(sender_idx)[:occupancy_limit]
    receiver_idx = out_receiver_idx.at[index].set(receiver_idx)[:occupancy_limit]

    features = apply_callback(pos_sender, pos_receiver, sender_idx, receiver_idx)

    return (
        GraphEdges(sender_idx + offsets[0], receiver_idx + offsets[1], features),
        occupancy,
    )


def difference_callback(pos_sender, pos_receiver, sender_idx, receiver_idx):
    r"""Feature_callback computing the Euclidian difference vector for each edge."""
    if len(pos_sender) == 0 or len(pos_receiver) == 0:
        return jnp.zeros((len(sender_idx), 3))
    diffs = pos_receiver[receiver_idx] - pos_sender[sender_idx]
    return diffs


def GraphEdgeBuilder(
    cutoff,
    mask_self,
    offsets,
    mask_vals,
    feature_callback,
):
    r"""
    Create a function that builds graph edges.

    Args:
        cutoff (float): the cutoff distance above which edges are discarded.
        mask_self (bool): whether to mask edges between nodes of the same index.
        offsets ((int, int)): node index offset to be added to the returned
            sender and receiver node indeces respectively.
        mask_vals ((int, int)): if ``occupancy_limit`` is larger than the number
            of valid edges, the remaining node indices will be filled with these
            values for the sender and receiver nodes respectively
            (i.e. the value to pad the node index arrays with).
        feature_callback (Callable): a function that takes the sender positions,
            receiver positions, sender node indeces and receiver node indeces and
            returns some data (features) computed for the edges.
    """

    def build(pos_sender, pos_receiver, occupancies, custom_mask=None):
        r"""
        Build graph edges.

        Args:
            pos_sender (float, (:math:`N_{nodes}`, 3)): coordinates of graph nodes
                that send edges.
            pos_receiver (float, (:math:`M_{nodes}`, 3)): coordinates of graph nodes
                that receive edges.
            occupancies (int, (:data:`occupancy_limit`)): array to store
                occupancies in.

        Returns:
            tuple: a tuple containing the graph edges, the input occupancies
            updated with the current occupancy, and the number of stored
            occupancies.
        """
        assert pos_sender.shape[-1] == 3 and pos_receiver.shape[-1] == 3
        assert len(pos_sender.shape) == 2
        assert not mask_self or pos_sender.shape[0] == pos_receiver.shape[0]

        occupancy_limit = occupancies.shape[0]

        edges_idx = all_graph_edges(pos_sender, pos_receiver)

        if mask_self:
            edges_idx = mask_self_edges(edges_idx)
        if custom_mask is not None:
            edges_idx = mask_custom_edges(edges_idx, custom_mask)
        edges, occupancy = prune_graph_edges(
            pos_sender,
            pos_receiver,
            cutoff,
            edges_idx,
            occupancy_limit,
            offsets,
            mask_vals,
            feature_callback,
        )

        return (
            edges,
            occupancies.at[1:].set(occupancies[:-1]).at[0].set(occupancy),
        )

    return build


def GraphUpdate(
    aggregate_edges_for_nodes_fn,
    update_nodes_fn=None,
    update_edges_fn=None,
):
    r"""
    Create a function that updates a graph.

    The update function is tailored to be used in GNNs.

    Args:
        aggregate_edges_for_nodes_fn (bool): whether to perform the aggregation
            of edges for nodes.
        update_nodes_fn (Callable): optional, function that updates the nodes.
        update_edges_fn (Callable): optional, function that updates the edges.
    """

    def update_graph(graph):
        nodes, edges = graph

        if update_edges_fn:
            edges = update_edges_fn(nodes, edges)

        if update_nodes_fn:
            aggregated_edges = aggregate_edges_for_nodes_fn(nodes, edges)
            nodes = update_nodes_fn(nodes, aggregated_edges)

        return Graph(nodes, edges)

    return update_graph
