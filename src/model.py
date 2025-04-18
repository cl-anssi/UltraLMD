import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from scipy.special import softmax

from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra.tasks import build_relation_graph

from graph import Graph, aggregate_graphs




MARGIN = 1e-10
MAX_SCORE = -np.log(MARGIN)


def build_graph_data(edges, graph, nodes=None):
    '''
    Preprocesses the context graph and the new edges for scoring by Ultra.

    Arguments
    ---------
    edges : Graph
        New edges to be scored.
    graph : Graph
        Context graph used for scoring.
    nodes : dict, default=None
        Name -> type map for the nodes.
        If None, no type nodes and has_type edges are added to the context
        graph.

    Returns
    -------
    triples : list
        Encoded new edges.
    data : Data
        Context graph.
    pred_nodes_id : list
        Node ids (in the context graph) of the candidate nodes for link
        prediction.

    '''

    all_nodes = graph.get_nodes().union(edges.get_nodes())
    pred_nodes = edges.get_pred_nodes().union(graph.get_pred_nodes())
    ctx_edges = graph.get_edges(
        return_aux=True, return_dataframe=True
    )
    if nodes is not None:
        # Add nodes representing node types and edges to these nodes
        type_edges = [
            (n, '_has_type', nodes[n])
            for n in all_nodes
        ]
        src, rel, dst = [
            list(tup)
            for tup in zip(*type_edges)
        ]
        all_nodes = all_nodes.union(dst)
    else:
        src, rel, dst = [], [], []
    # Build name -> id map for the nodes
    node_idx = {}
    all_nodes = sorted(list(all_nodes))
    for n in all_nodes:
        node_idx[n] = len(node_idx)
    # Split the edge list of the context graph into source, relation and
    # destination lists
    src, rel, dst = [
        l + ctx_edges[k].to_list()
        for l, k in zip(
            [src, rel, dst],
            ['src', 'rel', 'dst']
        )
    ]
    # Build name -> id map for the relations
    rel_types = sorted(list(set(rel)))
    rel_idx = {}
    for t in rel_types:
        rel_idx[t] = len(rel_idx)
    # Build the tensor representing the edges of the context graph (encoded
    # sources and destinations)
    edge_index = torch.tensor([
            [node_idx[s] for s in src],
            [node_idx[d] for d in dst]
        ],
        dtype=torch.long
    )
    # Add reciprocal edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    # Build the tensor of encoded edge types
    edge_type = torch.tensor(
        [rel_idx[r] for r in rel],
        dtype=torch.long
    )
    # Add reciprocal edges
    edge_type = torch.cat([edge_type, edge_type + len(rel_idx)], dim=-1)
    # Build the torch_geometric Data object representing the context graph
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_relations=2 * len(rel_idx),
        num_nodes=len(node_idx)
    )
    # Encode the edges to score and the list of candidate nodes for link
    # prediction
    triples = [
        (node_idx[h], node_idx[t], rel_idx[r])
        for h, r, t in edges.get_edges()
    ]
    pred_nodes_id = [node_idx[node] for node in pred_nodes]
    return triples, build_relation_graph(data), pred_nodes_id

@torch.no_grad()
def eval_model(
        model,
        data,
        edges,
        batch_size=128,
        flip=False,
        pred_nodes=None,
        device='cpu'
    ):
    '''
    Computes the negative conditional log-probabilities of the input edges
    given the model and the context graph.

    Arguments
    ---------
    model : Ultra
        Instance of Ultra used for scoring.
    data : Data
        Context graph.
    edges : list
        Edges to score.
        Each edge is a (source, destination, relation) tuple.
    batch_size : int, default=128
        Size of the input edge batches.
    flip : bool, default=False
        If True, the conditional probability of the source given the
        destination is computed.
        Otherwise, the conditional probability of the destination given the
        source is computed.
    pred_nodes : list, default=None
        Nodes considered for link prediction.
        The predicted probability of each node is normalized using only the
        logits of the nodes in pred_nodes.
        If None, all nodes are included.
    device : str, default='cpu'
        Device used by Torch for computation.

    Returns
    -------
    scores : np.array of shape (len(edges),)
        Negative conditional log-probabilities of the edges given the context
        graph.

    '''

    model.eval()
    scores = {}
    if flip:
        _edges = [
            (dst, src, rel + data.num_relations // 2)
            for src, dst, rel in edges
        ]
    else:
        _edges = edges
    # Group edges by (source, relation) for batching
    keys = sorted(list(set([(src, rel) for src, _, rel in _edges])))
    for i in range(0, len(keys), batch_size):
        # Build the input tensor : for each (source, relation) pair,
        # build all possible (source, relation, destination) triples,
        # then stack the results for the batch
        h_index, r_index = torch.tensor([
            [src, rel]
            for src, rel in keys[i:i + batch_size]
        ], dtype=torch.long, device=device).t()
        r_index = r_index.unsqueeze(-1).expand(-1, data.num_nodes)
        all_index = torch.arange(data.num_nodes, device=device)
        h_index, t_index = torch.meshgrid(h_index, all_index, indexing='ij')
        batch = torch.stack([h_index, t_index, r_index], dim=-1)
        # Compute logits for the current batch
        preds = model(data, batch).cpu().numpy()
        for j, key in enumerate(keys[i:i + batch_size]):
            scores[key] = preds[j, :]
        if (i // batch_size) % 10 == 0:
            print('Batch #', i // batch_size)
    # Compute the anomaly scores for the actual destinations
    if pred_nodes is None:
        pred_nodes = range(data.num_nodes)
    nodes = list(pred_nodes)
    idx = [0] * data.num_nodes
    for i, n in enumerate(nodes):
        idx[n] = i
    res = np.array([
        softmax(scores[(src, rel)][nodes])[idx[dst]]
        for src, dst, rel in _edges
    ])
    return -np.log(res + MARGIN)

def node_rel_similarity(graph, other_graphs):
    '''
    Computes the similarity between a given graph and each other graph in a
    list.

    Arguments
    ---------
    graph : Graph
        Reference graph.
    other_graphs : list
        List of graphs to compare with the reference graph.

    Returns
    -------
    node_similarity : np.array of shape (len(other_graphs),)
        Jaccard index of the node set of the reference graph and the node
        set of each other graph.
    rel_similarity : np.array of shape (len(other_graphs),)
        Cosine similarity between the standardized vector of relation counts
        of the reference graph and that of each other graph.

    '''

    node_vectors = [
        g.get_node_vector()
        for g in other_graphs + [graph]
    ]
    max_len = max(len(v) for v in node_vectors)
    node_vectors = np.stack([
        np.concatenate([v, [0] * (max_len - len(v))])
        if len(v) < max_len
        else v
        for v in node_vectors
    ])
    # Compute the Jaccard index: intersection is the dot product of the
    # indicator vectors, size of each set is the sum of its indicator vector
    node_similarity = node_vectors[:-1] @ node_vectors[-1]
    node_similarity /= (
        node_vectors[:-1].sum(1) + node_vectors[-1].sum() - node_similarity
    )
    rel_vectors = [
        g.get_rel_vector()
        for g in other_graphs + [graph]
    ]
    # Pad the relation count vectors to the same length
    max_len = max(len(v) for v in rel_vectors)
    rel_vectors = np.stack([
        np.concatenate([v, [0] * (max_len - len(v))])
        if len(v) < max_len
        else v
        for v in rel_vectors
    ])
    # Standardize the relation count vectors
    rel_vectors = StandardScaler().fit_transform(rel_vectors)
    # Compute the cosine similarity
    rel_similarity = rel_vectors[:-1] @ rel_vectors[-1]
    norms = np.linalg.norm(rel_vectors, axis=1)
    rel_similarity /= norms[:-1] * norms[-1]
    return node_similarity, rel_similarity


class UltraLMD:
    '''
    Class implementing the UltraLMD++ detector.

    Parameters
    ----------
    base_model : Ultra
        GFM used for anomaly scoring.
    num_context_graphs : int, default=10
        Number of past graphs to include in the short-term context.
        When set to a negative value, only the long-term context graph is used
        for scoring.
    batch_size : int, default=128
        Size of the batches passed to the GFM.
    refine_scores : bool, default=True
        Whether to use the graph-based score refinement algorithm.
    device : str, default='cpu'
        Device used by Torch for computation.

    Attributes
    ----------
    graphs : list
        List of past graphs available for building short-term context graphs.
    global_graph : Graph
        Union of all past graphs (i.e., long-term context graph).
    node_idx : dict
        Name -> id map for the nodes.
    rel_idx : dict
        Name -> id map for the relations.
    times : list
        Computation times for the retrieval, scoring, and refinement component.
        Each element of the list is a tuple (retrieval_time, scoring_time,
        refinement_time) for one time window that has been scored.

    '''

    def __init__(
        self,
        base_model,
        num_context_graphs=10,
        batch_size=128,
        refine_scores=True,
        device='cpu'
    ):
        self.base_model = base_model.to(device)
        self.num_context_graphs = num_context_graphs
        self.batch_size = batch_size
        self.refine_scores = refine_scores
        self.device = device

        self.graphs = []
        self.global_graph = None
        self.node_idx = {}
        self.rel_idx = {}

        self.times = []

    def get_node_idx(self):
        '''
        Returns the name -> id map for the nodes.

        Returns
        -------
        node_idx : dict
            Name -> id map for the nodes.

        '''

        return self.node_idx

    def get_rel_idx(self):
        '''
        Returns the name -> id map for the relations.

        Returns
        -------
        rel_idx : dict
            Name -> id map for the relations.

        '''

        return self.rel_idx

    def score(self, graph, nodes=None):
        '''
        Computes anomaly scores for the given edges.

        Arguments
        ---------
        graph : Graph
            Edges to score.
        nodes : dict, default=None
            Name -> type map for the nodes.

        Returns
        -------
        scores : np.array, shape=(graph.n_edges,)
            Anomaly scores.

        '''

        t0 = time.time()
        scores = self._score_edges(
            graph, self.global_graph, nodes=nodes
        )
        t1 = time.time()
        if self.num_context_graphs > 0:
            context_graphs = self._get_context_graphs(graph)
            cg = aggregate_graphs(context_graphs)
            t2 = time.time()
            scores += self._score_edges(graph, cg, nodes=nodes)
            scores /= 2
            t3 = time.time()
        else:
            t2 = t1
            t3 = t1
        if self.refine_scores:
            scores = graph.refine_scores(scores)
            t4 = time.time()
        else:
            t4 = t3
        self.times.append((t2 - t1, t1 - t0 + t3 - t2, t4 - t3))
        return scores

    def update(self, graph):
        '''
        Adds the input graph to the list of past graphs and updates the
        union of all past graphs.

        Arguments
        ---------
        graph : Graph
            New past graph.

        '''

        self.graphs.append(graph)
        if self.global_graph is None:
            self.global_graph = graph.copy()
        else:
            self.global_graph.update(graph)
        self.node_idx.update(graph.get_node_idx())
        self.rel_idx.update(graph.get_rel_idx())

    def _compute_scores(self, triples, data, pred_nodes):
        '''
        Returns the anomaly scores for the preprocessed edges and context graph
        passed as input.

        Arguments
        ---------
        triples : list
            List of (source, relation, destination) tuples to score, where each
            element is represented by its id in the encoded context graph.
        data : Data
            Context graph encoded as a torch_geometric Data object.
        pred_nodes : list
            Node ids (in the context graph) of the candidate nodes for link
            prediction.

        Returns
        -------
        scores : np.array, shape=(len(triples),)
            Anomaly scores.

        '''

        scores_fw = eval_model(
            self.base_model,
            data.to(self.device),
            triples,
            batch_size=self.batch_size,
            pred_nodes=pred_nodes,
            device=self.device
        )
        scores_bw = eval_model(
            self.base_model,
            data.to(self.device),
            triples,
            batch_size=self.batch_size,
            flip=True,
            pred_nodes=pred_nodes,
            device=self.device
        )
        return np.fmin(scores_fw, scores_bw)

    def _get_context_graphs(self, graph):
        '''
        Returns the past graphs most similar to the input graph.
        Past graphs are retrieved from self.graphs, and the number of
        retrieved graphs is min(self.num_context_graphs, len(self.graphs)).
        If self.num_context_graphs < 0, all past graphs are returned.

        Arguments
        ---------
        graph : Graph
            Input graph.

        Returns
        -------
        context_graphs : list
            List of past graphs (Graph objects) most similar to the input
            graph.

        '''

        if (
            self.num_context_graphs < 0
            or len(self.graphs) <= self.num_context_graphs
        ):
            return self.graphs
        similarity_scores = self._get_similarity_scores(graph)
        idx = np.argsort(similarity_scores)
        context_graphs = [
            self.graphs[i] for i in idx[-self.num_context_graphs:]
        ]
        return context_graphs

    def _get_similarity_scores(self, graph):
        '''
        Returns the similarities of the input graph with all past graphs
        stored in self.graphs.

        Arguments
        ---------
        graph : Graph
            Input graph.

        Returns
        -------
        similarity_scores : np.array, shape=(len(self.graphs),)
            Similarities of the input graph with all past graphs.

        '''

        node_sim, rel_sim = node_rel_similarity(graph, self.graphs)
        return node_sim + rel_sim

    def _score_edges(self, graph, context_graph, nodes=None):
        '''
        Computes the anomaly scores for the input edges given the input
        context graph.

        Arguments
        ---------
        graph : Graph
            Edges to score.
        context_graph : Graph
            Context graph used for scoring.
        nodes : dict, default=None
            Name -> type map for the nodes.
            If None, no type nodes and has_type edges are added to the context
            graph.

        Returns
        -------
        scores : np.array, shape=(graph.n_edges,)
            Anomaly scores.
            The index of the array is aligned with the dataframe containing
            the edges, i.e., scores[i] = score(graph.edges.iloc[i, :]).

        '''

        scores = self._split_and_score(
            graph, context_graph, nodes=nodes
        )
        # Compute scores for unknown relations
        unknown_rel_idx = np.where(scores == -1)[0]
        if len(unknown_rel_idx) > 0:
            edges = graph.get_edges(
                return_aux=True
            )
            edges = [edges[i] for i in unknown_rel_idx]
            known_rels = context_graph.get_rels()
            sub_graph = Graph(
                pd.DataFrame(
                    [
                        (s, r, d, a)
                        for s, _, d, a in edges
                        for r in known_rels
                    ],
                    columns=['src', 'rel', 'dst', 'auxiliary']
                )
            )
            alt_scores = self._split_and_score(
                sub_graph, context_graph, nodes=nodes
            )
            scores[unknown_rel_idx] = alt_scores.reshape(
                len(edges), len(known_rels)
            ).mean(1)
        # Handle other cases
        scores[scores < 0] = MAX_SCORE
        return scores

    def _split_and_score(self, graph, context_graph, nodes=None):
        '''
        Computes the anomaly scores for the input edges given the input
        context graph.
        Edges whose type does not exist in the context graph are not
        scored.

        Arguments
        ---------
        graph : Graph
            Edges to score.
        context_graph : Graph
            Context graph used for scoring.
        nodes : dict, default=None
            Name -> type map for the nodes.
            If None, no type nodes and has_type edges are added to the context
            graph.

        Returns
        -------
        scores : np.array, shape=(graph.n_edges,)
            Anomaly scores.
            The index of the array is aligned with the dataframe containing
            the edges, i.e., scores[i] = score(graph.edges.iloc[i, :]).
            Edges that cannot be scored (e.g., because their type does not
            exist in the context graph) have a negative integer instead of a
            score.

        '''

        scores, to_score = self._split_edges(
            graph, context_graph, nodes=nodes
        )
        edges = graph.get_edges(return_dataframe=True).iloc[to_score]
        edges = Graph(edges)
        triples, data, pred_nodes = build_graph_data(
            edges, context_graph, nodes=nodes
        )
        scores[to_score] = self._compute_scores(
            triples, data, pred_nodes
        )
        return scores

    def _split_edges(self, graph, context_graph, nodes=None):
        '''
        Splits the given edges into the following categories: edges that exist
        in the context graph, edges that do not need to be scored (auxiliary
        edges), edges that cannot be scored (unknown edge type or node that
        does not exist in the context graph), and edges to score (all other
        edges).

        Arguments
        ---------
        graph : Graph
            Input edges.
        context_graph : Graph
            Context graph used for scoring.
        nodes : dict, default=None
            Name -> type map for the nodes.
            If None, edges with at least one node that does not exist in the
            context graph cannot be scored.

        Returns
        -------
        scores : np.array, shape=(graph.n_edges,)
            Initialized anomaly scores.
        to_score : list
            Indices of edges for which anomaly scores must be computed.

        '''

        scores = np.zeros(graph.n_edges)
        to_score = []
        cg_nodes = context_graph.get_nodes()
        rels = context_graph.get_rels()
        edges = set(context_graph.get_edges())
        for idx, (src, rel, dst, aux) in enumerate(
            graph.get_edges(return_aux=True)
        ):
            if aux == 1:
                scores[idx] = 0
            elif (src, rel, dst) in edges:
                scores[idx] = 0
            elif rel not in rels:
                scores[idx] = -1
            elif src not in cg_nodes and (nodes is None or src not in nodes):
                scores[idx] = -2
            elif dst not in cg_nodes and (nodes is None or dst not in nodes):
                scores[idx] = -2
            else:
                to_score.append(idx)
        return scores, to_score
