import numpy as np
import pandas as pd




def make_idx_and_vector(entity_set, initial_idx=None):
    '''
    Builds a name -> id map and indicator vector for the given set.

    Arguments
    ---------
    entity_set : set
        Set of strings (typically, node names).
    initial_idx : dict, default=None
        Name -> id map that will be completed with the names in entity_set.

    Returns
    -------
    vec : np.array, shape=(len(idx),)
        Zero-one indicator for the entities in entity_set.
    idx : dict
        Name -> id map.

    '''

    if initial_idx is None:
        idx = {}
    else:
        idx = initial_idx
    for name in sorted(list(entity_set)):
        if name not in idx:
            idx[name] = len(idx)
    vec = np.zeros(len(idx))
    for name in entity_set:
        vec[idx[name]] = 1
    return vec, idx

def op_count_vectors(v1, v2, func):
    '''
    Applies the given function of two arguments to the given vectors after
    padding them to the same length if necessary.

    Arguments
    ---------
    v1 : np.array
        First argument to be passed to the function.
    v2 : np.array
        Second argument to be passed to the function.
    func : callable
        Function of two arguments to apply to v1 and v2.

    Returns
    -------
    result : object
        Result of func(pad(v1), pad(v2)).

    '''

    max_len = max(len(v1), len(v2))
    pad_vec = lambda v: (
        v if len(v) >= max_len
        else np.concatenate([v, [0] * (max_len - len(v))])
    )
    return func(pad_vec(v1), pad_vec(v2))

class Graph:
    '''
    Helper class for operations on authentication and network flow graphs.

    Parameters
    ---------
    edges : pd.DataFrame
        Edges of the graph.
        Should at least have columns 'src', 'rel', and 'dst' containing the
        source node, type, and destination node of each edge, respectively.
        The 'auxiliary' column can be added; it takes the values 0 or 1 and
        indicates whether the edge must be given an anomaly score (0) or
        not (1).
    node_idx : dict, default=None
        Name -> id map for the nodes.
        Used to build the node indicator vector, which enables efficient
        computation of the Jaccard index between the node sets of two graphs.
        If None, a new map is created based on the edges.
    rel_idx : dict, default=None
        Name -> id map for the relations.
        Used to build the relation count vector, which enables efficient
        computation of the cosine similarity of the relation counts of two
        graphs.
        If None, a new map is created based on the edges.

    Attributes
    ----------
    nodes : set
        Set of active nodes in the graph (i.e., nodes with at least one edge).
    edges : pd.DataFrame
        Edges of the graph.
    n_nodes : int
        Number of active nodes in the graph.
    n_edges : int
        Number of edges in the graph.
    pred_nodes : set
        Set of candidate nodes for link prediction.
        It is the set of active nodes that appear in at least one non-auxiliary
        edge (i.e., an edge that is considered for anomaly scoring).
    node_vector : np.array, shape=(len(self.node_idx),)
        Indicator vector of the set of active nodes, which enables efficient
        computation of the Jaccard index between the node sets of two graphs.
    node_idx : dict
        Name -> id map for the nodes.
    rels : set
        Relations of the graph.
    rel_vector : np.array, shape=(len(self.rel_idx),)
        Relation count vector, where each element is the number of occurrences
        of the corresponding relation in the graph.
        It enables efficient computation of the cosine similarity of the
        relation counts of two graphs.
    rel_idx : dict
        Name -> id map for the relations.

    '''

    def __init__(self, edges, node_idx=None, rel_idx=None):
        cols = ['src', 'rel', 'dst']
        if 'auxiliary' in edges.columns:
            cols.append('auxiliary')
            pred_edges = edges[edges['auxiliary'] == 0]
        else:
            pred_edges = edges
        self.edges = edges.loc[:, cols]
        self.nodes = set(edges['src']).union(set(edges['dst']))
        self.node_vector, self.node_idx = make_idx_and_vector(
            self.nodes, node_idx
        )
        self.pred_nodes = set(pred_edges['src']).union(set(pred_edges['dst']))
        self.n_nodes = len(self.nodes)
        self.n_edges = edges.shape[0]
        rels = edges['rel'].value_counts().to_dict()
        self.rels = set(rels.keys())
        self.rel_vector, self.rel_idx = make_idx_and_vector(
            self.rels, rel_idx
        )
        for rel in rels:
            self.rel_vector[self.rel_idx[rel]] = rels[rel]

    def copy(self):
        '''
        Returns a copy of this Graph instance.

        Returns
        -------
        new_graph : Graph
            New Graph with same attribute values as self.

        '''

        edges = self.edges.copy()
        node_idx = self.node_idx.copy()
        rel_idx = self.rel_idx.copy()
        return Graph(edges, node_idx=node_idx, rel_idx=rel_idx)

    def get_edges(
        self, return_aux=False, return_dataframe=False
    ):
        '''
        Returns the edges of this graph as a list or dataframe.

        Arguments
        ---------
        return_aux : boolean, default=False
            Whether to include the 'auxiliary' column.
        return_dataframe : boolean, default=False
            Whether to return a new dataframe or a list (list by default).

        Returns
        -------
        edges : list or pd.DataFrame
            List of edges.

        '''

        cols = ['src', 'rel', 'dst']
        if return_aux:
            cols.append('auxiliary')
        if return_dataframe:
            edges = self.edges.loc[:, cols].copy()
        else:
            edges = list(zip(*[self.edges[col] for col in cols]))
        return edges

    def get_node_idx(self):
        '''
        Returns the name -> id map for the graph's nodes.

        Returns
        -------
        node_idx : dict
            Name -> id map for the nodes.

        '''

        return self.node_idx

    def get_node_vector(self):
        '''
        Returns the indicator vector for the graph's active nodes.

        Returns
        -------
        node_vec : np.array, shape=(len(self.node_idx),)
            Indicator vector for the active nodes.

        '''

        return self.node_vector

    def get_nodes(self):
        '''
        Returns the set of active nodes in the graph.

        Returns
        -------
        nodes : set
            Set of active nodes.

        '''

        return self.nodes

    def get_pred_nodes(self):
        '''
        Returns the set of candidate nodes for link prediction.

        Returns
        -------
        pred_nodes : set
            Candidate nodes for link prediction.

        '''

        return self.pred_nodes

    def get_rel_idx(self):
        '''
        Returns the name -> id map for the graph's relations.

        Returns
        -------
        rel_idx : dict
            Name -> id map for the relations.

        '''
        return self.rel_idx

    def get_rel_vector(self):
        '''
        Returns the count vector for the graph's relations.

        Returns
        -------
        rel_vec : np.array, shape=(len(self.rel_idx),)
            Vector of relation counts.

        '''

        return self.rel_vector

    def get_rels(self):
        '''
        Returns the graph's set of relations.

        Returns
        -------
        rels : set
            Set of relations.

        '''

        return self.rels

    def refine_scores(self, scores):
        '''
        Applies the anomaly score refinement algorithm to the provided scores.
        The algorithm sets the score of each edge to the maximum score among
        neighboring edges if this maximum score is lower.

        Arguments
        ---------
        scores : np.array, shape=(self.n_edges,)
            Vector of anomaly scores.
            It should be aligned with self.edges, i.e., scores[i] is the
            anomaly score of self.edges.iloc[i, :].

        Returns
        -------
        refined : np.array, shape=(self.n_edges,)
            Vector of refined anomaly scores.

        '''

        edges = self.edges.loc[:, ['src', 'dst']].copy()
        edges['score'] = scores
        # We use a message passing algorithm :
        # - for each node, compute the second highest score among edges
        #   incident to this node
        # - for each edge, compute the maximum score of its nodes, then
        #   take the minimum between the result and its score.
        # If there is at least one neighboring edge with a higher score,
        # this changes nothing.
        # However, if the edge is the highest-scoring among its neighborhood,
        # the message passing algorithm computes the maximum score among
        # neighboring edges and replaces the central edge's score with it.
        node_scores = pd.concat([
            edges.loc[:, ['src', 'score']].rename({'src': 'node'}, axis=1),
            edges.loc[:, ['dst', 'score']].rename({'dst': 'node'}, axis=1)
        ])
        node_scores = node_scores.groupby(['node']).agg(
            {'score': lambda xs: sorted(xs)[-2] if len(xs) > 1 else max(xs)}
        ).to_dict()['score']
        refined = [
            min(y, max(node_scores[s], node_scores[d]))
            for y, s, d in zip(edges['score'], edges['src'], edges['dst'])
        ]
        return refined

    def update(self, graph):
        '''
        Merges this graph with another, so that it becomes the union of itself
        and the other graph.

        Arguments
        ---------
        graph : Graph
            Other graph to merge with self.

        '''

        self.nodes.update(graph.get_nodes())
        self.pred_nodes.update(graph.get_pred_nodes())
        cols = ['src', 'rel', 'dst', 'auxiliary']
        self.edges = pd.concat([
            self.edges,
            graph.get_edges(return_aux=True, return_dataframe=True)
        ]).drop_duplicates()
        self.n_nodes = len(self.nodes)
        self.n_edges = self.edges.shape[0]
        self.rels.update(graph.get_rels())
        self.node_vector = op_count_vectors(
            self.node_vector,
            graph.get_node_vector(),
            np.fmax
        )
        self.node_idx.update(graph.get_node_idx())
        self.rel_vector = op_count_vectors(
            self.rel_vector,
            graph.get_rel_vector(),
            lambda x, y: x + y
        )
        self.rel_idx.update(graph.get_rel_idx())


def aggregate_graphs(graphs):
    '''
    Computes the union of all graphs in the given iterable.

    Arguments
    ---------
    graphs : iterable
        Instances of Graph to merge.

    Returns
    -------
    union : Graph
        Union of the input graphs.

    '''

    cols = ['src', 'rel', 'dst', 'auxiliary']
    edges = pd.concat([
        g.get_edges(return_aux=True, return_dataframe=True)
        for g in graphs
    ]).loc[:, cols]
    return Graph(edges.drop_duplicates())