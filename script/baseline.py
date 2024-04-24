from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.special import digamma

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from pyCP_APR import CP_APR




class LinkPredictor(BaseEstimator, metaclass=ABCMeta):
    '''
    Abstract class for baseline link predictors.

    '''

    def fit(self, X, y=None, nodes=None, rels=None):
        '''
        Fit the model parameters.

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        self : LinkPredictor
            Fitted estimator.

        '''

        self.rnd = check_random_state(self.random_state)
        edges = self._make_graph(
            X,
            nodes=nodes,
            rels=rels
        )
        self.scores = self._fit_params(edges)
        return self


    def score(self, X):
        '''
        Returns a goodness-of-fit score for the estimator, computed using the
        given validation set (the higher, the better).

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        score : float
            Goodness-of-fit score (the higher, the better).

        '''

        edges = self._build_graph(X)
        return self._validation_score(edges)


    def score_samples(self, X):
        '''
        Returns the anomaly scores of the input edges (the higher, the more
        anomalous).

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        scores : np.array, shape (n_edges,)
            Anomaly scores (the higher, the more anomalous).

        '''

        edges = self._build_graph(X)
        return 1 - self._score(edges)


    def _build_graph(self, X):
        '''
        Encodes the input edges using internal node -> id and rel -> id
        mappings.

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        edges : np.array, shape (n_edges, 3)
            Encoded adjacency list.

        '''

        edges = np.stack(
            [
                self.node_encoder.transform(X[:, i])
                for i in (0, 2)
            ] + [self.rels_encoder.transform(X[:, 1])],
            axis=1
        )
        return edges


    @abstractmethod
    def _fit_params(self, edges):
        '''
        Learning algorithm specific to each link predictor.

        '''

        return 0


    def _make_graph(
            self,
            X,
            nodes=None,
            rels=None
        ):
        '''
        Builds the internal node -> id and rel -> id mappings and returns the
        encoded version of the input edges.

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).
        nodes : list, default=None
            Node list.
            If None, it is inferred from the input edges.
        rels : list, default=None
            Relation list.
            If None, it is inferred from the input edges.

        Returns
        -------
        edges : np.array, shape (n_edges, 3)
            Encoded adjacency list.

        '''

        if nodes is None:
            nodes = sorted(list(set(
                np.concatenate([X[:, 0], X[:, 2]])
            )))
        if rels is None:
            rels = sorted(list(set(X[:, 1])))
        self.node_encoder = LabelEncoder().fit(nodes)
        self.n_nodes = len(nodes)
        self.rels_encoder = LabelEncoder().fit(rels)
        self.n_rels = len(rels)
        return self._build_graph(X)


    @abstractmethod
    def _score(self, edges):
        '''
        Scoring method specific to each link predictor.

        '''

        return 0


    def _validation_score(self, edges):
        '''
        Returns a goodness-of-fit score for the estimator, computed using the
        given encoded validation set (the higher, the better).

        Arguments
        ---------
        edges : np.array, shape (n_edges, 3)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        score : float
            Goodness-of-fit score (the higher, the better).

        '''

        return self._score(edges).mean()


class PTF(LinkPredictor):
    '''
    Three-mode Poisson tensor factorization for link prediction in knowledge
    graphs, based on the implementation from [1].

    Parameters
    ----------
    dimension : int, default=64
        Dimension of the latent factors.
    max_iter : int, default=1000
        Maximum number of iterations in the learning procedure.
    random_state : int, default=None
        Seed for the random number generator.

    References
    ----------
    [1] Eren, Maksim, et al. General-purpose unsupervised cyber anomaly
        detection via non-negative tensor factorization.
        Digit. Threat. 4(1), 2023.

    '''

    def __init__(self, dimension=64, max_iter=1000, random_state=None):
        self.dimension = dimension
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None


    def _fit_params(self, edges):
        '''
        Fit the parameters of the model using the given encoded edges.

        Arguments
        ---------
        edges : np.array, shape (n_edges, 3)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        scores : list
            Values of the log-likelihood at successive iterations of the
            learning procedure.

        '''

        self.model = CP_APR(
            method='torch',
            device='gpu' if torch.cuda.is_available() else 'cpu',
            return_type='numpy',
            n_iters=self.max_iter,
            verbose=0,
            dtype='torch.FloatTensor',
            random_state=self.random_state
        )
        y = np.full(
            (edges.shape[0],),
            int(self.n_rels * self.n_nodes ** 2 / edges.shape[0])
        )
        self.model.fit(
            coords=edges,
            values=y,
            rank=[1, self.dimension]
        )
        # Initialize latent factors for nodes not seen in the training set
        for i in (0, 2):
            diff = self.n_nodes - edges[:, i].max()
            if diff > 0:
                for j in range(2):
                    fac = self.model.M[j]['Factors'][str(i)]
                    if fac.ndim == 2:
                        to_add = np.tile(
                            fac.mean(0),
                            (diff, 1)
                        )
                    else:
                        to_add = fac.mean()*np.ones(diff)
                    self.model.M[j]['Factors'][str(i)] = np.concatenate(
                        [fac, to_add],
                        axis=0
                    )
        return list(self.model.get_params()['logLikelihoods'])


    def _score(self, edges):
        '''
        Returns the predicted p-values of the input encoded edges (the lower,
        the more anomalous).

        Arguments
        ---------
        edges : np.array, shape (n_edges, 3)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        scores : np.array, shape (n_edges,)
            p-values of the input edges (the lower, the more anomalous).

        '''

        return self.model.predict_scores(
            coords=edges,
            values=np.ones(edges.shape[0])
        )


class HPF(LinkPredictor):
    '''
    Hierarchical Poisson factorization for link prediction in (knowledge)
    graphs, based on [1].

    Parameters
    ----------
    dimension : int, default=64
        Dimension of the latent factors.
    a : float, default=1
        Shape parameter for the Gamma prior of the latent factors in the
        HPF model.
    b : float, default=1
        Shape parameter for the Gamma prior of the rate parameters in the
        HPF model.
    c : Rate parameter for the Gamma prior of the rate parameters in the
        HPF model.
    epsilon : float, default=1e-4
        Stopping criterion for the learning procedure.
    max_iter : int, default=1000
        Maximum number of iterations in the learning procedure.
    random_state : int, default=None
        Seed for the random number generator.

    References
    ----------
    [1] Sanna Passino, Francesco, et al. Graph link prediction in computer
        networks using Poisson matrix factorisation.
        Ann. Appl. Stat. 16(3), 2022.

    '''

    def __init__(
        self,
        dimension=64,
        a=1,
        b=1,
        c=.1,
        epsilon=1e-4,
        max_iter=1000,
        random_state=None
    ):

        self.dimension = dimension
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state


    def _build_graph(self, X):
        '''
        Encodes the input edges using the internal node -> id mapping.
        The relations are dropped since the HPF model does not include
        them.

        Arguments
        ---------
        X : np.array, shape (n_edges, 3)
            Adjacency list.
            Each row of the array represents one edge, with elements
            (source, relation, destination).

        Returns
        -------
        edges : np.array, shape (n_edges, 2)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, destination).

        '''

        edges = np.stack(
            [
                self.node_encoder.transform(X[:, i])
                for i in (0, 2)
            ],
            axis=1
        )
        return edges


    def _fit_params(self, edges):
        '''
        Fit the parameters of the model using the given encoded edges.

        Arguments
        ---------
        edges : np.array, shape (n_edges, 2)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, destination).

        Returns
        -------
        scores : list
            Values of the likelihood at successive iterations of the
            learning procedure.

        '''

        self._initialize()
        unique_edges = np.unique(edges, axis=0)
        old_score = self._validation_score(unique_edges)
        diff = 10
        n_iter = 0
        scores = [old_score]
        while diff > self.epsilon and n_iter < self.max_iter:
            self.theta, self.chi = self._update_theta_chi(unique_edges)
            self.lambda_alpha, self.mu_alpha = self._update_lambda_mu(
                unique_edges, 0
            )
            self.lambda_beta, self.mu_beta = self._update_lambda_mu(
                unique_edges, 1
            )
            self.xi_alpha = self._update_xi(0)
            self.xi_beta = self._update_xi(1)
            self.alpha = self.lambda_alpha / self.mu_alpha
            self.beta = self.lambda_beta / self.mu_beta

            score = self._validation_score(unique_edges)
            scores.append(score)
            diff = np.abs(1 - score / old_score)
            old_score = score
            n_iter += 1
            if n_iter % 10 == 0:
                print(f'Iteration {n_iter}; score: {score}')
        return scores


    def _initialize(self):
        '''
        Random initialization of the model parameters.

        '''

        self.lambda_alpha, self.mu_alpha, self.alpha, \
            self.xi_alpha, self.nu_alpha = self._make_params()
        self.lambda_beta, self.mu_beta, self.beta, \
            self.xi_beta, self.nu_beta = self._make_params()


    def _make_params(self, offset=.1):
        '''
        Returns randomly initialized parameters and auxiliary variables.

        Arguments
        ---------
        offset : float, default=.1
            Minimum value of the generated parameters.

        Returns
        -------
        new_lambda : np.array, shape (n_nodes, dimension)
            Shape parameter of the variational distribution of the latent
            factors.
        new_mu : np.array, shape (n_nodes, dimension)
            Rate parameter of the variational distribution of the latent
            factors.
        new_fac : np.array, shape (n_nodes, dimension)
            Expected value of the latent factors under the variational
            distribution.
        new_xi : np.array, shape (n_nodes,)
            Rate parameter of the variational distribution of the
            node-specific rate parameters.
        new_nu : np.array, shape (n_nodes,)
            Shape parameter of the variational distribution of the
            node-specific rate parameters.

        '''

        new_lambda = self.a + self.rnd.uniform(
            offset, size=(self.n_nodes, self.dimension)
        )
        new_mu = self.b / self.c + self.rnd.uniform(
            offset, size=(self.n_nodes, self.dimension)
        )
        new_fac = new_lambda / new_mu
        new_xi = self.c + self.rnd.uniform(
            offset, size=(self.n_nodes,)
        )
        new_nu = np.full(
            self.n_nodes, self.b + self.dimension * self.a
        )
        return new_lambda, new_mu, new_fac, new_xi, new_nu


    def _score(self, edges):
        '''
        Returns the predicted probabilities of the input encoded edges (the
        lower, the more anomalous).

        Arguments
        ---------
        edges : np.array, shape (n_edges, 2)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, destination).

        Returns
        -------
        scores : np.array, shape (n_edges,)
            Probabilities of the input edges (the lower, the more anomalous).

        '''

        alpha = self.alpha[edges[:, 0], :]
        beta = self.beta[edges[:, 1], :]
        return 1 - np.exp(-(alpha * beta).sum(1))


    def _update_lambda_mu(self, edges, axis):
        '''
        Updates the parameters of the variational distribution of the latent
        factors of the source or destination nodes given the training edges.

        Arguments
        ---------
        edges : np.array, shape (n_edges, 2)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, destination).
        axis : int (0 or 1)
            If axis==0, the latent factors of the source nodes are updated.
            Otherwise, the latent factors of the destination nodes are
            updated.

        Returns
        -------
        res_lambda : np.array, shape (n_nodes, dimension)
            New shape parameter of the variational distribution of the latent
            factors.
        res_mu : np.array, shape (n_nodes, dimension)
            New rate parameter of the variational distribution of the latent
            factors.

        '''

        tmp = self.theta[:, np.newaxis] * self.chi
        tmp /= 1 - np.exp(-self.theta)[:, np.newaxis]
        tmp_mat = [
            csr_matrix(
                (
                    tmp[:, k],
                    (edges[:, 0], edges[:, 1])
                ),
                shape=(self.n_nodes, self.n_nodes)
            )
            for k in range(self.dimension)
        ]
        if axis == 0:
            res_lambda = self.a + np.stack([
                np.array(mat.sum(1))[:, 0]
                for mat in tmp_mat
            ], axis=1)
            n, x = self.nu_alpha, self.xi_alpha
            l, m = self.lambda_beta, self.mu_beta
        else:
            res_lambda = self.a + np.stack([
                np.array(mat.sum(0))[0, :]
                for mat in tmp_mat
            ], axis=1)
            n, x = self.nu_beta, self.xi_beta
            l, m = self.lambda_alpha, self.mu_alpha
        res_mu = (n / x)[:, np.newaxis] + (l / m).sum(0)[np.newaxis, :]
        return res_lambda, res_mu


    def _update_theta_chi(self, edges):
        '''
        Updates the auxiliary variables used in the inference procedure.

        Arguments
        ---------
        edges : np.array, shape (n_edges, 2)
            Encoded adjacency list.
            Each row of the array represents one edge, with elements
            (source, destination).

        Returns
        -------
        theta : np.array, shape (n_edges,)
            New estimates of the rates corresponding to the training edges.
        chi : np.array, shape (n_edges, dimension)
            New estimates of the contributions of each latent dimension to
            the rates of the training edges.

        '''

        lambda_alpha = self.lambda_alpha[edges[:, 0], :]
        mu_alpha = self.mu_alpha[edges[:, 0], :]
        lambda_beta = self.lambda_beta[edges[:, 1], :]
        mu_beta = self.mu_beta[edges[:, 1], :]
        chi = np.exp(
            digamma(lambda_alpha) - np.log(mu_alpha)
            + digamma(lambda_beta) - np.log(mu_beta)
        )
        theta = chi.sum(1)
        chi /= theta[:, np.newaxis]
        return theta, chi


    def _update_xi(self, axis):
        '''
        Updates the parameters of the variational distribution of the
        node-specific rate parameters.

        Arguments
        ---------
        axis : int (0 or 1)
            If axis==0, the rate parameters of the source nodes are updated.
            Otherwise, the rate parameters of the destination nodes are
            updated.

        Returns
        -------
        res_xi : np.array, shape (n_nodes,)
            New shape parameter of the variational distribution of the
            node-specific rate parameters.

        '''

        if axis == 0:
            res = self.c + (self.lambda_alpha / self.mu_alpha).sum(1)
        else:
            res = self.c + (self.lambda_beta / self.mu_beta).sum(1)
        return res


def read_dataset(data_dir):
    '''
    Reads the input files and returns the preprocessed data.

    Arguments
    ---------
    data_dir : str
        Path to the directory containing the input files.

    Returns
    -------
    edges_train : np.array, shape (n_edges_train, 3)
        Adjacency list of the training graph.
        Each row of the array represents one edge, with elements
        (source, relation, destination).
    edges_valid : np.array, shape (n_edges_valid, 3)
        Adjacency list of the validation edges.
        Each row of the array represents one edge, with elements
        (source, relation, destination).
    edges_test : np.array, shape (n_edges_test, 3)
        Adjacency list of the test edges.
        Each row of the array represents one edge, with elements
        (source, relation, destination).
    labels : np.array, shape (n_edges_test,)
        Labels of the test edges (0 for benign, 1 for malicious).
    pred_nodes : list
        List of nodes that are considered for link prediction.

    '''

    cols = ['src', 'rel', 'dst']
    edges_train = pd.read_csv(
        os.path.join(data_dir, 'train.txt'),
        names=cols
    ).fillna('NaN')
    edges_valid = pd.read_csv(
        os.path.join(data_dir, 'valid.txt'),
        names=cols
    ).fillna('NaN')
    edges_test = pd.read_csv(
        os.path.join(data_dir, 'detect.txt'),
        names=cols + ['lab']
    ).fillna('NaN')
    X_test = edges_test.to_numpy()[:, :-1]
    y_test = edges_test['lab'].to_numpy()
    pred_nodes = pd.read_csv(
        os.path.join(data_dir, 'nodes.txt'),
        names=['node']
    )['node'].to_list()
    return (
        edges_train.to_numpy(),
        edges_valid.to_numpy(),
        X_test,
        y_test,
        pred_nodes
    )


if __name__ == '__main__':
    import argparse
    import json
    import os

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to the directory containing the input files.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to the output directory.'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        nargs='+',
        default=[16, 32, 64, 128],
        help=(
            'Candidate values for the embedding dimension (the best value) '
            'is selected by maximizing the probability of the validation set.'
        )
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for the random number generator.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    X_train, X_val, X_test, y_test, pred_nodes = read_dataset(args.data_dir)
    # Exclude from the validation set the edges involving nodes or relations
    # absent from the training set
    nodes = set().union(X_train[:, 0], X_train[:, 2])
    rels = set(X_train[:, 1])
    idx = [
        src in nodes and dst in nodes and rel in rels
        for src, rel, dst in X_val
    ]
    X_val_checked = X_val[idx]

    # Fit one estimator for each candidate embedding dimension
    estimators = [
        HPF(dimension=d, random_state=args.seed).fit(X_train)
        for d in args.dimension
    ]
    # Select the best embedding dimension
    idx = np.argmax([est.score(X_val_checked) for est in estimators])
    # Refit the estimator
    est = estimators[idx].fit(np.concatenate([X_train, X_val]))
    # Score test edges
    scores = est.score_samples(X_test)
    res = {
        'scores': scores.astype(float).tolist(),
        'labels': y_test.tolist(),
        'seed': args.seed,
        'model': 'HPF'
    }
    out_path = os.path.join(
        args.output_dir,
        f"res_{res['model']}_none_{args.seed}.json"
    )
    with open(out_path, 'w') as out:
        out.write(json.dumps(res))

    # Fit one estimator for each candidate embedding dimension
    estimators = [
        PTF(dimension=d, random_state=args.seed).fit(X_train)
        for d in args.dimension
    ]
    # Select the best embedding dimension
    idx = np.argmax([est.score(X_val_checked) for est in estimators])
    # Refit the estimator
    est = estimators[idx].fit(np.concatenate([X_train, X_val]))
    # Score test edges
    scores = est.score_samples(X_test)
    res = {
        'scores': scores.astype(float).tolist(),
        'labels': y_test.tolist(),
        'seed': args.seed,
        'model': 'PTF'
    }
    out_path = os.path.join(
        args.output_dir,
        f"res_{res['model']}_none_{args.seed}.json"
    )
    with open(out_path, 'w') as out:
        out.write(json.dumps(res))