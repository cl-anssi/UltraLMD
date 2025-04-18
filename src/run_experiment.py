import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra.models import Ultra

from model import UltraLMD
from graph import Graph




def evaluate_model(model, base_dir, train_cutoff, nodes=None):
    '''
    Evaluates an instance of UltraLMD on a dataset.

    Arguments
    ---------
    model : UltraLMD
        Detector used for anomaly scoring.
    base_dir : str
        Path to the directory containing the dataset files.
        The file named {i}.csv represents the i-th time window.
        It has the following columns: src, rel, dst, auxiliary, label.
        Label should be 1 for malicious edges and 0 otherwise.
    train_cutoff : int
        Index of the first time window of the evaluation set.
        Previous windows are only stored and used to build future context
        graphs.
    nodes : dict, default=None
        Name -> type map for the nodes.
        If None, no type nodes and has_type edges are added to the context
        graphs used for anomaly scoring.

    Returns
    -------
    labels : list
        List of arrays containing the labels of the edges for each evaluation
        window.
    scores : list
        List of arrays containing the anomaly scores of the edges for each
        evaluation window.

    '''

    ids = [int(fname.split('.')[0]) for fname in os.listdir(base_dir)]
    scores, labels = [], []
    for i in sorted(ids):
        edges = pd.read_csv(
            os.path.join(base_dir, f'{i}.csv'),
            keep_default_na=False
        )
        graph = Graph(
            edges,
            node_idx=model.get_node_idx(),
            rel_idx=model.get_rel_idx()
        )
        if i >= train_cutoff:
            _scores, _labels = zip(*[
                (s, l)
                for s, l, a in zip(
                    model.score(graph, nodes=nodes),
                    edges['label'],
                    edges['auxiliary']
                )
                if a == 0
            ])
            labels.append(_labels)
            scores.append(_scores)
        model.update(graph)
    return labels, scores


parser = argparse.ArgumentParser()
parser.add_argument(
    'input_dir',
    help=(
        'Path to the directory containing the dataset.'
        'It should contain a file named nodes.csv with the types '
        'of the nodes (columns: "name" and "type"), and a directory '
        'named edges/ containing the edge lists. '
        'Each file in the edges/ directory, named $i.csv, represents one '
        'time window (with index $i) and has the following columns: '
        'src, rel, dst, auxiliary, label.'
    )
)
parser.add_argument(
    'output_dir',
    help='Path to the directory where the results are written.'
)
parser.add_argument(
    '--ckpt', required=True,
    help='Full path to the Ultra checkpoint.'
)
parser.add_argument(
    '--train_cutoff', required=True, type=int,
    help=(
        'Index of the first time window of the evaluation set. '
        'Previous windows are only stored and used to build future context '
        'graphs.'
    )
)
parser.add_argument(
    '--num_context_graphs', type=int, default=10,
    help=(
        'Number of past graphs to include in the short-term context.'
        'When set to a negative value, only the long-term context graph '
        'is used for scoring.'
    )
)
parser.add_argument(
    '--refine_scores', action='store_true',
    help='Whether to use the graph-based score refinement algorithm.'
)
parser.add_argument(
    '--batch_size', type=int, default=128,
    help='Size of the batches passed to the GFM.'
)
parser.add_argument(
    '--device', default='cuda',
    help='Device used by Torch for computation.'
)
args = parser.parse_args()

# Read the Ultra checkpoint and build the GFM
state = torch.load(
    args.ckpt,
    map_location='cpu',
    weights_only=True
)
config = [
    {
        'class': 'NBFNet',
        'input_dim': 64,
        'hidden_dims': [64] * 6,
        'layer_norm': True,
        'short_cut': True
    },
    {
        'class': 'IndNBFNet',
        'input_dim': 64,
        'hidden_dims': [64] * 6,
        'layer_norm': True,
        'short_cut': True
    }
]
base_model = Ultra(*config)
base_model.load_state_dict(state['model'])

# Build the detector
model = UltraLMD(
	base_model,
	num_context_graphs=args.num_context_graphs,
    batch_size=args.batch_size,
    refine_scores=args.refine_scores,
	device=args.device
)

# Read the name -> type map for the nodes
node_id = pd.read_csv(
    os.path.join(args.input_dir, 'nodes.csv'),
    keep_default_na=False
)
nodes = {
    node: node_type
    for node, node_type in zip(node_id['name'], node_id['type'])
}

# Run the detector on the dataset
labels, scores = evaluate_model(
	model,
	os.path.join(args.input_dir, 'edges'),
	train_cutoff=args.train_cutoff,
    nodes=nodes
)

# Write results
res = {
	'labels': [[int(l) for l in _labels] for _labels in labels],
	'scores': [[float(s) for s in _scores] for _scores in scores],
    'times': dict(zip(
        ['retrieval', 'scoring', 'refinement'],
        list(zip(*model.times))
    ))
}
if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)
fp = os.path.join(
	args.output_dir,
	(
        f'res_{args.num_context_graphs}'
        f'{"_refine" if args.refine_scores else ""}'
        '.json.gz'
    )
)
with gzip.open(fp, 'wt') as out:
	out.write(json.dumps(res))