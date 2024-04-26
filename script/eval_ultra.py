import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

from scipy.special import softmax

from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra.models import Ultra
from ultra.tasks import build_relation_graph




@torch.no_grad()
def eval_model(
        model,
        data,
        edges,
        batch_size=128,
        flip=False,
        pred_nodes=None
    ):
    '''
    Compute the conditional probabilities of the input edges given the model
    and the context graph.

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
    keys = sorted(list(set([(src, rel) for src, _, rel in _edges])))
    for i in range(0, len(keys), batch_size):
        h_index, r_index = torch.tensor([
            [src, rel]
            for src, rel in keys[i:i + batch_size]
        ], dtype=torch.long, device=device).t()
        r_index = r_index.unsqueeze(-1).expand(-1, data.num_nodes)
        all_index = torch.arange(data.num_nodes, device=device)
        h_index, t_index = torch.meshgrid(h_index, all_index, indexing='ij')
        batch = torch.stack([h_index, t_index, r_index], dim=-1)
        preds = model(data, batch).cpu().numpy()
        for j, key in enumerate(keys[i:i + batch_size]):
            scores[key] = preds[j, :]
        if (i // batch_size) % 10 == 0:
            print('Batch #', i // batch_size)
    if pred_nodes is None:
        pred_nodes = range(data.num_nodes)
    nodes = list(pred_nodes)
    idx = [0] * data.num_nodes
    for i, n in enumerate(nodes):
        idx[n] = i
    res = 1 - np.array([
        softmax(scores[(src, rel)][nodes])[idx[dst]]
        for src, dst, rel in _edges
    ])
    return res


def parse_log_file(log_file_path):
    '''
    Reads the log file generated during training and returns the RNG seed,
    initial checkpoint, checkpoint corresponding to maximal validation
    performance, and dataset used for training.

    Arguments
    ---------
    log_file_path : str
        Path to the log.txt file.

    Returns
    -------
    seed : int
        Seed of the random number generator.
    init_ckpt : str
        File name of the initial checkpoint (none if the model was trained
        from scratch).
    ckpt : str
        Path to the checkpoint achieving maximal validation performance.
    dataset : str
        Name of the dataset used for training.

    '''

    with open(log_file_path) as file:
        log = file.read().split('\n')
    ckpt_lines = [l for l in log if "'checkpoint': " in l]
    if len(ckpt_lines) > 0:
        line = ckpt_lines[0]
        init_ckpt = line.split(':')[-1][1:-1].replace("'", '').split('/')[-1].split('.')[0]
    else:
        init_ckpt = 'none'
    seed = log[0].split(':')[-1][1:]
    line = [l for l in log if 'Load checkpoint from' in l][0]
    ckpt = line.strip().split(' ')[-1]
    dirpath = os.path.dirname(os.path.abspath(log_file_path))
    ckpt = os.path.join(dirpath, ckpt)
    line = [l for l in log if "'dataset': {" in l][0]
    dataset = re.search("'class': '[a-zA-Z0-9_]+'", line).group(0)[10:-1]
    return seed, init_ckpt, ckpt, dataset


def read_dataset(data_dir):
    '''
    Reads the input files and returns the preprocessed data.

    Arguments
    ---------
    data_dir : str
        Path to the directory containing the input files.

    Returns
    -------
    data : Data
        Context graph.
    new_edges : list
        Adjacency list of the test edges.
        Each element of the list is a tuple representing one edge, with
        elements (source, destination, relation).
    labels : list
        Labels of the test edges (0 for benign, 1 for malicious).
    pred_nodes : list
        List of nodes that are considered for link prediction.

    '''

    cols = ['src', 'rel', 'dst']
    edges_train = pd.concat([
        pd.read_csv(os.path.join(data_dir, 'train.txt'), names=cols),
        pd.read_csv(os.path.join(data_dir, 'valid.txt'), names=cols)
    ]).fillna('NaN')
    edges_test = pd.read_csv(
        os.path.join(data_dir, 'detect.txt'),
        names=cols + ['lab']
    ).fillna('NaN')
    node_idx = {}
    nodes = sorted(list(set().union(
        edges_train['src'],
        edges_train['dst']
    )))
    for n in nodes:
        node_idx[n] = len(node_idx)
    rel_types = sorted(list(set(edges_train['rel'])))
    rel_idx = {}
    for t in rel_types:
        rel_idx[t] = len(rel_idx)
    new_edges = [
        (node_idx[h], node_idx[t], rel_idx[r])
        for _, h, r, t, _ in edges_test.itertuples()
    ]
    labels = edges_test['lab'].to_list()
    edge_index = torch.tensor([
            [node_idx[s] for s in edges_train['src']],
            [node_idx[d] for d in edges_train['dst']]
        ],
        dtype=torch.long
    )
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    edge_type = torch.tensor(
        [rel_idx[r] for r in edges_train['rel']],
        dtype=torch.long
    )
    edge_type = torch.cat([edge_type, edge_type + len(rel_idx)], dim=-1)
    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_relations=2 * len(rel_idx),
        num_nodes=len(node_idx)
    )
    nodes = pd.read_csv(os.path.join(data_dir, 'nodes.txt'), names=['node'])
    pred_nodes = [
        node_idx[n]
        for n in nodes['node']
    ]
    return data, new_edges, labels, pred_nodes


if __name__ == '__main__':
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
        '--ckpt',
        default=None,
        help=(
            'Path to the model checkpoint to use for scoring. '
            'Overrides the checkpoint extracted from the log file.'
        )
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help='Path to the log.txt file generated during training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Size of the input edge batches for scoring.'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device used for computations.'
    )
    args = parser.parse_args()

    data, new_edges, labels, pred_nodes = read_dataset(
        args.data_dir
    )
    device = torch.device(args.device)
    data = build_relation_graph(data).to(device)

    if args.log_file is None:
        # If no log file is provided, the model used for scoring must be
        # passed using the ckpt argument
        seed = 0
        init_ckpt = args.ckpt
        ckpt = args.ckpt
        dataset = 'none'
    else:
        seed, init_ckpt, ckpt, dataset = parse_log_file(args.log_file)
        if args.ckpt is not None:
            ckpt = args.ckpt
    state = torch.load(
        ckpt,
        map_location='cpu',
        weights_only=True
    )
    # Build model
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
    model = Ultra(*config)
    model.load_state_dict(state['model'])
    model = model.to(device)
    # Compute scores
    scores = eval_model(
        model,
        data,
        new_edges,
        batch_size=args.batch_size,
        pred_nodes=pred_nodes
    )
    scores += eval_model(
        model,
        data,
        new_edges,
        batch_size=args.batch_size,
        flip=True,
        pred_nodes=pred_nodes
    )
    res = {
        'scores': scores.astype(float).tolist(),
        'labels': labels,
        'seed': seed,
        'model': init_ckpt.split('/')[-1].split('.')[0],
        'finetuning_dataset': dataset
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_path = os.path.join(
        args.output_dir,
        f"res_{res['model']}_{dataset}_{seed}.json"
    )
    with open(out_path, 'w') as out:
        out.write(json.dumps(res))
