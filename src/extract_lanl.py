import argparse
import csv
import gzip
import os

from collections import defaultdict

import pandas as pd




WIN_LEN = 3600
TEST_START = 150885
TOP_K_PROTO_PORT = 20

def domain_type(domain):
    '''
    Returns the type of a domain (host for a local account, domain for a domain
    account) based on its name.

    Arguments
    ---------
    domain : str
        Domain name.

    Returns
    -------
    type : str
        Type of the domain (host for a local account, domain for a domain
        account).

    '''

    if domain.startswith('C'):
        return '_host'
    return '_domain'

def extract_lanl_dataset(base_dir):
    '''
    Preprocesses the LANL dataset and returns the edge set for each time
    window as well as the name -> type mapping for the nodes.
    The input directory should contain the auth.txt.gz and redteam.txt.gz
    files.

    Arguments
    ---------
    base_dir : str
        Path to the directory containing the auth.txt.gz and redteam.txt.gz
        files.

    Returns
    -------
    edges : dict
        Window -> edge set map.
        Each edge set is a set of tuples (src, rel, dst, auxiliary, label),
        representing the source node, edge type, destination node, whether
        the edge should be scored (0 if it should be scored and 1 otherwise),
        and label (1 for malicious and 0 otherwise).
    nodes : dict
        Node name -> type map.

    '''

    edges = defaultdict(set)
    nodes = dict()
    # Extract labeled lateral movement edges
    rt_path = os.path.join(base_dir, 'redteam.txt.gz')
    redteam = pd.read_csv(rt_path, names=('timestamp', 'usr', 'src', 'dst'))
    rt_tuples = set([
        (ts, usr, src, dst)
        for ts, usr, src, dst in zip(
            redteam['timestamp'],
            redteam['usr'],
            redteam['src'],
            redteam['dst']
        )
    ])
    # Read the auth file
    with gzip.open(os.path.join(base_dir, 'auth.txt.gz'), 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[7] != 'LogOn':
                # We only consider LogOn events
                continue
            if line[5].startswith('MICROSOFT'):
                # Fix inconsistencies in the name of the MSV_1_0 authentication
                # package
                line[5] = 'MSV_1_0'
            ts = int(line[0])
            window = ts // WIN_LEN
            src_usr, src_dom = line[1].split('@')[:2]
            dst_usr, dst_dom = line[2].split('@')[:2]
            src_host, dst_host = line[3:5]
            auth_type = '_'.join(line[5:7])
            src_usr, src_dom, dst_usr, dst_dom, src_host, dst_host, auth_type = [
                x if x != '' else 'None'
                for x in (
                    src_usr, src_dom,
                    dst_usr, dst_dom,
                    src_host, dst_host,
                    auth_type
                )
            ]
            nodes.update({
                src_usr: user_type(src_usr),
                dst_usr: user_type(dst_usr),
                src_dom: domain_type(src_dom),
                dst_dom: domain_type(dst_dom),
                src_host: '_host',
                dst_host: '_host'
            })
            edges[window].update([
                (src_usr, 'has_domain', src_dom, 1, 0),
                (dst_usr, 'has_domain', dst_dom, 1, 0)
            ])
            if src_usr != dst_usr:
                # Authentication with new user account
                edges[window].add((src_usr, f'{auth_type}_uu', dst_usr, 1, 0))
            if src_host != dst_host:
                # Remote authentication
                lab = int((ts, line[1], src_host, dst_host) in rt_tuples)
                hh = (src_host, f'{auth_type}_hh', dst_host, 0, lab)
                edges[window].add(hh)
                edges[window].update([
                    (src_usr, f'{auth_type}_from', src_host, 1, 0),
                    (dst_usr, f'{auth_type}_to', dst_host, 1, 0)
                ])
            else:
                # Local authentication
                edges[window].update([
                    (src_usr, f'{auth_type}_local', src_host, 1, 0),
                    (dst_usr, f'{auth_type}_local', dst_host, 1, 0)
                ])
    # Read the flow file
    flows = set()
    with gzip.open(os.path.join(base_dir, 'flows.txt.gz'), 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            window = int(line[0]) // WIN_LEN
            src_host, dst_host = line[2], line[4]
            proto_port = f'{line[6]}_{line[5]}'
            for host in (src_host, dst_host):
                nodes[host] = '_host'
            flows.add((window, src_host, proto_port, dst_host))
    flows = pd.DataFrame(list(flows), columns=('time', 'src', 'rel', 'dst'))
    # Replace infrequent proto-port pairs with a placeholder
    # We count the occurrences in the training period only to avoid data
    # snooping
    first_test_window = TEST_START // WIN_LEN
    flows_train = flows[flows['time'] < first_test_window]
    counts = flows_train['rel'].value_counts().to_dict()
    freq_rels = set(sorted(
        list(counts.keys()), key=lambda x: -counts[x]
    )[:TOP_K_PROTO_PORT])
    encode_rel = lambda x: x if x in freq_rels else 'OTHER'
    flows['rel'] = flows['rel'].apply(encode_rel)
    flows.drop_duplicates(inplace=True)
    # Add flow-related edges to the edge sets
    for window, src, rel, dst in zip(
        flows['time'], flows['src'], flows['rel'], flows['dst']
    ):
        edges[window].add((src, rel, dst, 1, 0))
    return edges, nodes


def user_type(user):
    '''
    Returns the type of a user account (user, computer, built-in) based on
    its name.

    Arguments
    ---------
    user : str
        User name.

    Returns
    -------
    type : str
        Type of the user account (user, computer, or built-in).

    '''

    if user.startswith('U'):
        return '_user_account'
    if user.startswith('C'):
        return '_computer_account'
    return '_builtin_account'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir',
        help=(
            'Path to the directory containing the auth.txt.gz and '
            'redteam.txt.gz files of the LANL dataset.'
        )
    )
    parser.add_argument(
        '--output_dir',
        default='../datasets/lanl/',
        help='Path to the output directory.'
    )
    args = parser.parse_args()

    edges, nodes = extract_lanl_dataset(
        args.input_dir
    )
    if not os.path.exists(os.path.join(args.output_dir, 'edges')):
        os.makedirs(os.path.join(args.output_dir, 'edges'))
    # Write one file per time window
    for window, edge_set in edges.items():
        fp = os.path.join(args.output_dir, 'edges', f'{window}.csv')
        with open(fp, 'w') as out:
            out.write('src,rel,dst,auxiliary,label\n')
            for src, rel, dst, aux, label in edge_set:
                out.write(
                    f'{src},{rel},{dst},{aux},{label}\n'
                )
    # Write the node name -> type mapping
    node_df = pd.DataFrame(
        [[node, node_type] for node, node_type in nodes.items()],
        columns=('name', 'type')
    )
    node_df.to_csv(os.path.join(args.output_dir, 'nodes.csv'), index=False)