import argparse
import csv
import gzip
import json
import os

from collections import defaultdict

import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split




#####################################
# Helper functions for LANL dataset #
#####################################

def extract_lanl_dataset(base_dir):
    '''
    Preprocesses the LANL dataset and returns lists of edges corresponding to
    the basic and rich representations of the context graph as well as the
    test set.
    The input directory should contain the auth.txt.gz and redteam.txt.gz
    files.

    Arguments
    ---------
    base_dir : str
        Path to the directory containing the auth.txt.gz and redteam.txt.gz
        files.

    Returns
    -------
    edges_train_base : list
        Adjacency list for the basic representation of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    edges_test : list
        Adjacency list for the test edges.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    edges_train_rich : list
        Adjacency list for the additional edges in the enriched representation
        of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).

    '''

    edges_train_base = set()
    edges_train_rich = set()
    edges_test = set()
    edges_malicious = set()
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
    # Extract the list of days with red team activity (test set)
    rt_days = set([ts // 86400 for ts, _, _, _ in rt_tuples])
    # Read the main file
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
            day = ts // 86400
            src_usr, src_dom = line[1].split('@')[:2]
            dst_usr, dst_dom = line[2].split('@')[:2]
            src_host, dst_host = line[3:5]
            auth_type = '_'.join(line[5:7])
            hh = (src_host, f'{auth_type}_hh', dst_host)
            if day in rt_days and src_host != dst_host:
                # Remote authentications happening on days with red team
                # activity are added to the test set
                if (ts, line[1], src_host, dst_host) in rt_tuples:
                    edges_malicious.add(hh)
                edges_test.add(hh)
            else:
                # If the event happens on a day without red team activity, we
                # add the corresponding edges to the training set
                edges_train_rich.update([
                    (src_usr, 'has_domain', src_dom),
                    (dst_usr, 'has_domain', dst_dom)
                ])
                if src_usr != dst_usr:
                    # Authentication with new user account
                    edges_train_rich.add((src_usr, f'{auth_type}_uu', dst_usr))
                if src_host != dst_host:
                    # Remote authentication
                    edges_train_base.add(hh)
                    edges_train_rich.update([
                        (src_usr, f'{auth_type}_from', src_host),
                        (dst_usr, f'{auth_type}_to', dst_host)
                    ])
                else:
                    # Local authentication
                    edges_train_rich.update([
                        (src_usr, f'{auth_type}_local', src_host),
                        (dst_usr, f'{auth_type}_local', dst_host)
                    ])
    # Filter out test edges present in the context graph or involving nodes or
    # relations absent from the context graph
    nodelist = set().union(
        [h for h, _, _ in edges_train_base],
        [t for _, _, t in edges_train_base]
    )
    rels = set([r for _, r, _ in edges_train_base])
    edges_test = [
        (*edge, int(edge in edges_malicious))
        for edge in edges_test
        if edge not in edges_train_base
        and edge[0] in nodelist
        and edge[1] in rels
        and edge[2] in nodelist
    ]
    return list(edges_train_base), edges_test, list(edges_train_rich)


#####################################
# Helper functions for OpTC dataset #
#####################################

def extract_optc_dataset(base_dir, n_jobs=-1):
    '''
    Preprocesses the OpTC dataset and returns lists of edges corresponding to
    the basic and rich representations of the context graph as well as the
    test set.
    The input directory should be the ecar/ directory, with subdirectories
    benign/, short/ and evaluation/ containing the original GZIP-compressed
    JSON log files.

    Arguments
    ---------
    base_dir : str
        Path to the ecar/ directory containing the input log files.
    n_jobs : int, default=-1
        Number of parallel jobs to create (if -1, it is set to the number of
        available CPU cores).

    Returns
    -------
    edges_train_base : list
        Adjacency list for the basic representation of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    edges_test : list
        Adjacency list for the test edges.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    edges_train_rich : list
        Adjacency list for the additional edges in the enriched representation
        of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).

    '''

    # Read the raw data
    res = Parallel(n_jobs=n_jobs)(
        delayed(process_dir)(dirpath, filenames)
        for dirpath, _, filenames in os.walk(base_dir)
    )
    # Merge the results of the reading jobs
    edges_train_base, edges_train_rich, _ = merge_edges_hosts([
        (edges_base, edges_rich, hosts)
        for edges_base, edges_rich, hosts, is_test in res
        if not is_test
    ])
    edges_test_base, edges_test_rich, _ = merge_edges_hosts([
        (edges_base, edges_rich, hosts)
        for edges_base, edges_rich, hosts, is_test in res
        if is_test
    ])
    # Build the global hostname -> IP and IP -> hostname mappings
    _, _, hosts = merge_edges_hosts([
        (edges_base, edges_rich, hosts)
        for edges_base, edges_rich, hosts, is_test in res
    ])
    # Logs from DC1 are not included in the dataset but we can infer some of
    # its addresses
    hosts['DC1.systemia.com'] = set(['142.20.61.130', 'fe80::e964:163b:6cd:1937'])
    addr_to_host = {
        addr: host
        for host in hosts
        for addr in hosts[host]
    }
    # Logs from Web servers and DC2 are not included in the dataset but we can
    # infer one address for each using the Bro logs
    for i in range(10):
        addr_to_host[f'142.20.61.{157 + i}'] = (
            f'CentOSWeb{"0" if i < 9 else ""}{i + 1}.systemia.com'
        )
    addr_to_host['142.20.61.131'] = 'DC2.systemia.com'

    cols = ('src', 'dst', 'proto', 'port')
    train_base = pd.DataFrame(
        list(edges_train_base),
        columns=cols
    )
    test_base = pd.DataFrame(
        list(edges_test_base),
        columns=cols
    )
    # Replace IP addresses with hostnames when possible
    encode_addr = lambda a: addr_to_host[a] if a in addr_to_host else a
    # Replace infrequent ports with a placeholder
    counts = train_base['port'].value_counts().to_dict()
    freq_ports = set([port for port, cnt in counts.items() if cnt > 100])
    encode_port = lambda p: p if p in freq_ports else 'OTHER'
    for df in (train_base, test_base):
        for col in ('src', 'dst'):
            df[col] = df[col].apply(encode_addr)
        df['port'] = df['port'].apply(encode_port)
    # Filter out test edges present in the context graph or involving nodes or
    # relations absent from the context graph; add labels for test edges
    nodelist = set(train_base['src']).union(set(train_base['dst']))
    test_base = test_base[[
        src in nodelist and dst in nodelist
        for src, dst in zip(test_base['src'], test_base['dst'])
    ]]
    malicious_edges = set([
        (encode_addr(src), encode_addr(dst), proto, encode_port(port))
        for src, dst, proto, port in malicious_edges_optc()
    ])
    train_edges = set([t[1:] for t in train_base.itertuples()])
    test_base = test_base[[
        t[1:] not in train_edges
        for t in test_base.itertuples()
    ]]
    test_base['label'] = [
        int(t[1:] in malicious_edges)
        for t in test_base.itertuples()
    ]

    edges_base_train = sorted(list(set([
        (src, f'flow_{proto}_{port}', dst)
        for _, src, dst, proto, port in train_base.itertuples()
    ])))
    edges_base_test = sorted(list(set([
        (src, f'flow_{proto}_{port}', dst, label)
        for _, src, dst, proto, port, label in test_base.itertuples()
    ])))
    return edges_base_train, edges_base_test, list(edges_train_rich)


def include_ip(addr):
    '''
    Indicates whether an IP address should be included in the graph.
    Since we only consider internal flows, addresses outside the enterprise
    network are excluded.
    We also exclude broadcast addresses.

    Arguments
    ---------
    addr : str
        IP address.

    Returns
    -------
    is_included : bool
        True if the address is internal and not a broadcast address, False
        otherwise.

    '''

    if addr.endswith('.255'):
        return False
    if addr.startswith('10.'):
        return True
    if addr.startswith('142.'):
        return True
    if addr.startswith('fe80:'):
        return True
    return False


def malicious_edges_optc():
    '''
    Returns the edges corresponding to lateral movement in the OpTC dataset.
    Each edge is a 4-tuple (source IP, destination IP, protocol, destination
    port).
    We gathered this list by inspecting the logs using the ground truth
    description of the red team scenarios.

    Returns
    -------
    edges : set
        Set of 4-tuples corresponding to the lateral movement edges.

    '''

    edges = set()
    # Day 1 - SysClient0201 -> 142.20.56.204 (SMB)
    edges.add(('142.20.56.202', '142.20.56.204', 6, 445))
    # Day 1 - Ping sweep from SysClient0201
    for dst in range(1, 255):
        edges.add(('142.20.56.202', f'142.20.56.{dst}', 1, 0))
        if dst >= 202 and dst <= 226:
            edges.add(('142.20.56.202', f'142.20.56.{dst}', 1, 8))
    # Day 1 - SysClient0201 -> SysClient0402 (invoke_wmi)
    edges.add(('142.20.56.202', '142.20.57.147', 6, 135))
    edges.add(('142.20.56.202', '142.20.57.147', 6, 49666))
    # Day 1 - Ping sweep from SysClient0402
    for dst in range(1, 255):
        edges.add(('142.20.57.147', f'142.20.57.{dst}', 1, 0))
        if dst >= 146 and dst <= 170:
            edges.add(('142.20.57.147', f'142.20.57.{dst}', 1, 8))
    # Day 1 - SysClient0402 -> SysClient0660 (invoke_wmi)
    edges.add(('142.20.57.147', '142.20.58.149', 6, 135))
    edges.add(('142.20.57.147', '142.20.58.149', 6, 49665))
    # Day 1 - SysClient0660 -> DC1 (invoke_wmi)
    edges.add(('142.20.58.149', '142.20.61.130', 6, 135))
    edges.add(('142.20.58.149', '142.20.61.130', 6, 49154))
    # Day 1 - DC1 -> multiple targets (invoke_wmi)
    targets = [
        ('fe80::f1dc:7e94:ab12:84df', 49666), # SysClient104
        ('fe80::8126:e0f2:5abb:c53c', 49666), # SysClient170
        ('fe80::94ea:d285:f015:bd64', 49666), # SysClient205
        ('fe80::6ce4:10d0:b816:ebc9', 49666), # SysClient255
        ('fe80::913d:865c:689e:6868', 49665), # SysClient321
        ('fe80::f52d:48c0:4a3:e52e', 49665), # SysClient355
        ('fe80::4154:8214:d086:ff7a', 49666), # SysClient419
        ('fe80::8991:e7b3:541e:74df', 49666), # SysClient462
        ('fe80::8867:66e2:e2e3:51ae', 49665), # SysClient503
        ('fe80::643c:f8be:e91f:710c', 49665), # SysClient559
        ('fe80::c1c3:88d8:3334:e879', 49666), # SysClient609
        ('fe80::dd88:f526:fa4e:b6df', 49665), # SysClient771
        ('fe80::1847:f2d8:a76a:dba6', 49666), # SysClient874
        ('fe80::18e3:fd02:1733:43f5', 49666) # SysClient955
    ]
    for dst, port in targets:
        edges.update([
            ('fe80::e964:163b:6cd:1937', dst, 6, 135),
            ('fe80::e964:163b:6cd:1937', dst, 6, port)
        ])
    edges.update([
        ('fe80::e964:163b:6cd:1937', 'fe80::65f5:20ff:8fe0:2593', 6, 135),
        ('fe80::e964:163b:6cd:1937', 'fe80::65f5:20ff:8fe0:2593', 6, 49665),
        
    ])
    # Day 2 - SysClient0501 -> DC1 (unspecified method)
    edges.add(('142.20.57.246', '142.20.61.130', 6, 135))
    edges.add(('142.20.57.246', '142.20.61.130', 6, 49154))
    edges.add(('142.20.57.246', '142.20.61.130', 6, 49187))
    # Day 2 - DC1 -> SysClient0502 (failed invoke_wmi)
    edges.add(('fe80::e964:163b:6cd:1937', 'fe80::a876:7d33:6578:af1d', 6, 135))
    # Day 2 - DC1 -> SysClient0501 (unspecified method)
    edges.update([
        ('fe80::e964:163b:6cd:1937', 'fe80::65f5:20ff:8fe0:2593', 6, 135),
        ('fe80::e964:163b:6cd:1937', 'fe80::65f5:20ff:8fe0:2593', 6, 49665)
    ])
    # Day 2 - SysClient0501 -> SysClient0974 (RDP)
    edges.add(('142.20.57.246', '142.20.59.207', 6, 3389))
    edges.add(('142.20.57.246', '142.20.59.207', 17, 3389))
    # Day 2 - SysClient0974 -> SysClient0005 (RDP)
    edges.add(('142.20.59.207', '142.20.56.6', 6, 3389))
    edges.add(('142.20.59.207', '142.20.56.6', 17, 3389))
    # Day 2 - DC1 -> multiple hosts (invoke_wmi)
    targets = [
        ('fe80::fde6:4a91:3849:dad1', 49666), # SysClient010
        ('fe80::fd78:f214:f09:ae0d', 49666), # SysClient069
        ('fe80::98aa:e444:feca:de0b', 49666), # SysClient203
        ('fe80::4d9d:a6f8:c9fb:b1cc', 49666), # SysClient358
        ('fe80::d0b6:84c1:2619:741a', 49666), # SysClient618
        ('fe80::ccb1:78ce:8a54:258b', 49665) # SysClient851
    ]
    for dst, port in targets:
        edges.update([
            ('fe80::e964:163b:6cd:1937', dst, 6, 135),
            ('fe80::e964:163b:6cd:1937', dst, 6, port)
        ])
    return edges


def merge_edges_hosts(tuples):
    '''
    Merges the sets of edges and host -> address mappings generated by reading
    the log files of the OpTC dataset.

    Arguments
    ---------
    tuples : list
        List of (edges_base, edges_rich, hosts) tuples, where edges_base (resp.
        edges_rich) is a set of edges to include in the basic (resp. enriched)
        representation, and hosts is a host -> address mapping.

    Returns
    -------
    edges_base : set
        Merged set of edges to include in the basic representation.
    edges_rich : set
        Merged set of edges to include in the enriched representation.
    hosts : dict
        Merged host -> address mapping.

    '''

    edges_base = set().union(*[_edges for _edges, _, _ in tuples])
    edges_rich = set().union(*[_edges for _, _edges, _ in tuples])
    hosts = defaultdict(set)
    for _, _, _hosts in tuples:
        for host, ips in _hosts.items():
            hosts[host].update(ips)
    return edges_base, edges_rich, hosts


def process_auth(line, edges):
    '''
    Extracts edges from an authentication event from the OpTC dataset.

    Arguments
    ---------
    line : str
        Log line to process.
    edges : set
        Edge set to update with the new edges.

    '''

    evt = json.loads(line)
    new_edges = set()
    dst_dom, dst_usr = evt['properties']['user'].split('\\')
    host = evt['hostname']
    if 'requesting_user' in evt['properties']:
        src_usr, src_dom = [
            evt['properties'][f'requesting_{s}']
            for s in ('user', 'domain')
        ]
    else:
        src_usr, src_dom = None, None
    if evt['action'] == 'LOGIN':
        # Login: we add user-host and source user-destination user edges
        for usr, dom in zip([src_usr, dst_usr], [src_dom, dst_dom]):
            if usr is not None:
                new_edges.add((usr, 'has_domain', dom))
                new_edges.add((usr, 'login_uh', host))
        new_edges.add((src_usr, 'login_uu', dst_usr))
    elif evt['action'] == 'GRANT':
        # Privileged login: we add one user-host edge per privilege
        privs = evt['properties']['privileges'].replace('\t', '').split('\n')
        new_edges.add((dst_usr, 'has_domain', dst_dom))
        for priv in privs:
            new_edges.add((dst_usr, f'has_priv_{priv}', host))
    elif evt['action'] == 'REMOTE':
        # Remote login: we add one user-host edge
        new_edges.add((dst_usr, 'has_domain', dst_dom))
        new_edges.add((dst_usr, 'remote_uh', host))
    elif evt['action'] == 'INTERACTIVE':
        # Interactive login: we add user-host and source user-destination user
        # edges
        for usr, dom in zip([src_usr, dst_usr], [src_dom, dst_dom]):
            new_edges.add((usr, 'has_domain', dom))
            new_edges.add((usr, 'interactive_uh', host))
        new_edges.add((src_usr, 'interactive_uu', dst_usr))
    edges.update(new_edges)


def process_dir(dirpath, filenames):
    '''
    Extracts edges and a host -> address mapping from a list of raw log files
    within a given directory.

    Arguments
    ---------
    dirpath : str
        Path to the directory containing the log files.
    filenames : list
        Names of the log files.

    Returns
    -------
    edges_base : set
        Set of edges to include in the basic representation.
    edges_rich : set
        Set of edges to include in the enriched representation.
    hosts : dict
        Host -> address mapping.
    is_test : bool
        True if the logs come from the evaluation period, False otherwise.

    '''

    res = [
        process_evts(os.path.join(dirpath, fname))
        for fname in filenames
    ]
    edges_base, edges_rich, hosts = merge_edges_hosts(res)
    is_test = 'evaluation/' in dirpath
    return edges_base, edges_rich, hosts, is_test


def process_evts(filepath):
    '''
    Extracts edges and a host -> address mapping from a raw log file
    (GZIP-compressed JSON format).

    Arguments
    ---------
    filepath : str
        Path to the log file.

    Returns
    -------
    edges_base : set
        Set of edges to include in the basic representation.
    edges_rich : set
        Set of edges to include in the enriched representation.
    hosts : dict
        Host -> address mapping.

    '''

    edges_base = set()
    edges_rich = set()
    hosts = defaultdict(set)
    with gzip.open(filepath, 'rt') as file:
        for line in file:
            if line[1:17] == '"action":"START"' and '"object":"FLOW"' in line:
                process_flow(line, edges_base, hosts)
            elif '"object":"USER_SESSION"' in line:
                process_auth(line, edges_rich)
            else:
                continue
    return edges_base, edges_rich, hosts


def process_flow(line, edges, hosts):
    '''
    Extracts edges and host -> address correspondence from a flow start
    event from the OpTC dataset.

    Arguments
    ---------
    line : str
        Log line to process.
    edges : set
        Edge set to update with the new edges.
    hosts : dict
        Host -> address mapping to update.

    '''

    evt = json.loads(line)
    host, src, dst, proto, port, direction = (
        evt['hostname'],
        evt['properties']['src_ip'],
        evt['properties']['dest_ip'],
        int(evt['properties']['l4protocol']),
        int(evt['properties']['dest_port']),
        evt['properties']['direction']
    )
    if not include_ip(src) or not include_ip(dst):
        return
    edges.add((src, dst, proto, port))
    if direction == 'outbound':
        hosts[host].add(src)
    else:
        hosts[host].add(dst)


############################
# General helper functions #
############################

def write_dataset(
        edges_base_train,
        edges_base_test,
        edges_rich,
        output_dir,
        dataset_name,
        random_state=0
    ):
    '''
    Converts edge lists into train, validation, test, and detection
    sets for both the basic and extended representation of the context graph,
    and saves them to disk.

    Arguments
    ---------
    edges_base_train : list
        Adjacency list for the basic representation of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    edges_base_test : list
        Adjacency list for the test edges.
        Each element of the list is a tuple with elements
        (source, relation, destination, label).
    edges_rich : list
        Adjacency list for the additional edges in the enriched representation
        of the context graph.
        Each element of the list is a tuple with elements
        (source, relation, destination).
    output_dir : str
        Path to the directory where the dataset files should be written.
    dataset_name : str
        Name of the dataset (used to name the subdirectories).
    random_state : int, default=0
        Seed for the RNG (used to split the set of train edges into the train
        and validation sets).

    '''
    
    nodes = sorted(list(set().union(
        [h for h, _, _ in edges_base_train],
        [t for _, _, t in edges_base_train]
    )))
    edges_train, edges_valid = train_test_split(
        edges_base_train,
        test_size=.01,
        random_state=random_state
    )
    edges_train_rich = edges_train + edges_rich
    edges_test = [
        (src, rel, dst)
        for src, rel, dst, lab in edges_base_test
        if lab == 0
    ]
    for train, data_dir in zip(
            (edges_train, edges_train_rich),
            (dataset_name, dataset_name + '-rich')
        ):
        dirpath = os.path.join(output_dir, data_dir, 'raw')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        for edges, file in zip(
                (train, edges_valid, edges_test, edges_base_test),
                ('train', 'valid', 'test', 'detect')
            ):
            write_file(
                os.path.join(dirpath, f'{file}.txt'),
                edges
            )
        with open(os.path.join(dirpath, 'nodes.txt'), 'w') as out:
            out.write('\n'.join(nodes) + '\n')


def write_file(filepath, edges):
    '''
    Writes a list of edges into a file.

    Arguments
    ---------
    filepath : str
        Path to the output file.
    edges : list
        List of (source, relation, destination[, label]) tuples representing
        the edges.

    '''

    with open(filepath, 'w') as out:
        for edge in edges:
            out.write(
                ','.join([str(x) for x in edge]) + '\n'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lanl_dir',
        default=None,
        help=(
            'Path to the directory containing the auth.txt.gz and '
            'redteam.txt.gz files of the LANL dataset. '
            'If None, the LANL dataset is not extracted.'
        )
    )
    parser.add_argument(
        '--optc_dir',
        default=None,
        help=(
            'Path to the directory containing the OpTC dataset. '
            'This should be the ecar/ directory, with subdirectories benign/, '
            'short/, and evaluation/ containing the original GZIP-compressed '
            'JSON log files.'
        )
    )
    parser.add_argument(
        '--output_dir',
        default='../datasets/',
        help='Path to the directory where the dataset files will be written.'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=-1,
        help=(
            'Number of parallel reading jobs to use when preprocessing the '
            'OpTC dataset.'
        )
    )
    args = parser.parse_args()

    if args.optc_dir is not None:
        edges_base_train, edges_base_test, edges_rich = extract_optc_dataset(
            args.optc_dir,
            n_jobs=args.jobs
        )
        write_dataset(
            edges_base_train,
            edges_base_test,
            edges_rich,
            args.output_dir,
            'optc'
        )
    if args.lanl_dir is not None:
        edges_base_train, edges_base_test, edges_rich = extract_lanl_dataset(
            args.lanl_dir
        )
        write_dataset(
            edges_base_train,
            edges_base_test,
            edges_rich,
            args.output_dir,
            'lanl'
        )