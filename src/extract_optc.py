import argparse
import datetime as dt
import gzip
import json
import os

from collections import defaultdict

import pandas as pd

from joblib import Parallel, delayed




WIN_LEN = 360
START_DATE = dt.datetime(2019, 9, 16, 19, 0, 0)
TEST_DATE = dt.datetime(2019, 9, 23, 0, 0, 0)
TOP_K_PROTO_PORT = 20


def extract_optc_dataset(base_dir, rt_path, n_jobs=-1):
    '''
    Preprocesses the OpTC dataset and returns a dataframe containing the list
    of edges as well as the name -> type map for the nodes.
    The input directory should be the ecar/ directory, with subdirectories
    benign/, short/ and evaluation/ containing the original GZIP-compressed
    JSON log files.

    Arguments
    ---------
    base_dir : str
        Path to the ecar/ directory containing the input log files.
    rt_path : str
        Path to the optc_redteam.csv file.
    n_jobs : int, default=-1
        Number of parallel jobs to create (if -1, it is set to the number of
        available CPU cores).

    Returns
    -------
    edges : pd.DataFrame
        Edges in each time window of the OpTC dataset.
        The dataframe has the following columns: time, src, dst, rel, label,
        auxiliary, corresponding to the time window, source node, destination
        node, edge type, label (1 for malicious and 0 otherwise), and whether
        the edge should be scored (0 if it should be scored and 1 otherwise),
        respectively.
    nodes : dict
        Name -> type map for the nodes.

    '''

    # Read the raw data
    rt_evts = pd.read_csv(rt_path)
    redteam = set(rt_evts['id'])
    res = Parallel(n_jobs=n_jobs)(
        delayed(process_dir)(dirpath, filenames, redteam)
        for dirpath, _, filenames in os.walk(base_dir)
    )
    # Merge the results of the reading jobs
    flows, auths, hosts, nodes = merge_edges_hosts_nodes(res)
    # Logs from DC1 are not included in the dataset but we can infer some of
    # its addresses
    hosts['DC1.systemia.com'] = set(['142.20.61.130', 'fe80::e964:163b:6cd:1937'])
    nodes['DC1.systemia.com'] = '_host'
    addr_to_host = {
        addr: host
        for host in hosts
        for addr in hosts[host]
    }
    # Logs from Web servers and DC2 are not included in the dataset but we can
    # infer one address for each using the Bro logs
    for i in range(10):
        hostname = f'CentOSWeb{"0" if i < 9 else ""}{i + 1}.systemia.com'
        addr_to_host[f'142.20.61.{157 + i}'] = hostname
        nodes[hostname] = '_host'
    addr_to_host['142.20.61.131'] = 'DC2.systemia.com'
    nodes['DC2.systemia.com'] = '_host'

    cols = ('time', 'src', 'dst', 'rel', 'label')
    flows = pd.DataFrame(
        list(flows),
        columns=cols
    )
    # Replace infrequent proto-port pairs with a placeholder
    # We count the occurrences in the training period only to avoid data
    # snooping
    test_time = int((TEST_DATE - START_DATE).total_seconds()) // WIN_LEN
    flows_train = flows[flows['time'] < test_time]
    counts = flows_train['rel'].value_counts().to_dict()
    freq_rels = set(sorted(
        list(counts.keys()), key=lambda x: -counts[x]
    )[:TOP_K_PROTO_PORT])
    encode_rel = lambda x: x if x in freq_rels else 'OTHER'
    flows['rel'] = flows['rel'].apply(encode_rel)
    # Replace IP addresses with hostnames when possible
    encode_addr = lambda a: addr_to_host[a] if a in addr_to_host else a
    for col in ('src', 'dst'):
        flows[col] = flows[col].apply(encode_addr)
    flows.drop_duplicates(inplace=True)
    flows['auxiliary'] = 0

    # Add authentication-related edges
    cols = ('time', 'src', 'rel', 'dst')
    auths = pd.DataFrame(
        list(auths),
        columns=cols
    )
    auths['auxiliary'] = 1
    auths['label'] = 0
    cols = ['time', 'src', 'rel', 'dst', 'label', 'auxiliary']
    edges = pd.concat([
        flows.loc[:, cols],
        auths.loc[:, cols]
    ]).sort_values('time').reset_index()
    return edges, nodes


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


def include_user(user):
    '''
    Indicates whether a username should be included in the graph.
    We exclude accounts related to Desktop Window Manager and User Mode Driver
    Framework system services, as they are generated on the fly.

    Arguments
    ---------
    user : str
        User name.

    Returns
    -------
    is_included : bool
        False if the user name is related to Desktop Window Manager and User
        Mode Driver Framework system services, True otherwise.

    '''

    if user.startswith('UMFD-'):
        return False
    if user.startswith('DWM-'):
        return False
    return True


def make_timestamp(ts_str):
    '''
    Converts an ISO-formatted timestamp into the index of the corresponding
    time window.

    Arguments
    ---------
    ts_str : str
        ISO-formatted timestamp.

    Returns
    -------
    window : int
        Index of the corresponding time window.

    '''

    date = dt.datetime.fromisoformat(ts_str[:19])
    delta = date - START_DATE
    return int(delta.total_seconds()) // WIN_LEN


def merge_edges_hosts_nodes(tuples):
    '''
    Merges the sets of edges, host -> address and node -> type mappings
    generated by reading the log files in the OpTC dataset.

    Arguments
    ---------
    tuples : list
        List of (flow_edges, auth_edges, hosts, node_types) tuples, where
        flow_edges (resp. auth_edges) is a set of network flow (resp.
        authentication) edges, hosts is a host -> address mapping, and
        node_types is a name -> type mapping for the nodes.

    Returns
    -------
    flows : set
        Merged set of flow-related edges.
    auths : set
        Merged set of authentication-related edges.
    hosts : dict
        Merged host -> address mapping.
    nodes : dict
        Merged node name -> type mapping.

    '''

    flows = set().union(*[_flows for _flows, _, _, _ in tuples])
    auths = set().union(*[_auths for _, _auths, _, _ in tuples])
    hosts = defaultdict(set)
    nodes = dict()
    for _, _, _hosts, _nodes in tuples:
        for host, ips in _hosts.items():
            hosts[host].update(ips)
        nodes.update(_nodes)
    return flows, auths, hosts, nodes


def process_auth(line, edges, nodes):
    '''
    Extracts edges from an authentication event from the OpTC dataset.

    Arguments
    ---------
    line : str
        Log line to process.
    edges : set
        Edge set to update with the new edges.
    nodes : dict
        Node name -> type mapping to update with the new nodes.

    '''

    evt = json.loads(line)
    new_edges = set()
    time = make_timestamp(evt['timestamp'])
    dst_dom, dst_usr = evt['properties']['user'].split('\\')
    if not include_user(dst_usr):
        return
    new_edges.add((time, dst_usr, 'has_domain', dst_dom))
    nodes.update({
        dst_usr: user_type(dst_usr),
        dst_dom: '_domain'
    })
    host = evt['hostname']
    nodes[host] = '_host'
    if 'requesting_user' in evt['properties']:
        # The source user is identified in the event
        src_usr, src_dom = [
            evt['properties'][f'requesting_{s}']
            for s in ('user', 'domain')
        ]
        if not include_user(src_usr):
            return
        if src_usr is not None and src_dom is not None:
            new_edges.add((time, src_usr, 'has_domain', src_dom))
        nodes.update({
            src_usr: user_type(src_usr),
            src_dom: '_domain'
        })
    else:
        # No source user identified in the event
        src_usr, src_dom = None, None
    if evt['action'] == 'LOGIN':
        # Login: we add user-host and source user-destination user edges
        for usr, dom in zip([src_usr, dst_usr], [src_dom, dst_dom]):
            if usr is not None:
                new_edges.add((time, usr, 'login_uh', host))
        if src_usr is not None and dst_usr is not None and src_usr != dst_usr:
            new_edges.add((time, src_usr, 'login_uu', dst_usr))
    elif evt['action'] == 'GRANT':
        # Privileged login: we add one user-host edge per privilege
        privs = evt['properties']['privileges'].replace('\t', '').split('\n')
        for priv in privs:
            new_edges.add((time, dst_usr, f'has_priv_{priv}', host))
    elif evt['action'] == 'REMOTE':
        # Remote login: we add one user-host edge
        new_edges.add((time, dst_usr, 'remote_uh', host))
    elif evt['action'] == 'INTERACTIVE':
        # Interactive login: we add user-host and source user-destination user
        # edges
        for usr, dom in zip([src_usr, dst_usr], [src_dom, dst_dom]):
            if usr is not None:
                new_edges.add((time, usr, 'interactive_uh', host))
        if src_usr is not None and dst_usr is not None and src_usr != dst_usr:
            new_edges.add((time, src_usr, 'interactive_uu', dst_usr))
    edges.update(new_edges)


def process_dir(dirpath, filenames, redteam):
    '''
    Extracts network flow edges, authentication edges, a host -> address
    mapping, and a node name -> type mapping from a list of raw log files
    within a given directory.

    Arguments
    ---------
    dirpath : str
        Path to the directory containing the log files.
    filenames : list
        Names of the log files.
    redteam : set
        Set of IDs corresponding to lateral movement events.

    Returns
    -------
    flows : set
        Set of network flow-related edges.
    auths : set
        Set of authentication-related edges.
    hosts : dict
        Host -> address mapping.
    nodes : dict
        Node name -> type mapping.

    '''

    res = [
        process_evts(os.path.join(dirpath, fname), redteam)
        for fname in filenames
    ]
    return merge_edges_hosts_nodes(res)


def process_evts(filepath, redteam):
    '''
    Extracts network flow edges, authentication edges, a host -> address
    mapping, and a node name -> type mapping from a raw log file
    (GZIP-compressed JSON format).

    Arguments
    ---------
    filepath : str
        Path to the log file.
    redteam : set
        Set of IDs corresponding to lateral movement events.

    Returns
    -------
    flows : set
        Set of network flow-related edges.
    auths : set
        Set of authentication-related edges.
    hosts : dict
        Host -> address mapping.
    nodes : dict
        Node name -> type mapping.

    '''

    flows = set()
    auths = set()
    nodes = dict()
    hosts = defaultdict(set)
    with gzip.open(filepath, 'rt') as file:
        for line in file:
            if line[1:17] == '"action":"START"' and '"object":"FLOW"' in line:
                process_flow(line, flows, hosts, nodes, redteam)
            elif '"object":"USER_SESSION"' in line:
                process_auth(line, auths, nodes)
            else:
                continue
    return flows, auths, hosts, nodes


def process_flow(line, edges, hosts, nodes, redteam):
    '''
    Extracts edges, host -> address correspondence, and node name -> type
    correspondence from a flow start event in the OpTC dataset.

    Arguments
    ---------
    line : str
        Log line to process.
    edges : set
        Edge set to update with the new edges.
    hosts : dict
        Host -> address mapping to update.
    nodes : dict
        Node name -> type mapping to update.
    redteam : set
        Set of IDs corresponding to lateral movement events.

    '''

    evt = json.loads(line)
    time, evt_id, host, src, dst, proto, port, direction = (
        make_timestamp(evt['timestamp']),
        evt['id'],
        evt['hostname'],
        evt['properties']['src_ip'],
        evt['properties']['dest_ip'],
        int(evt['properties']['l4protocol']),
        int(evt['properties']['dest_port']),
        evt['properties']['direction']
    )
    if not include_ip(src) or not include_ip(dst):
        return
    label = int(evt_id in redteam)
    edges.add((time, src, dst, f'{proto}_{port}', label))
    nodes.update({
        src: '_host',
        dst: '_host'
    })
    if direction == 'outbound':
        hosts[host].add(src)
    else:
        hosts[host].add(dst)


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

    if user.endswith('$'):
        return '_computer_account'
    if user in (
        'Local service', 'Local system', 'Network service',
        'ANONYMOUS LOGON'
    ):
        return '_builtin_account'
    else:
        return '_user_account'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir',
        help='Path to the ecar/ directory containing the input log files.'
    )
    parser.add_argument(
        'redteam',
        help='Path to the optc_redteam.csv file.'
    )
    parser.add_argument(
        '--output_dir',
        default='../datasets/optc/',
        help='Path to the output directory.'
    )
    args = parser.parse_args()

    edges, nodes = extract_optc_dataset(
        args.input_dir, args.redteam
    )
    if not os.path.exists(os.path.join(args.output_dir, 'edges')):
        os.makedirs(os.path.join(args.output_dir, 'edges'))
    # Split the dataframe into one file per time window
    gby = edges.groupby(['time']).groups
    for window, idx in gby.items():
        sub = edges.loc[idx, ['src', 'rel', 'dst', 'auxiliary', 'label']]
        fp = os.path.join(args.output_dir, 'edges', f'{window}.csv')
        sub.to_csv(fp, index=False)
    # Write the node name -> type mapping
    node_df = pd.DataFrame(
        [[node, node_type] for node, node_type in nodes.items()],
        columns=('name', 'type')
    )
    node_df.to_csv(os.path.join(args.output_dir, 'nodes.csv'), index=False)
