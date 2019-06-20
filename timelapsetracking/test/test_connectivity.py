from collections import defaultdict
from typing import List

import pandas as pd

from timelapsetracking.tracks import connectivity


def _get_test_out_lists():
    """Returns adjacency list for test graph."""
    # 4 weakly-connected subgraphs
    return {
        'a': [],
        'b': ['c'],
        'c': ['d'],
        'd': [],
        'e': ['f', 'g'],
        'f': [],
        'g': ['h'],
        'h': [],
        'i': ['j'],
        'j': ['k'],
        'k': ['l', 'n'],
        'l': ['m'],
        'm': [],
        'n': ['o', 'p'],
        'o': [],
        'p': ['q'],
        'q': [],
    }


def _get_test_track_id_groups():
    return [
        set(['a']),
        set(['b', 'c', 'd']),
        set(['e']),
        set(['f']),
        set(['g', 'h']),
        set(['i', 'j', 'k']),
        set(['l', 'm']),
        set(['n']),
        set(['o']),
        set(['q', 'p'])
    ]


def _get_test_lineage_id_groups():
    return [
        set(['a']),
        set(['b', 'c', 'd']),
        set(['e', 'f', 'g', 'h']),
        set(['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']),
    ]


def _labels_to_groups(
        nodes_to_labels: connectivity.NodetoEdgesMap
) -> List[set]:
    """Converts node-to-label mapping into groups of nodes with same label."""
    groups = defaultdict(set)
    for node, label in nodes_to_labels.items():
        groups[label].add(node)
    return list(groups.values())


def _label_groups_equal(groups_a: List[set], groups_b: List[set]) -> bool:
    """Returns True of input group lists are equal.

    Order of groups in each group list does not matter.

    """
    if len(groups_a) != len(groups_b):
        return False
    indices = set()
    for group in groups_a:
        try:
            idx = groups_b.index(group)
        except ValueError:
            return False
        indices.add(idx)
    return len(indices) == len(groups_b)


def test_graph_valid():
    """Tests _calc_track_ids with bad inputs."""
    assert connectivity.graph_valid({})
    out_lists = _get_test_out_lists()
    assert connectivity.graph_valid(out_lists)

    # Test two parents case. Give "d" two parents.
    out_lists = _get_test_out_lists()
    out_lists['a'].append('d')
    assert not connectivity.graph_valid(out_lists)

    # Test cycle case. Create cycle e -> g -> h -> e.
    out_lists = _get_test_out_lists()
    out_lists['h'].append('e')
    assert not connectivity.graph_valid(out_lists)


def test_add_connectivity_labels():
    """Tests add_connectivity_labels function."""
    out_lists = _get_test_out_lists()
    entries = [
        {'node_id': k, 'out_list': str(v)} for k, v in out_lists.items()
    ]
    df = pd.DataFrame(entries).set_index('node_id')
    df = connectivity.add_connectivity_labels(df)
    assert 'lineage_id' in df.columns and 'track_id' in df.columns
    lineage_groups_got = _labels_to_groups(df['lineage_id'].to_dict())
    lineage_groups_exp = _get_test_lineage_id_groups()
    assert _label_groups_equal(lineage_groups_got, lineage_groups_exp)
    track_groups_got = _labels_to_groups(df['track_id'].to_dict())
    track_groups_exp = _get_test_track_id_groups()
    assert _label_groups_equal(track_groups_got, track_groups_exp)
