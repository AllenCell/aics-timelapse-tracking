from typing import Dict, List, Union
import ast

import pandas as pd


NodeType = Union[int, str]
NodetoEdgesMap = Dict[NodeType, List[NodeType]]


def _in_from_out(out_lists: NodetoEdgesMap):
    """Generates dict of in_lists from dict of out_lists."""
    in_lists = {k: [] for k in out_lists}
    for parent, children in out_lists.items():
        for child in children:
            in_lists[child].append(parent)
    return in_lists


def graph_valid(out_lists: NodetoEdgesMap) -> bool:
    """Verifies that input graph is valid.

    Acyclic. For each node, at most 1 in edge and at most 2 out edges.

    """

    def _valid_helper(v: NodeType) -> bool:
        """Validates input node and its children."""
        being_explored[v] = True
        # Check for too many in or out edges
        if len(in_lists[v]) > 1 or len(out_lists[v]) > 2:
            print('in-out error')
            print(v)
            print(str(in_lists[v]) + ' ' + str(out_lists[v]))
            return False
        for child in out_lists[v]:
            # print(child)
            if child not in being_explored:
                if not _valid_helper(child):
                    print('child not valid helper')
                    print(child)
                    return False
            elif being_explored[child]:  # Cycle detected
                # print('being explored error')
                return False
        being_explored[v] = False
        return True

    in_lists = _in_from_out(out_lists)
    # being_explored maps from node to whether node is currently being explored
    #   no key: node has not been visited
    #   value == True: node is being explored
    #   value == False: node and its children have been explored
    being_explored = {}
    for v in out_lists:
        if v not in being_explored:
            if not _valid_helper(v):
                print(f'node {v} failed')
                print(in_lists[v])
                print(out_lists[v])
                return False
    return True


class _Explorer:
    def __init__(self, adj_list: Dict[int, List[int]]):
        self.adj_list = adj_list
        self.wcc_list = {}
        self.visited = self.wcc_list

    def explore_assign_wcc(self, node: int, wcc: int):
        if node in self.wcc_list:
            return
        self.wcc_list[node] = wcc
        for neighbor in self.adj_list[node]:
            self.explore_assign_wcc(neighbor, wcc)


def _calc_weakly_connected_components(
        out_lists: Dict[int, List[int]]
) -> Dict[int, int]:
    """Maps nodes to unique weakly-connected component (WCC) ids. In the
    context of cell tracks, each wcc identifies tracks that originate from the
    same ancestor node.

    Parameters
    ----------
    out_lists
        'out' edges for each node

    Returns
    -------
    Dict[int, int]
        Mapping from nodes to WCC ids.

    """
    in_lists = _in_from_out(out_lists)
    adj_list = {k: (in_lists[k] + out_lists[k]) for k in in_lists.keys()}
    explorer = _Explorer(adj_list)
    wcc = 0
    for node in adj_list.keys():
        if node in explorer.visited:
            continue
        explorer.explore_assign_wcc(node, wcc)
        wcc += 1
    return explorer.wcc_list


def _calc_track_ids(
        out_lists: Dict[int, List[int]]
) -> Dict[int, int]:
    """Returns mapping from node to track_id.

    Assumes all trees in the graph are single-sourced and acyclic.

    """
    in_lists = _in_from_out(out_lists)
    track_ids = {}
    track_id = 0
    for source in out_lists.keys():
        if len(in_lists[source]) != 0:
            continue
        stack = [(source, track_id)]
        track_id += 1
        while len(stack) > 0:
            v, val = stack.pop()
            if v in track_ids:
                raise ValueError('Invalid input graph')
            track_ids[v] = val
            children = out_lists[v]
            if len(children) == 1:
                stack.append((children[0], val))
            elif len(children) > 1:
                for child in children:
                    stack.append((child, track_id))
                    track_id += 1
    return track_ids


def add_connectivity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Labels nodes in time-lapse graph with lineage and track ids.

    Parameters
    ----------
    df
        Input DataFrame representing time-lapse graph.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added 'lineage_id' and 'track_id' columns.

    """
    if 'out_list' not in df.columns:
        raise ValueError('"out_list" column expected in df')
    out_lists = {
        k: ast.literal_eval(v) if isinstance(v, str) else v
        for k, v in df['out_list'].to_dict().items()
    }
    if not graph_valid(out_lists):
        raise ValueError('Bad input graph')
    lineage_ids = _calc_weakly_connected_components(out_lists)
    track_ids = _calc_track_ids(out_lists)
    return df.assign(
        lineage_id=pd.Series(lineage_ids),
        track_id=pd.Series(track_ids),
    )
