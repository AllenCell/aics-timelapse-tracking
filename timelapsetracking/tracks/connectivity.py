from typing import Dict, List

import numpy as np
import pandas as pd


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
        in_list: Dict[int, List[int]], out_list: Dict[int, List[int]]
) -> Dict[int, int]:
    if in_list.keys() != out_list.keys():
        raise ValueError('Input lists must have same keys')
    adj_list = {k: (in_list[k] + out_list[k]) for k in in_list.keys()}
    explorer = _Explorer(adj_list)
    wcc = 0
    for node in adj_list.keys():
        if node in explorer.visited:
            continue
        explorer.explore_assign_wcc(node, wcc)
        wcc += 1
    return dict(explorer.wcc_list)


def add_track_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Labels nodes in time-lapse graph with track ids.

    Parameters
    ----------
    df
        Input DataFrame representing time-lapse graph.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added 'track_id' column.

    """
    if 'in_list' not in df.columns or 'out_list' not in df.columns:
        raise ValueError('"in_list", "out_list" columns expect in df')
    in_list = {
        k: ([] if np.isnan(v) else [int(v)])
        for k, v in df['in_list'].to_dict().items()
    }
    out_list = {
        k: ([] if np.isnan(v) else [int(v)])
        for k, v in df['out_list'].to_dict().items()
    }
    wcc_map = _calc_weakly_connected_components(in_list, out_list)
    return df.assign(track_id=pd.Series(wcc_map))
