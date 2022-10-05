from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from timelapsetracking.util import report_run_time

logger = logging.getLogger(__name__)

def csv_to_nodes(
    df: pd.DataFrame, 
    metatable: Dict[str, List[str]],
    field_shape: Tuple[int]
    ):

    nodes = []
    for idx, row in tqdm(df.iterrows()):
        node = {}

        node['time_index'] = int(row[metatable['time_index']])
        node['index_sequence'] = int(row[metatable['time_index']])
        node['volume'] = row[metatable['volume']]
        node['label_img'] = int(row[metatable['label_img']])
        node['is_pair'] = False
        node['has_pair'] = False

        edge_distance = np.amax(field_shape)
        for idx_d, dim in enumerate('zyx'[-len(metatable['zyx_cols']):]):
            node[f'centroid_{dim}'] = row[metatable['zyx_cols'][idx_d]]
            if dim != 'z':
                edge_distance = np.amin([edge_distance, 
                                        row[metatable['zyx_cols'][idx_d]], 
                                        field_shape[idx_d]-row[metatable['zyx_cols'][idx_d]]])

        if len(metatable['zyx_cols']) == 2:
            node[f'centroid_z'] = 0

        node['edge_distance'] = edge_distance

        if metatable['edge_cell'] is not None:
            node['edge_cell'] = bool(metatable['edge_cell'])
        else:
            node['edge_cell'] = False

        node['has_pair'] = False

        nodes.append(node)

    nodes = pd.DataFrame(nodes)

    return nodes