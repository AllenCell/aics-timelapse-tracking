"""Functions to convert objects in segmentation images to node representation.

"""

from multiprocessing import Pool
from typing import Dict, List, Optional
import os

import numpy as np
import pandas as pd
import tifffile

from timelapsetracking.util import images_from_dir
from timelapsetracking.util import report_run_time


def img_to_nodes(
        img: np.ndarray, meta: Optional[Dict] = None
) -> List[Dict]:
    """Converts image of object labels to graph nodes.

    For each object, creates a node with attributes:
    centroid_x, centroid_y, centroid_z, volume, label, metadata

    Parameters
    ----------
    img
        Image of labeled objects.
    meta
        Optional attributes to be applied to all objects found in input image.

    Returns
    -------
    List[Dict]
        List of dictionaries representing each object.

    """
    if img.ndim != 3:
        raise ValueError('Images must be 3d')
    if not np.issubdtype(img.dtype, np.unsignedinteger):
        raise TypeError('Image must be np.unsignedinteger compatible')
    if meta is None:
        meta = {}
    labels = np.unique(img)
    if labels[0] == 0:
        labels = labels[1:]
    nodes = []
    for label in labels:
        node = {}
        mask = img == label
        coords = np.where(mask)
        node['volume'] = mask.sum()
        node['label_img'] = label
        for idx_d, dim in enumerate('zyx'):
            node[f'centroid_{dim}'] = coords[idx_d].mean()
        node.update(meta)
        nodes.append(node)
    return nodes


def _img_to_nodes_wrapper(meta: Dict) -> List[Dict]:
    """Wrapper for 'map' method of multiprocessing.Pool."""
    print('Processing:', meta['path_tif'])
    return img_to_nodes(tifffile.imread(meta['path_tif']), meta=meta)


@report_run_time
def dir_to_nodes(
        path_img_dir: str,
        path_save_csv: str,
        num_processes: int = 8,
) -> None:
    """Converts directory of label images to graph nodes.

    Parameters
    ----------
    path_img_dir
        Directory of images.
    path_save_csv
        Save path.
    num_processes
        Maximum number of processes used to perform conversion.

    Returns
    -------
    None

    """
    paths_img = images_from_dir(path_img_dir)
    metas = [
        {'index_sequence': idx_s, 'path_tif': path} for idx_s, path in enumerate(paths_img)
    ]
    with Pool(min(num_processes, os.cpu_count())) as pool:
        nodes_per_img = pool.map(_img_to_nodes_wrapper, metas)
    nodes = []
    for per_img in nodes_per_img:
        nodes.extend(per_img)
    dirname = os.path.dirname(path_save_csv)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('Created:', dirname)
    pd.DataFrame(nodes).rename_axis('node_id', axis=0).to_csv(path_save_csv)
    print('Saved:', path_save_csv)
