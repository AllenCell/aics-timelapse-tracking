"""Functions to convert objects in segmentation images to node representation.

"""

from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os

import numpy as np
import pandas as pd
import tifffile

from skimage import measure as skmeasure 

from timelapsetracking.util import images_from_dir
from timelapsetracking.util import report_run_time


logger = logging.getLogger(__name__)


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
    if img.ndim not in [2, 3]:
        raise ValueError('Images must be 2d or 3d')
    if not np.issubdtype(img.dtype, np.unsignedinteger):
        raise TypeError('Image must be np.unsignedinteger compatible')
    if meta is None:
        meta = {}

    labels = np.unique(img)
    if labels[0] == 0:
        labels = labels[1:]
    nodes = []
    for label in labels:
        print('Label [{0}]'.format(label))
        node = {}
        mask = img == label
        coords = np.where(mask)
        node['volume'] = mask.sum()
        node['label_img'] = label
        # Iterate over 'yx' for 2d, 'zyx' for 3d
        for idx_d, dim in enumerate('zyx'[-img.ndim:]):
            node[f'centroid_{dim}'] = coords[idx_d].mean()
        node.update(meta)
        nodes.append(node)
    return nodes


def _img_to_nodes_wrapper(meta: Dict) -> List[Dict]:
    """Wrapper for 'map' method of multiprocessing.Pool."""
    logger.info(f'Processing: {meta["path_tif"]}')
    return img_to_nodes(tifffile.imread(meta['path_tif']), meta=meta)


@report_run_time
def dir_to_nodes(
        path_img_dir: Path,
        num_processes: int = 8,
) -> pd.DataFrame:
    """Converts directory of label images to graph nodes.

    Parameters
    ----------
    path_img_dir
        Directory of images.
    num_processes
        Maximum number of processes used to perform conversion.

    Returns
    -------
    pd.DataFrame
        Generated nodes as a DataFrame.

    """
    paths_img = images_from_dir(path_img_dir)
    metas = [
        {'index_sequence': idx_s, 'path_tif': path}
        for idx_s, path in enumerate(paths_img)
    ]
    with Pool(min(num_processes, os.cpu_count())) as pool:
        nodes_per_img = pool.map(_img_to_nodes_wrapper, metas)
    nodes = []
    for per_img in nodes_per_img:
        nodes.extend(per_img)
    return pd.DataFrame(nodes).rename_axis('node_id', axis=0)
