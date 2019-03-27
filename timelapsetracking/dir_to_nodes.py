"""Functions to convert objects in segmentation images to node representation.

"""

from typing import List, Optional
import os

import numpy as np
import pandas as pd
import tifffile

from timelapsetracking.util import images_from_dir
from timelapsetracking.util import report_run_time


def img_to_nodes(img: np.ndarray, meta: Optional[dict] = None) -> List[dict]:
    """Converts image of object labels to graph nodes.

    centroid_x, centroid_y, centroid_z, volume, label, metadata

    Parameters
    ----------
    img
        Image of labeled objects.

    Returns
    -------
    List[dict]
        List of dictionaries representing each object.

    """
    assert img.ndim == 3
    assert np.issubdtype(img.dtype, np.unsignedinteger)
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


@report_run_time
def dir_to_nodes(path_dir: str, path_save_csv: str) -> None:
    """Converts directory of label images to graph nodes.

    Parameters
    ----------
    path_dir
        Directory of images.
    path_save_csv
        Save path.

    Returns
    -------
    None

    """
    paths_img = images_from_dir(path_dir)
    nodes = []
    for idx_s, path_img in enumerate(paths_img):
        print(f'Processing ({idx_s + 1}/{len(paths_img)}):', path_img)
        meta = {
            'path_tif': os.path.abspath(path_img),
            'index_sequence': idx_s,
        }
        nodes.extend(img_to_nodes(tifffile.imread(path_img), meta=meta))
        if idx_s >= 1:
            break
    dirname = os.path.dirname(path_save_csv)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('Created:', dirname)
    pd.DataFrame(nodes).rename_axis('node_id', axis=0).to_csv(path_save_csv)
    print('Saved:', path_save_csv)
