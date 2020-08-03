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
        im_shape: np.ndarray, fov: List[Dict], meta: Optional[Dict] = None
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
    if len(im_shape) not in [2, 3]:
        raise ValueError('Images must be 2d or 3d')
    # if not np.issubdtype(img.dtype, np.unsignedinteger):
    #     raise TypeError('Image must be np.unsignedinteger compatible')
    if meta is None:
        meta = {}

    # field_shape = np.array(img.shape)
    field_shape = im_shape
    # origin = np.zeros_like(field_shape)
    nodes = []
    pairs_done = []
    for index in fov.index:
        node = {}
        # if fov.Volume[index] < 50000:
        #     continue
        node['volume'] = fov.Volume[index]
        node['label_img'] = fov.CellLabel[index]
        node['is_pair'] = False
        node['has_pair'] = False

        rcm = eval(fov.Centroid[index])
        edge_distance = np.amax(field_shape)
        for idx_d, dim in enumerate('zyx'[-len(im_shape):]):
            node[f'centroid_{dim}'] = rcm[idx_d]
            if dim != 'z':
                edge_distance = np.amin([edge_distance, 
                                        rcm[idx_d], 
                                        field_shape[idx_d]-rcm[idx_d]])
        node['edge_distance'] = edge_distance

        # check if cell segmentation is touching boundary and label as edge cell
        # obj_idxs = None
        # obj_idxs = np.argwhere(img==node['label_img'])

        # if obj_idxs.size == 0:
        #     continue
        
        # min_coors = np.min(obj_idxs, axis=0)
        # max_coors = np.max(obj_idxs, axis=0)
        # is_edge =  np.any(np.logical_or(np.equal(min_coors, origin),
        #                         np.equal(max_coors, field_shape-1)))

        # node['edge_cell'] = is_edge

        # import pdb
        # pdb.set_trace()

        node['edge_cell'] = bool(fov.Edge_Cell[index])

        if fov.Pair[index] != 0:
            node['has_pair'] = True

        node.update(meta)
        nodes.append(node)

        if fov.Pair[index] != 0:
            pair_node = {}
            idx_partner = fov.index[fov['CellLabel']==fov.Pair[index]].tolist()[0]
            if idx_partner in pairs_done:
                continue
            else:
                pairs_done.append(index)
                pairs_done.append(idx_partner)
            pair_node['volume'] = (node['volume'] + fov.Volume[idx_partner])//2
            pair_labels = [node['label_img'], fov.CellLabel[idx_partner]]
            pair_labels.sort()
            pair_node['label_img'] = pair_labels
            pair_node['is_pair'] = True
            pair_node['has_pair'] = False


            # obj_idxs = None
            # for label in pair_labels:
            #     if obj_idxs is None:
            #         obj_idxs = np.argwhere(img==label)
            #     else:
            #         obj_idxs = np.concatenate((obj_idxs, np.argwhere(img==label)),
            #                                 axis=0)

            # min_coors = np.min(obj_idxs, axis=0)
            # max_coors = np.max(obj_idxs, axis=0)
            # is_edge =  np.any(np.logical_or(np.equal(min_coors, origin),
            #                     np.equal(max_coors, field_shape-1)))
            
            # pair_node['edge_cell'] = is_edge

            pair_node['edge_cell'] = bool(fov.Edge_Cell[index] + fov.Edge_Cell[idx_partner])

            rcm_p = eval(fov.Centroid[idx_partner])
            edge_distance = np.amax(field_shape)
            for idx_d, dim in enumerate('zyx'[-len(im_shape):]):
                pair_node[f'centroid_{dim}'] = (rcm_p[idx_d] + node[f'centroid_{dim}'])/2
                if dim != 'z':
                    edge_distance = np.amin([edge_distance, pair_node[f'centroid_{dim}'], field_shape[idx_d]-pair_node[f'centroid_{dim}']])
            pair_node['edge_distance'] = edge_distance
            pair_node.update(meta)
            nodes.append(pair_node)
            print('Pair Node Created')

        
    return nodes


def _img_to_nodes_wrapper(in_args) -> List[Dict]:
    # meta: Dict, im_shape: np.ndarray
    meta = in_args[0]
    im_shape = in_args[1]
    """Wrapper for 'map' method of multiprocessing.Pool."""
    logger.info(f'Processing: {meta["path_tif"]}')
    print(meta['path_tif'])
    # img = tifffile.imread(meta['path_tif'])
    fov = pd.read_csv(meta['path_tif'].replace('.tiff','.tif').replace('.tif','_region_props.csv'))#, index_col=0)
    return img_to_nodes(im_shape, fov, meta=meta)


@report_run_time
def dir_to_nodes(
        path_img_dir: Path,
        im_shape: np.ndarray,
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
    im_shapes = [
        im_shape
        for _, path in enumerate(paths_img)
    ]

    # import pdb
    # pdb.set_trace()

    # _img_to_nodes_wrapper(metas[0])

    with Pool(min(num_processes, os.cpu_count())) as pool:
        nodes_per_img = pool.map(_img_to_nodes_wrapper, zip(metas, im_shapes))

    nodes = []
    for per_img in nodes_per_img:
        nodes.extend(per_img)
    
    df = pd.DataFrame(nodes)
    df = pd.concat([df[[not i for i in df.is_pair.values]],
                    df[df.is_pair.values]]).reset_index(drop=True).rename_axis('node_id', axis=0)
    # df = df.drop(columns=['is_pair'])
    return df
