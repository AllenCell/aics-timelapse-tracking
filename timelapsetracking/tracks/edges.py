"""Functions to find connections between nodes in time-lapse graph."""

import logging

import pandas as pd

from timelapsetracking.tracks.correspondances import find_correspondances


LOGGER = logging.getLogger(__name__)


def add_edges(
        df: pd.DataFrame,
        col_index_sequence: str = 'index_sequence',
        **kwargs,
) -> pd.DataFrame:
    """Add edges to track graph based on centroids.

    Parameters
    ----------
    df
        Input time-lapse graph with unconnected nodes.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    **kwargs
        Additional arguments to be passed to find_correspondances function.

    Returns
    -------
    pd.DataFrame
        Time-lapse graph with added edges

    """
    if col_index_sequence not in df.columns:
        raise ValueError(f'"{col_index_sequence}" column expected in df')
    df_prev = None
    cols_zyx = ['centroid_z', 'centroid_y', 'centroid_x']

    df_edges = (
        pd.DataFrame(index=df.index, columns=['in_list', 'out_list'])
        .fillna('[]')
    )
    df_edges['label_img'] = df['label_img']
    df_edges['index_sequence'] = df['index_sequence']
    df_edges['volume'] = df['volume']
    df_edges['edge_distance'] = df['edge_distance']
    df_edges['edge_cell'] = df['edge_cell']
    for col in cols_zyx:
        df_edges[col] = df[col]
    index_vals = df[col_index_sequence].unique()
    for idx_s in range(index_vals.min(), index_vals.max() + 1):
        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')
        df_curr = df.query(f'{col_index_sequence} == @idx_s')
        if df_prev is None:
            df_prev = df_curr
            continue

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None
        # import pdb; pdb.set_trace()
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        edge_distances_prev = df_prev['edge_distance'].values
        edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            cost_add=edge_distances_curr,
            cost_delete=edge_distances_prev,
            **kwargs
        )
        for edge_from, edge_to in edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)
            idxs_curr = list(df_curr.index[edge_to])

            # split edge pointing to pair node
            pairs = [index for index, row in df_curr.iterrows() 
                        if isinstance(row['label_img'], list)] 
            df_curr_not_pair = df_curr.copy()
            for idx in pairs:
                df_curr_not_pair['label_img'][idx] = []

            if isinstance(df_edges.loc[idxs_curr, 'label_img'].values[0], list):
                child_list = []
                for label in df_edges.loc[idxs_curr,'label_img'].values[0]:
                    child_list.append(df_curr_not_pair.index[df_curr_not_pair['label_img']==label].values[0])
                idxs_curr = child_list
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'
        # do not include pair nodes in list of previous nodes
        keep_idxs = [index for index, row in df_curr.iterrows() 
                        if not isinstance(row['label_img'], list)]
        df_prev = df_curr.loc[keep_idxs]
        
    # attempt to bridge non-edge cells with missing graph edges
    df_edges = bridge_edges(df_edges, col_index_sequence, 3, **kwargs)

    # remove pair nodes in cuur from full dataframe
    remove_idx = [index for index, row in df_edges.iterrows() 
                    if isinstance(row['label_img'], list)]
    
    df = df.drop(remove_idx)


    df_edges = df_edges.drop(columns=['label_img', 'index_sequence', 'volume', 'edge_distance', 'edge_cell'])
    df_edges = df_edges.drop(columns=cols_zyx)
    return df.join(df_edges)

def bridge_edges(
        df_edges: pd.DataFrame, 
        col_index_sequence: str = 'index_sequence',
        window: int = 5, 
        **kwargs
    ) -> pd.DataFrame:
    """Attempt to bridge gaps in node graph.

    Parameters
    ----------
    df_edges
        List of nodes after intial tracking with missing graph edges.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    window
        Number of timepoints in the future to look for potential edges.
    **kwargs
        Additional arguments to be passed to find_correspondances function.

    Returns
    -------
    pd.DataFrame
        Time-lapse graph with added edges

    """

    cols_zyx = ['centroid_z', 'centroid_y', 'centroid_x']

    index_vals = df_edges[col_index_sequence].unique()
    for idx_s in range(index_vals.min(), index_vals.max()):
        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')

        # collect non-edge cells with no outbound edges
        df_prev = df_edges.query(f'{col_index_sequence} == @idx_s')
        no_out = [idx for idx, row in df_prev.iterrows() 
                    if len(eval(row['out_list'])) == 0 
                    and not row['edge_cell']]
        df_prev = df_prev.loc[no_out]

        # collect all non-edge cells in future window with no inbound edges
        df_curr = df_edges.query(f'{col_index_sequence} > @idx_s + 1 and {col_index_sequence} <= @idx_s + {window}')
        no_in = [idx for idx, row in df_curr.iterrows() 
                    if len(eval(row['in_list'])) == 0
                    and not row['edge_cell']]
        df_curr = df_curr.loc[no_in]

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None
        
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        edge_distances_prev = df_prev['edge_distance'].values
        edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            cost_add=edge_distances_curr,
            cost_delete=edge_distances_prev,
            **kwargs
        )
        for edge_from, edge_to in edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)
            idxs_curr = list(df_curr.index[edge_to])

            # split edge pointing to pair node
            pairs = [index for index, row in df_curr.iterrows() 
                        if isinstance(row['label_img'], list)] 
            df_curr_not_pair = df_curr.copy()
            for idx in pairs:
                df_curr_not_pair['label_img'][idx] = []

            if isinstance(df_edges.loc[idxs_curr, 'label_img'].values[0], list):
                child_list = []
                for label in df_edges.loc[idxs_curr,'label_img'].values[0]:
                    child_list.append(df_curr_not_pair.index[df_curr_not_pair['label_img']==label].values[0])
                idxs_curr = child_list
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'

    return df_edges