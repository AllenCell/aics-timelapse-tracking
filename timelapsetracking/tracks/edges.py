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
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].to_numpy()
            volumes_curr = df_curr['volume'].to_numpy()
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
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
            indices_curr = list(df_curr.index[edge_to])
            df_edges.loc[idx_prev, 'out_list'] = str(indices_curr)
            for idx_curr in indices_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'
        df_prev = df_curr
    return df.join(df_edges)
