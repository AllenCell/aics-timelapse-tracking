"""Functions to find connections between nodes in time-lapse graph."""


import pandas as pd

from timelapsetracking.tracks.correspondances import find_correspondances


def add_edges(
        df: pd.DataFrame,
        col_index_sequence: str = 'index_sequence',
        method_first: str = 'interior-point',
) -> pd.DataFrame:
    """Add edges to track graph based on centroids.

    Parameters
    ----------
    df
        Input time-lapse graph with unconnected nodes.
    col_index_sequence
        DataFrame column corresponding to the time-lapse frame/sequence number.
    method_first
        Linear programming method to try first.

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
        .fillna('')
    )
    for idx_s, df_curr in df.groupby(col_index_sequence):
        print(f'Processing {col_index_sequence}:', idx_s)
        if df_prev is None:
            df_prev = df_curr
            continue
        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        edges = find_correspondances(
            centroids_prev, centroids_curr, method_first=method_first
        )
        for edge in edges:
            if edge[0] is not None and edge[1] is not None:
                idx_prev = df_prev.index[edge[0]]
                if isinstance(edge[1], tuple):
                    raise NotImplementedError
                idx_curr = df_curr.index[edge[1]]
                df_edges.loc[idx_prev, 'out_list'] = str(idx_curr)
                df_edges.loc[idx_curr, 'in_list'] = str(idx_prev)
        df_prev = df_curr
    return df.join(df_edges)
