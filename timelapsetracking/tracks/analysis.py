import pandas as pd


def calc_track_lengths(
        df: pd.DataFrame, use_lineage_id: bool = False
) -> pd.Series:
    """Returns track or lineage ids sorted by frame count.

    Parameters
    ----------
    df
        DataFrame representing time-lapse graph with 'track_id' column.
    use_lineage_id
        Set to calculate frame counts for each lineage id instead of track id.

    Returns
    -------
    pd.Series
        Track or lineage lengths.

    """
    col = 'track_id' if not use_lineage_id else 'lineage_id'
    if col not in df.columns:
        raise ValueError(f'{col} column expected in df')
    lengths = (
        df
        .groupby(col)['index_sequence']
        .nunique()
        .sort_values(ascending=False)
    )
    return lengths


def add_delta_pos(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative xyz displacement columns for nodes with parents.

    Parameters
    ----------
    df
        Track graph as a DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added x, y, or z displacements.

    """
    if 'in_list' not in df.columns:
        raise ValueError('DataFrame should have "in_list" column')
    cols_pos = ('centroid_x', 'centroid_y', 'centroid_z')
    if not any(col in df.columns for col in cols_pos):
        raise ValueError(
            f'DataFrame should have at least 1 position column {cols_pos}'
        )
    mask = df['in_list'].notna()
    parents = df['in_list'].dropna().astype(int).to_numpy()
    for col in cols_pos:
        if col not in df.columns:
            continue
        children_pos = df.loc[mask, col]
        parents_pos = df.loc[parents, col].values
        df.loc[mask, 'delta_' + col] = children_pos - parents_pos
    return df
