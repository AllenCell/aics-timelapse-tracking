import pandas as pd


def calc_track_lengths(df: pd.DataFrame) -> pd.Series:
    """Returns track_ids sorted by track length.

    Parameters
    ----------
    df
        DataFrame representing time-lapse graph with 'track_id' column.

    """
    if 'track_id' not in df.columns:
        raise ValueError('"track_id" column expected in df')
    # For each track_id, count number of unique frames to get track length
    track_lengths = (
        df
        .groupby('track_id')['index_sequence']
        .nunique()
        .sort_values(ascending=False)
    )
    return track_lengths


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
