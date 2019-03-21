from typing import List
import pandas as pd


def find_longest_tracks(df: pd.DataFrame) -> List[int]:
    """Returns sorted list of track_ids of longest track(s).

    Parameters
    ----------
    df
        DataFrame representing time-lapse graph with 'track_id' column.

    """
    assert 'track_id' in df.columns, '"track_id" should be in input DataFrame'
    # For each track_id, count number of unique frames to get track length
    track_lengths = (
        df
        .groupby('track_id')['index_sequence']
        .nunique()
        .sort_values(ascending=False)
    )
    track_ids = sorted(
        track_lengths.loc[track_lengths == track_lengths.max()].index
    )
    return track_ids
