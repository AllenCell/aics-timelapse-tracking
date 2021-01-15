"""Functions to find connections between nodes in time-lapse graph."""

import logging

import pandas as pd

from timelapsetracking.tracks.correspondances import find_correspondances

from tqdm import tqdm


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
    df_edges['is_pair'] = df['is_pair']
    df_edges['has_pair'] = df['has_pair']
    for col in cols_zyx:
        df_edges[col] = df[col]
    index_vals = df[col_index_sequence].unique()
    for idx_s in tqdm(range(index_vals.min()+1, index_vals.max() + 1)):

        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')
        print(f'Processing {col_index_sequence}: {idx_s}')
        df_prev = df.query(f'{col_index_sequence} == (@idx_s-1)')
        keep_idxs = [index for index, row in df_prev.iterrows() 
                        if not row['is_pair']]
        df_prev = df_prev.loc[keep_idxs].copy()

        df_curr = df.query(f'{col_index_sequence} == @idx_s')
        # keep_idxs = [index for index, row in df_curr.iterrows() 
        #                 if not row['has_pair']]
        # df_curr = df_curr.loc[keep_idxs].copy()


        
        has_pair = [idx for idx, row in df_prev.iterrows() 
                        if row['has_pair']]
        
        is_pair = [idx for idx, row in df_curr.iterrows() 
                        if row['is_pair']]

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None

        has_pair_curr = df_curr['has_pair'].values

        # import pdb; pdb.set_trace()
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        # edge_distances_prev = df_prev['edge_distance'].values
        # edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            # cost_add=edge_distances_curr,
            # cost_delete=edge_distances_prev,
            has_pair_b=has_pair_curr,
            **kwargs
        )

        pair_edges = []

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


            # skip edges pointing to pair nodes for next step
            skip = False
            for idx_curr in idxs_curr:
                if idx_curr in is_pair:
                    # print('edge to is pair')
                    # print((edge_from, edge_to))
                    pair_edges.append((edge_from, edge_to))
                    skip = True
                    break
            if skip:
                continue
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'

        #redo edge mapping and splitting for edge nodes
        df_curr_has_pair = df.query(f'{col_index_sequence} == @idx_s')
        keep_idxs = [index for index, row in df_curr_has_pair.iterrows() 
                        if row['has_pair']]
        df_curr_has_pair = df_curr_has_pair.loc[keep_idxs].copy()
        for edge_from, edge_to in pair_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            # pairs = [index for index, row in df_curr.iterrows() 
            #             if row['is_pair']] 

            # df_curr_not_pair = df_curr.copy()
            # for idx in pairs:
            #     df_curr_not_pair['label_img'][idx] = []

            idxs_curr = df_curr.index[edge_to]
            

            if df_edges.loc[idxs_curr, 'is_pair'].values:
                child_list = []
                for label in df_edges.loc[idxs_curr,'label_img'].values[0]:
                    child_list.append(df_curr_has_pair.index[df_curr_has_pair['label_img'] == label].values[0])
                
                idxs_curr = child_list
                
            print('pair match')
            print(idx_prev)
            print(idxs_curr)

            # import pdb
            # pdb.set_trace()

            for idx_curr in idxs_curr:
                divorce_cell = eval(df_edges.loc[idx_curr, 'in_list'])
                if len(divorce_cell) > 0:
                    # print('overlapping match to pair cell. Overwriting previous match')
                    print('divorcing cell')
                    for cell in divorce_cell:
                        df_edges.loc[divorce_cell[0], 'out_list'] = '[]'

            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            df_edges.loc[idxs_curr, 'in_list'] = f'[{idx_prev}]'

        # del centroids_prev, centroids_curr, volumes_curr, volumes_prev, edge_distances_curr, edge_distances_prev, df_prev
        # do not include pair nodes in list of previous nodes
        # keep_idxs = [index for index, row in df_curr.iterrows() 
        #                 if not row['is_pair']]
        # df_prev = df_curr.loc[keep_idxs].copy()


        
    # attempt to bridge non-edge cells with missing graph edges
    df_edges = bridge_edges(df_edges, col_index_sequence, 3, **kwargs)

    # remove pair nodes in cuur from full dataframe
    remove_idx = [index for index, row in df_edges.iterrows() 
                    if row['is_pair']]
    
    df = df.drop(remove_idx)


    df_edges = df_edges.drop(columns=['label_img', 
                                        'index_sequence', 
                                        'volume', 
                                        'edge_distance', 
                                        'edge_cell', 
                                        'is_pair',
                                        'has_pair'])
    df_edges = df_edges.drop(columns=cols_zyx)
    return df.join(df_edges)

def bridge_edges(
        df_edges: pd.DataFrame, 
        col_index_sequence: str = 'index_sequence',
        window: int = 3, 
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
    if window < 0:
        window = 0
    
    for idx_s in tqdm(range(index_vals.min()+1, index_vals.max() + 1)):

        LOGGER.info(f'Processing {col_index_sequence}: {idx_s}')
        df_prev = df_edges.query(f'{col_index_sequence} == (@idx_s-1)')
        keep_idxs = [idx for idx, row in df_prev.iterrows() 
                        if len(eval(row['out_list'])) == 0 
                        and not row['edge_cell']
                        and not row['is_pair']]
        df_prev = df_prev.loc[keep_idxs].copy()

        df_curr = df_edges.query(f'{col_index_sequence} >= @idx_s and {col_index_sequence} <= @idx_s + {window}')
        keep_idxs = [idx for idx, row in df_curr.iterrows() 
                        if len(eval(row['in_list'])) == 0
                        and not row['edge_cell']]
        df_curr = df_curr.loc[keep_idxs].copy()

        if len(df_prev.index) == 0 or len(df_curr.index) == 0:
            continue
        
        # import pdb
        # pdb.set_trace()
        has_pair = [idx for idx, row in df_prev.iterrows() 
                        if row['has_pair']]
        is_pair = [idx for idx, row in df_curr.iterrows() 
                        if row['is_pair']]

        centroids_prev = df_prev.filter(cols_zyx).values
        centroids_curr = df_curr.filter(cols_zyx).values
        volumes_prev, volumes_curr = None, None
        
        has_pair_curr = df_curr['has_pair'].values

        # import pdb; pdb.set_trace()
        if 'volume' in df_curr.columns:
            volumes_prev = df_prev['volume'].values#to_numpy()
            volumes_curr = df_curr['volume'].values#to_numpy()
        # edge_distances_prev = df_prev['edge_distance'].values
        # edge_distances_curr = df_curr['edge_distance'].values
        edges = find_correspondances(
            centroids_a=centroids_prev,
            centroids_b=centroids_curr,
            volumes_a=volumes_prev,
            volumes_b=volumes_curr,
            # cost_add=edge_distances_curr,
            # cost_delete=edge_distances_prev,
            has_pair_b=has_pair_curr,
            is_bridge=True,
            **kwargs
        )

        pair_edges = []

        new_edges = [edge for edge in edges if all(edge)]
        print(f'{len(new_edges)} new edges bridged')

        for edge_from, edge_to in new_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            idxs_curr = list(df_curr.index[edge_to])


            # skip edges pointing to pair nodes for next step
            skip = False
            for idx_curr in idxs_curr:
                if idx_curr in is_pair:
                    # print('edge to is pair')
                    # print((edge_from, edge_to))
                    pair_edges.append((edge_from, edge_to))
                    skip = True
                    break
            if skip:
                continue
             
            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            for idx_curr in idxs_curr:
                df_edges.loc[idx_curr, 'in_list'] = f'[{idx_prev}]'

        #redo edge mapping and splitting for edge nodes
        df_curr_has_pair = df_edges.query(f'{col_index_sequence} >= @idx_s and {col_index_sequence} <= @idx_s + {window}')
        keep_idxs = [index for index, row in df_curr_has_pair.iterrows() 
                        if row['has_pair']]
        df_curr_has_pair = df_curr_has_pair.loc[keep_idxs].copy()
        for edge_from, edge_to in pair_edges:
            if edge_from is None or edge_to is None:
                continue
            idx_prev = df_prev.index[edge_from]
            if isinstance(edge_to, int):
                edge_to = [edge_to]
            elif isinstance(edge_to, tuple):
                # Convert to list for proper indexing below
                edge_to = list(edge_to)

            # pairs = [index for index, row in df_curr.iterrows() 
            #             if row['is_pair']] 

            # df_curr_not_pair = df_curr.copy()
            # for idx in pairs:
            #     df_curr_not_pair['label_img'][idx] = []

            idxs_curr = df_curr.index[edge_to]
            

            if df_edges.loc[idxs_curr, 'is_pair'].values:
                child_list = []
                for label in df_edges.loc[idxs_curr,'label_img'].values[0]:
                    try:
                        child_list.append(df_curr_has_pair.index[df_curr_has_pair['label_img'] == label].values[0])
                    except:
                        import pdb
                        pdb.set_trace()
                
                idxs_curr = child_list
                
            print('pair match')
            print(idx_prev)
            print(idxs_curr)

            # import pdb
            # pdb.set_trace()

            for idx_curr in idxs_curr:
                divorce_cell = eval(df_edges.loc[idx_curr, 'in_list'])
                if len(divorce_cell) > 0:
                    # print('overlapping match to pair cell. Overwriting previous match')
                    print('divorcing cell')
                    for cell in divorce_cell:
                        df_edges.loc[cell, 'out_list'] = '[]'

            df_edges.loc[idx_prev, 'out_list'] = str(idxs_curr)
            df_edges.loc[idxs_curr, 'in_list'] = f'[{idx_prev}]'

    return df_edges