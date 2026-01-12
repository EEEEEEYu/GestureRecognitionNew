import numpy as np


def filter_noise_spatial(
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    grid_size: int = 4,
    threshold: int = 2
):
    """
    STAGE 1: Polarity-Agnostic Spatial Denoising (Numpy Implementation)
    Filters out events that lack local neighbors within the current time batch.
    This filter is polarity-agnostic (counts events regardless of polarity).
    
    Args:
        t, y, x, p: Event arrays (time, y-coord, x-coord, polarity)
        height, width: Sensor dimensions
        grid_size: Size of the bucket to check density (e.g., 4x4 pixels).
        threshold: Minimum number of events required in a bucket to survive.
    
    Returns:
        Filtered t, y, x, p arrays (numpy)
    """
    if len(t) == 0:
        return t, y, x, p
    
    # 1. Map to Coarse Grid
    y_grid = (y.astype(np.int32) // grid_size)
    x_grid = (x.astype(np.int32) // grid_size)
    
    # Calculate dimensions of the coarse grid
    h_grid = (height + grid_size - 1) // grid_size
    w_grid = (width + grid_size - 1) // grid_size
    
    # 2. Compute Flat Indices for the Grid
    flat_indices = y_grid * w_grid + x_grid
    
    # 3. Count Events per Grid Cell (Histogram)
    # Polarity-agnostic: count all events in the cell
    num_bins = h_grid * w_grid
    counts = np.bincount(flat_indices, minlength=num_bins)
    
    # 4. Create Validity Mask
    valid_bins_mask = counts >= threshold
    
    # 5. Map back to per-event mask
    event_mask = valid_bins_mask[flat_indices]
    
    # 6. Filter
    return t[event_mask], y[event_mask], x[event_mask], p[event_mask]


def sample_grid_decimation_fast(
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    target_grid: int = 2,
    retention_ratio: float = 0.3,
    return_indices: bool = False
):
    """
    Real-time Optimized Stratified Sampling (O(N log N)).
    
    Logic:
    1. Sort events by (Block_ID, -Timestamp).
    2. Assign a 'rank' to every event within its block (0=newest).
    3. Select events with the lowest ranks globaly.
    
    Args:
        retention_ratio: 0.0 to 1.0 (Target % of events to keep)
    """
    num_events = len(t)
    if num_events == 0:
        return (np.array([], dtype=np.int64) if return_indices 
                else (t, y, x, p))
    
    # Target number of samples
    k_target = int(num_events * retention_ratio)
    if k_target >= num_events:
        indices = np.arange(num_events)
        if return_indices: return indices
        return t, y, x, p

    # 1. Compute Block Keys (Integer Hashing)
    # Using int32 is faster for sorting than int64
    y_block = (y // target_grid).astype(np.int32)
    x_block = (x // target_grid).astype(np.int32)
    w_block = (width + target_grid - 1) // target_grid
    
    # key = y * width + x
    block_keys = y_block * w_block + x_block

    # 2. LexSort: Primary=Block, Secondary=Time (Descending)
    # np.lexsort sorts by the LAST key in the sequence first.
    # We want: Sort by Block (Asc), then Time (Desc).
    # Since we can't easily negate 't' if it's unsigned or large, 
    # we sort by Block(Asc), Time(Asc) and handle the order via indexing.
    # OR: just negate t if it fits in memory. Let's trust t is float or int.
    
    # Sort order: Block ASC, Time DESC (Newest first)
    sort_idx = np.lexsort((-t, block_keys))
    
    # Get the block keys in sorted order
    sorted_keys = block_keys[sort_idx]
    
    # 3. Vectorized Rank Calculation (The "Magic" Step)
    # We want to know: "Is this the 1st, 2nd, or 3rd event in this block?"
    
    # Find where blocks change. (True at index i if key[i] != key[i-1])
    # We append True at start to handle the first group
    diff = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    
    # Get the indices where groups start
    group_starts = np.flatnonzero(diff)
    
    # Create a "group_start_pointer" for every element.
    # We repeat the start_index for the length of the group.
    # This can be done by repeating the diffs, but easier method:
    # Use maximum.accumulate on a sparse array.
    
    # Create an array that maps every index to the start index of its group
    group_start_map = np.zeros(num_events, dtype=np.int32)
    group_start_map[group_starts] = group_starts
    
    # Propagate the start index forward
    # e.g. [0, 0, 0, 5, 0, 0] -> [0, 0, 0, 5, 5, 5]
    np.maximum.accumulate(group_start_map, out=group_start_map)
    
    # Rank = Current_Index - Group_Start_Index
    # rank 0 = Latest, rank 1 = 2nd latest...
    ranks = np.arange(num_events) - group_start_map
    
    # 4. Selection based on Rank
    # We want the 'k_target' events with the lowest ranks.
    # If we just take the first k_target, we prioritize Block 0 over Block 100. Bad.
    # We must sort by Rank to distribute samples evenly across space.
    
    # Stable sort by rank ensures we cycle through blocks evenly
    # (Rank 0 from Block A, Rank 0 from Block B ... Rank 1 from Block A...)
    
    # We sort the *ranks* to find the "cut" indices.
    # Optimization: We don't need a full sort, just 'argpartition' 
    # or just sort the valid subset. But 'argsort(ranks)' is fast on integers.
    rank_order = np.argsort(ranks, kind='stable')
    
    # Select the top K candidates from the sorted view
    selected_candidates = rank_order[:k_target]
    
    # Map back to original indices
    final_indices = sort_idx[selected_candidates]
    
    # 5. Temporal Sort (Optional but recommended)
    final_indices.sort()
    
    if return_indices:
        return final_indices
        
    return t[final_indices], y[final_indices], x[final_indices], p[final_indices]