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


def filter_noise_spatial_temporal(
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    grid_size: int = 4,
    time_window_us: int = 10000,
    threshold: int = 2
):
    """
    STAGE 1.5: Spatial-Temporal Denoising
    Filters events using a 3D grid (x, y, t).
    
    Args:
        time_window_us: Temporal grid size in microseconds.
    """
    if len(t) == 0:
        return t, y, x, p
    
    # 1. Map to Coarse 3D Grid
    y_grid = (y.astype(np.int32) // grid_size)
    x_grid = (x.astype(np.int32) // grid_size)
    t_grid = (t.astype(np.int64) // time_window_us)
    
    # Calculate spatial grid dims
    h_grid = (height + grid_size - 1) // grid_size
    w_grid = (width + grid_size - 1) // grid_size
    
    # We don't know max time, but we can re-index t_grid to start from 0 for hashing
    t_grid_offset = t_grid - t_grid.min()
    
    # 2. Compute Flat Indices (using a larger space or tuple hashing)
    # Since numpy bincount needs 1D int, we can use linear indexing if size permits.
    # Safe bet: unique hashing
    # key = t * (H*W) + y * W + x
    grid_area = h_grid * w_grid
    flat_indices = t_grid_offset * grid_area + (y_grid * w_grid + x_grid)
    
    # 3. Count Events per ST Grid Cell
    # Using np.unique is safer than bincount for potentially large/sparse indices, 
    # but slower. Recalibrating indices for bincount if possible.
    # If time span is small (50ms) and time_window is 10ms, t_grid_max is small (~5).
    # So bincount is safe.
    
    t_max = t_grid_offset.max()
    num_bins = (t_max + 1) * grid_area
    
    if num_bins > 1e8: # Safety for memory
        # Fallback to unique return_counts if range is huge
        unique_ids, counts = np.unique(flat_indices, return_counts=True)
        valid_ids = unique_ids[counts >= threshold]
        # Use simple map
        # This is slow for lookups.
        # Better: construct mask directly.
        valid_set = set(valid_ids) # Optimization?
        # Vectorized lookup:
        # np.isin is okay.
        event_mask = np.isin(flat_indices, valid_ids)
    else:
        counts = np.bincount(flat_indices.astype(np.int64), minlength=num_bins.astype(np.int64))
        valid_bins_mask = counts >= threshold
        event_mask = valid_bins_mask[flat_indices]

    return t[event_mask], y[event_mask], x[event_mask], p[event_mask]


def filter_background_activity(
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    time_threshold: int = 1000  # in same unit as t (us)
):
    """
    Background Activity Filter (BAF).
    Checks if an event interacts with recent activity at the same pixel (or neighborhood).
    Simplified: Per-pixel Check.
    
    Args:
        time_threshold: Max delta-t to consider "correlated".
    """
    if len(t) == 0:
        return t, y, x, p
        
    # We process in order.
    # For a batch, we can identify "supported" events.
    # An event is supported if there is another event at (x,y) within time_threshold.
    
    # Sort just in case (though usually sorted)
    sort_idx = np.argsort(t)
    t_s, y_s, x_s = t[sort_idx], y[sort_idx], x[sort_idx]
    
    # Linear index for pixels
    pixel_idx = y_s.astype(np.int64) * width + x_s.astype(np.int64)
    
    # To vectorize:
    # We want d_t = t[i] - t[prev_at_pixel].
    # We can use lexsort(t, pixel) -> group by pixel, then time.
    
    # 1. Sort by Pixel then Time
    lex_idx = np.lexsort((t_s, pixel_idx))
    
    pixel_sorted = pixel_idx[lex_idx]
    t_sorted = t_s[lex_idx]
    
    # 2. Compute dt
    # dt[i] = t[i] - t[i-1]
    dt = np.zeros_like(t_sorted)
    dt[1:] = t_sorted[1:] - t_sorted[:-1]
    
    # 3. Mask: valid if (same pixel as prev) AND (dt < threshold)
    # Check pixel continuity
    pixel_diff = np.zeros_like(pixel_sorted)
    pixel_diff[1:] = pixel_sorted[1:] - pixel_sorted[:-1]
    
    # Condition 1: Support from PAST
    supported_by_past = (pixel_diff == 0) & (dt <= time_threshold)
    
    # Condition 2: Support from FUTURE (since we are doing batch processing, future is available!)
    # Shift arrays to check next element
    # A cluster of 2 events: A -> B.
    # A is supported by B? B is supported by A?
    # Usually BAF is causal (online). But offline we can keep both.
    # Let's start with Causal (supported by past) to mimic hardware BAF?
    # Or "Any neighbor". "Any neighbor" is better for quality.
    
    supported_by_future = np.zeros_like(supported_by_past)
    supported_by_future[:-1] = supported_by_past[1:] # If i+1 is supported by i, then i is supported by future i+1 (same pixel)
    
    keep_mask_sorted = supported_by_past | supported_by_future
    
    # Map back to original indices
    # We have sort_idx and lex_idx.
    # original -> [sort] -> [lex]
    # We need to map boolean mask back.
    
    final_mask = np.zeros_like(keep_mask_sorted)
    
    # Reversing the permutation:
    # The mask corresponds to t_sorted, which corresponds to lex_idx applied to sort_idx applied to original.
    # combined_perm = sort_idx[lex_idx]
    # final_mask[combined_perm] = keep_mask_sorted
    
    combined_perm = sort_idx[lex_idx]
    final_mask[combined_perm] = keep_mask_sorted
    
    return t[final_mask], y[final_mask], x[final_mask], p[final_mask]
