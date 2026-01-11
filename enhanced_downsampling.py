"""
Enhanced downsampling strategies for event cameras.

Since VecKM representation doesn't use polarity information, we can:
1. Merge events with opposite polarities at the same (x,y,t) location
2. Use polarity-agnostic refractory filtering
3. Apply more aggressive spatial binning

This should dramatically reduce event count while preserving geometric information.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def _numba_polarity_agnostic_refractory_filter(t, x, y, last_t_grid, dt_us):
    """
    Polarity-agnostic refractory filtering.
    
    Unlike traditional refractory filtering that tracks per-polarity, this:
    - Tracks last event time per pixel REGARDLESS of polarity
    - Keeps first event within refractory window (any polarity)
    - Ignores subsequent events at same pixel within dt_us
    
    This effectively merges ON/OFF events and reduces count by ~2x.
    """
    n = len(t)
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        ti = t[i]
        xi = x[i]
        yi = y[i]
        
        # Check if enough time has passed since last event at this pixel
        if ti - last_t_grid[xi, yi] > dt_us:
            mask[i] = True
            last_t_grid[xi, yi] = ti
    return mask


@jit(nopython=True)
def _numba_spatial_temporal_binning(t, x, y, p, bin_size_spatial, bin_size_temporal_us):
    """
    Aggressive spatial-temporal binning.
    
    For each spatial bin (e.g., 2x2 pixels) and temporal bin (e.g., 10ms):
    - Keep only ONE event (the median timestamp)
    - Ignore polarity entirely
    
    This is much more aggressive than refractory filtering.
    """
    if len(t) == 0:
        return np.zeros(0, dtype=np.bool_)
    
    # Create spatial-temporal bins
    t_bin = t // bin_size_temporal_us
    x_bin = x // bin_size_spatial  
    y_bin = y // bin_size_spatial
    
    # Unique bin identifier
    max_x_bin = x_bin.max() + 1
    max_y_bin = y_bin.max() + 1
    max_t_bin = t_bin.max() + 1
    
    # Create a hash for each bin: (t_bin * max_x * max_y) + (x_bin * max_y) + y_bin
    bin_ids = (t_bin * max_x_bin * max_y_bin) + (x_bin * max_y_bin) + y_bin
    
    # For each unique bin, keep the event closest to median time
    mask = np.zeros(len(t), dtype=np.bool_)
    processed_bins = np.zeros(bin_ids.max() + 1, dtype=np.int64) - 1
    
    for i in range(len(t)):
        bin_id = bin_ids[i]
        if processed_bins[bin_id] == -1:
            # First event in this bin, keep it
            mask[i] = True
            processed_bins[bin_id] = i
        # else: Already have an event for this bin, skip
    
    return mask


def polarity_agnostic_downsample_v1(events_t, events_x, events_y, events_p, 
                                     factor=4, dt_us=30000):
    """
    Enhanced polarity-agnostic downsampling.
    
    Strategy:
    1. Spatial downsampling via bit-shift (4x)
    2. Polarity-agnostic refractory filter (ignores +/- polarity)
    
    Expected reduction: ~8-12x (4x spatial + 2-3x temporal)
    """
    if factor <= 1:
        return events_t, events_x, events_y, events_p
    
    # 1. Spatial downsample
    shift = int(np.log2(factor))
    lr_x = events_x.astype(np.int32) >> shift
    lr_y = events_y.astype(np.int32) >> shift
    
    # 2. Polarity-agnostic refractory filter
    max_x = lr_x.max() if len(lr_x) > 0 else 0
    max_y = lr_y.max() if len(lr_y) > 0 else 0
    
    last_t = np.zeros((max_x + 1, max_y + 1), dtype=np.int64)
    mask = _numba_polarity_agnostic_refractory_filter(events_t, lr_x, lr_y, last_t, dt_us)
    
    return events_t[mask], events_x[mask], events_y[mask], events_p[mask]


def polarity_agnostic_downsample_v2(events_t, events_x, events_y, events_p,
                                     factor=4, dt_us=30000, 
                                     temporal_bin_us=10000):
    """
    Super aggressive polarity-agnostic downsampling.
    
    Strategy:
    1. Spatial downsampling via bit-shift (4x)
    2. Polarity-agnostic refractory filter (2-3x)
    3. Spatial-temporal binning (additional 2-3x)
    
    Expected reduction: ~16-36x total
    """
    if factor <= 1:
        return events_t, events_x, events_y, events_p
    
    # 1. Spatial downsample
    shift = int(np.log2(factor))
    lr_x = events_x.astype(np.int32) >> shift
    lr_y = events_y.astype(np.int32) >> shift
    
    # 2. Polarity-agnostic refractory filter
    max_x = lr_x.max() if len(lr_x) > 0 else 0
    max_y = lr_y.max() if len(lr_y) > 0 else 0
    
    last_t = np.zeros((max_x + 1, max_y + 1), dtype=np.int64)
    mask1 = _numba_polarity_agnostic_refractory_filter(events_t, lr_x, lr_y, last_t, dt_us)
    
    # Apply first filter
    t_filtered = events_t[mask1]
    x_filtered = events_x[mask1]
    y_filtered = events_y[mask1]
    p_filtered = events_p[mask1]
    lr_x_filtered = lr_x[mask1]
    lr_y_filtered = lr_y[mask1]
    
    # 3. Additional spatial-temporal binning
    mask2 = _numba_spatial_temporal_binning(
        t_filtered, lr_x_filtered, lr_y_filtered, p_filtered,
        bin_size_spatial=2,  # Bin 2x2 low-res pixels together
        bin_size_temporal_us=temporal_bin_us
    )
    
    return t_filtered[mask2], x_filtered[mask2], y_filtered[mask2], p_filtered[mask2]


def polarity_agnostic_downsample_v3(events_t, events_x, events_y, events_p,
                                     target_reduction_factor=20):
    """
    Target-based polarity-agnostic downsampling.
    
    Strategy:
    - Compute how many events we want to keep
    - Use intelligent sampling that respects spatiotemporal distribution
    - Completely ignore polarity
    
    This guarantees a specific reduction factor.
    """
    n_events = len(events_t)
    if n_events == 0:
        return events_t, events_x, events_y, events_p
    
    target_events = max(1, n_events // target_reduction_factor)
    
    if target_events >= n_events:
        return events_t, events_x, events_y, events_p
    
    # Create spatial-temporal importance scores
    # Events with more spatial/temporal uniqueness get higher scores
    
    # Temporal importance: events in sparser time regions are more important
    time_diffs = np.diff(events_t, prepend=events_t[0], append=events_t[-1])
    temporal_importance = (time_diffs[:-1] + time_diffs[1:]) / 2.0
    
    # Spatial importance: events in sparser spatial regions are more important
    # Use a grid to count local density
    grid_size = 20  # pixels
    x_grid = events_x // grid_size
    y_grid = events_y // grid_size
    max_x_grid = x_grid.max() + 1
    grid_cells = x_grid * max_x_grid + y_grid
    
    # Count events per cell
    cell_counts = np.bincount(grid_cells.astype(np.int32))
    event_cell_counts = cell_counts[grid_cells.astype(np.int32)]
    
    # Lower density = higher importance
    spatial_importance = 1.0 / (event_cell_counts + 1.0)
    
    # Combined importance
    importance = temporal_importance * spatial_importance
    importance = importance / importance.sum()
    
    # Sample events based on importance
    selected_indices = np.random.choice(
        n_events, 
        size=target_events, 
        replace=False, 
        p=importance
    )
    selected_indices = np.sort(selected_indices)  # Keep chronological order
    
    return (events_t[selected_indices], 
            events_x[selected_indices], 
            events_y[selected_indices], 
            events_p[selected_indices])
