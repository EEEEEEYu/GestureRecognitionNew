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
