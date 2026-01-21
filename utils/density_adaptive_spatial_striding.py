import numpy as np
import math

def adaptive_spatial_sampling(
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    kernel_size: int = 17,
    overlap_factor: float = 0.0,
    sort_by_time: bool = True
):
    """
    Kernel-Aware Adaptive Spatial Striding.
    
    Selects a subset of events such that they cover the spatial domain with 
    kernels of size `kernel_size` and a specified `overlap_factor`.
    
    User Questions Addressed:
    1. Sparsity: This method uses spatial hashing on *existing* events only. 
       It does NOT scan the entire image, so it is efficient for sparse data.
    2. Density Focus: Increasing `overlap_factor` reduces the stride, creating 
       a finer grid. This captures more detail in dense/active regions while 
       naturally ignoring empty regions.
    
    Args:
        t, y, x, p: Event arrays.
        height, width: Image dimensions.
        kernel_size: The size of the receptive field (e.g., 17 for 17x17 kernel).
        overlap_factor: 0.0 to <1.0. 
                        0.0 = Tiling (stride = kernel_size). 
                        0.5 = Half-overlap (stride = kernel_size / 2).
                        Higher = More vectors, finer granularity.
        sort_by_time: If True, picks the most recent event in each grid cell.
                      If False, picks the first event found (not recommended).
                      
    Returns:
        indices: Indices of the selected events.
    """
    num_events = len(t)
    if num_events == 0:
        return np.array([], dtype=np.int64)
    
    # 1. Calculate Stride
    # Stride is the distance between kernel centers.
    # Stride = Kernel_Size * (1 - Overlap)
    # limit overlap to reasonable max (e.g. 0.9) to prevent stride=0
    safe_overlap = min(max(overlap_factor, 0.0), 0.9)
    stride = max(1, int(kernel_size * (1 - safe_overlap)))
    
    # 2. Spatial Hashing (Block ID calculation)
    # We map events to grid cells of size (stride x stride).
    # Events in the same cell compete to be the "representative" for that local area.
    y_block = (y // stride).astype(np.int64)
    x_block = (x // stride).astype(np.int64)
    
    # Block dimensions (for unique key generation)
    w_block = (width + stride - 1) // stride
    
    # Unique Key: y_block * width_in_blocks + x_block
    # ensuring int64 to prevent overflow
    block_keys = y_block * w_block + x_block
    
    # 3. Selection Strategy
    # We want ONE event per block.
    # Strategy: Pick the NEWEST event in the block.
    # Why? Old events fade. New events represent current state.
    
    if sort_by_time:
        # Sort by Key (ASC) then Time (DESC)
        # implementation: lexsort keys, -t
        # Note: lexsort uses the last argument as primary key.
        # So we pass keys as last arg.
        # To sort Time DESC, we use -t.
        
        # Casting t to potentially float64 for safety if needed, but usually fine.
        # If t is unsigned, negating might be an issue, but usually t is float or int32/64.
        sort_idx = np.lexsort((-t, block_keys))
    else:
        # Just sort by keys to group them
        sort_idx = np.argsort(block_keys)
        
    sorted_keys = block_keys[sort_idx]
    
    # 4. Filter Unique Blocks
    # np.unique with return_index=True gives the index of the *first* occurrence 
    # of each unique value in the sorted array.
    # Since we sorted by Key, Time_DESC, the "first" occurrence is the NEWEST event.
    _, unique_indices_in_sorted = np.unique(sorted_keys, return_index=True)
    
    # Map back to original indices
    selected_indices = sort_idx[unique_indices_in_sorted]
    
    # 5. Temporal Sort (Optional but good for downstream processing)
    selected_indices.sort()
    
    return selected_indices
