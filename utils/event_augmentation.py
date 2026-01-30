"""
Event augmentation utilities for event camera data.

This module provides transformation functions for augmenting event data,
specifically rotation transformations for precomputing augmented datasets.
"""

import numpy as np
from typing import Tuple


def rotate_events_90deg(
    events_xy: np.ndarray,
    events_t: np.ndarray,
    events_p: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate events 90 degrees clockwise around image center.
    
    For non-square resolutions, coordinates are clipped to fit within the original
    resolution bounds, maintaining the encoder's expected dimensions.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 90° clockwise: (x, y) -> (y, width - 1 - x)
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = y
    new_y = width - 1 - x
    
    # Clip coordinates to fit within the ORIGINAL resolution (width × height)
    # This maintains the encoder's expected resolution for non-square images
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    rotated_xy = np.stack([new_x, new_y], axis=1).astype(events_xy.dtype)
    rotated_t = events_t.copy()
    rotated_p = events_p.copy()
    
    return rotated_xy, rotated_t, rotated_p


def rotate_events_180deg(
    events_xy: np.ndarray,
    events_t: np.ndarray,
    events_p: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate events 180 degrees around image center.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 180°: (x, y) -> (width - 1 - x, height - 1 - y)
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = width - 1 - x
    new_y = height - 1 - y
    
    # Clip to ensure coordinates stay within bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    rotated_xy = np.stack([new_x, new_y], axis=1).astype(events_xy.dtype)
    rotated_t = events_t.copy()
    rotated_p = events_p.copy()
    
    return rotated_xy, rotated_t, rotated_p


def rotate_events_270deg(
    events_xy: np.ndarray,
    events_t: np.ndarray,
    events_p: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate events 270 degrees clockwise (90 degrees counter-clockwise) around image center.
    
    For non-square resolutions, coordinates are clipped to fit within the original
    resolution bounds, maintaining the encoder's expected dimensions.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 270° clockwise (90° counter-clockwise): (x, y) -> (height - 1 - y, x)
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = height - 1 - y
    new_y = x
    
    # Clip coordinates to fit within the ORIGINAL resolution (width × height)
    new_x =np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    rotated_xy = np.stack([new_x, new_y], axis=1).astype(events_xy.dtype)
    rotated_t = events_t.copy()
    rotated_p = events_p.copy()
    
    return rotated_xy, rotated_t, rotated_p


def rotate_events(
    events_xy: np.ndarray,
    events_t: np.ndarray,
    events_p: np.ndarray,
    angle_deg: int,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate events by specified angle around image center.
    
    Supports cardinal rotations: 0°, 90°, 180°, 270°.
    
    For non-square resolutions, rotated events are clipped to fit within the original
    resolution bounds (width × height). This maintains the VecKM encoder's expected
    resolution while allowing all rotation angles.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        angle_deg: Rotation angle in degrees (0, 90, 180, or 270)
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
        
    Raises:
        ValueError: If angle is not a supported cardinal angle
    """
    # Normalize angle to [0, 360)
    angle_deg = angle_deg % 360
    
    if angle_deg == 0:
        # No rotation - return copies
        return events_xy.copy(), events_t.copy(), events_p.copy()
    elif angle_deg == 90:
        return rotate_events_90deg(events_xy, events_t, events_p, height, width)
    elif angle_deg == 180:
        return rotate_events_180deg(events_xy, events_t, events_p, height, width)
    elif angle_deg == 270:
        return rotate_events_270deg(events_xy, events_t, events_p, height, width)
    else:
        # For non-cardinal angles, use arbitrary rotation
        return rotate_events_arbitrary(events_xy, events_t, events_p, angle_deg, height, width)


def rotate_sliced_events(
    events_xy_sliced: list,
    events_t_sliced: list,
    events_p_sliced: list,
    angle_deg: int,
    height: int,
    width: int
) -> Tuple[list, list, list]:
    """
    Rotate a list of event slices (intervals).
    
    This is a convenience function for rotating all intervals in a sequence.
    
    Args:
        events_xy_sliced: List of event coordinate arrays
        events_t_sliced: List of event timestamp arrays
        events_p_sliced: List of event polarity arrays
        angle_deg: Rotation angle in degrees
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy_sliced, rotated_t_sliced, rotated_p_sliced)
    """
    rotated_xy_sliced = []
    rotated_t_sliced = []
    rotated_p_sliced = []
    
    for xy, t, p in zip(events_xy_sliced, events_t_sliced, events_p_sliced):
        rot_xy, rot_t, rot_p = rotate_events(xy, t, p, angle_deg, height, width)
        rotated_xy_sliced.append(rot_xy)
        rotated_t_sliced.append(rot_t)
        rotated_p_sliced.append(rot_p)
    
    return rotated_xy_sliced, rotated_t_sliced, rotated_p_sliced


def rotate_events_arbitrary(
    events_xy: np.ndarray,
    events_t: np.ndarray,
    events_p: np.ndarray,
    angle_deg: float,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate events by an ARBITRARY angle around image center.
    
    Uses vectorized NumPy operations and rotation matrix.
    Note: Arbitrary rotation introduces quantization error as float coordinates
    must be rounded back to integers.
    
    Args:
        events_xy: Event coordinates [N, 2]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        angle_deg: Rotation angle in degrees
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
        
    # Convert to radians
    theta = np.deg2rad(angle_deg)
    
    # Rotation Matrix
    # [ cos -sin ]
    # [ sin  cos ]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    # Center of rotation
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    
    # 1. Center the coordinates
    coords = events_xy.astype(np.float32)
    coords[:, 0] -= center_x
    coords[:, 1] -= center_y
    
    # 2. Rotate (Matrix Multiplication)
    # R is (2, 2), coords is (N, 2). New coords = coords @ R.T
    coords_rotated = coords @ R.T
    
    # 3. Un-center
    coords_rotated[:, 0] += center_x
    coords_rotated[:, 1] += center_y
    
    # 4. Round and Clip
    # This step is destructive (quantization)
    coords_rotated = np.round(coords_rotated).astype(np.int32)
    
    # Clip to bounds
    coords_rotated[:, 0] = np.clip(coords_rotated[:, 0], 0, width - 1)
    coords_rotated[:, 1] = np.clip(coords_rotated[:, 1], 0, height - 1)
    
    return coords_rotated, events_t.copy(), events_p.copy()


def scale_time_sliced_events(
    events_xy_sliced: list,
    events_t_sliced: list,
    events_p_sliced: list,
    scale_factor: float,
    accumulation_interval_ms: float
) -> Tuple[list, list, list]:
    """
    Apply time scaling to sliced events.
    
    This function:
    1. Flattens the sliced events into a continuous stream.
    2. Scales the timestamps by scale_factor (t_new = t_old * scale).
    3. Re-slices the events into intervals based on the fixed accumulation_interval_ms.
    
    Args:
        events_xy_sliced: List of event coordinate arrays
        events_t_sliced: List of event timestamp arrays
        events_p_sliced: List of event polarity arrays
        scale_factor: Scaling factor (e.g. 1.1 for slowing down, 0.9 for speeding up)
        accumulation_interval_ms: The fixed size of one time bin in milliseconds
        
    Returns:
        Tuple of (new_xy_sliced, new_t_sliced, new_p_sliced)
    """
    if scale_factor == 1.0 or scale_factor <= 0:
         return events_xy_sliced, events_t_sliced, events_p_sliced

    # 1. Flatten
    if len(events_t_sliced) == 0:
        return events_xy_sliced, events_t_sliced, events_p_sliced
        
    all_xy = np.concatenate(events_xy_sliced, axis=0)
    all_t = np.concatenate(events_t_sliced, axis=0)
    all_p = np.concatenate(events_p_sliced, axis=0)
    
    if len(all_t) == 0:
        return events_xy_sliced, events_t_sliced, events_p_sliced
        
    t_start = all_t[0]
    all_t_rel = all_t - t_start
    
    # 2. Scale
    all_t_scaled = all_t_rel * scale_factor
    all_t_new = all_t_scaled + t_start
    
    # 3. Re-slice
    interval_us = accumulation_interval_ms * 1000.0
    
    # Boundaries: t_start, t_start+dt, t_start+2dt, ...
    # Determine end time
    t_end = all_t_new[-1]
    
    boundaries = np.arange(t_start, t_end + interval_us, interval_us)
    indices = np.searchsorted(all_t_new, boundaries)
    
    new_xy_sliced = []
    new_t_sliced = []
    new_p_sliced = []
    
    for i in range(len(indices) - 1):
        idx0 = indices[i]
        idx1 = indices[i+1]
        
        if idx1 > idx0:
            new_xy_sliced.append(all_xy[idx0:idx1])
            new_t_sliced.append(all_t_new[idx0:idx1])
            new_p_sliced.append(all_p[idx0:idx1])
            
    return new_xy_sliced, new_t_sliced, new_p_sliced
