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
    
    Uses integer arithmetic for exact transformation.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
        Out-of-bounds events are filtered out.
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 90° clockwise: (x, y) -> (y, width - 1 - x)
    # Note: For DVSGesture (128x128), this maps correctly
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = y
    new_y = width - 1 - x
    
    # Filter out-of-bounds events
    valid_mask = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
    
    rotated_xy = np.stack([new_x[valid_mask], new_y[valid_mask]], axis=1).astype(events_xy.dtype)
    rotated_t = events_t[valid_mask]
    rotated_p = events_p[valid_mask]
    
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
    
    Uses integer arithmetic for exact transformation.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
        Out-of-bounds events are filtered out.
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 180°: (x, y) -> (width - 1 - x, height - 1 - y)
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = width - 1 - x
    new_y = height - 1 - y
    
    # Filter out-of-bounds events
    valid_mask = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
    
    rotated_xy = np.stack([new_x[valid_mask], new_y[valid_mask]], axis=1).astype(events_xy.dtype)
    rotated_t = events_t[valid_mask]
    rotated_p = events_p[valid_mask]
    
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
    
    Uses integer arithmetic for exact transformation.
    
    Args:
        events_xy: Event coordinates [N, 2] with columns [x, y]
        events_t: Event timestamps [N]
        events_p: Event polarities [N]
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (rotated_xy, timestamps, polarities)
        Out-of-bounds events are filtered out.
    """
    if len(events_xy) == 0:
        return events_xy.copy(), events_t.copy(), events_p.copy()
    
    # 270° clockwise (90° counter-clockwise): (x, y) -> (height - 1 - y, x)
    x = events_xy[:, 0]
    y = events_xy[:, 1]
    
    new_x = height - 1 - y
    new_y = x
    
    # Filter out-of-bounds events
    valid_mask = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
    
    rotated_xy = np.stack([new_x[valid_mask], new_y[valid_mask]], axis=1).astype(events_xy.dtype)
    rotated_t = events_t[valid_mask]
    rotated_p = events_p[valid_mask]
    
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
    
    Currently supports cardinal rotations: 0°, 90°, 180°, 270°.
    
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
        raise ValueError(
            f"Unsupported rotation angle: {angle_deg}. "
            f"Only cardinal rotations (0, 90, 180, 270) are supported."
        )


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
