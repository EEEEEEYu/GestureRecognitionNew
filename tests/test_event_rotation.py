"""
Unit tests for event rotation augmentation.

Tests rotation transformation correctness for event camera data.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.event_augmentation import (
    rotate_events,
    rotate_events_90deg,
    rotate_events_180deg,
    rotate_events_270deg
)


def test_rotate_90deg_correctness():
    """Test that 90° rotation transforms coordinates correctly."""
    print("Testing 90° rotation correctness...")
    
    # Create simple test events
    height, width = 128, 128
    events_xy = np.array([[64, 32], [10, 20], [100, 100]], dtype=np.int32)
    events_t = np.array([1000, 2000, 3000], dtype=np.float32)
    events_p = np.array([1, -1, 1], dtype=np.float32)
    
    # Apply 90° rotation: (x, y) -> (y, width - 1 - x)
    rotated_xy, rotated_t, rotated_p = rotate_events_90deg(
        events_xy, events_t, events_p, height, width
    )
    
    # Verify transformation
    expected_x = events_xy[:, 1]  # y becomes x
    expected_y = width - 1 - events_xy[:, 0]  # width - 1 - x becomes y
    
    assert len(rotated_xy) == len(events_xy), "Should preserve event count"
    assert np.allclose(rotated_xy[:, 0], expected_x), f"X coordinates incorrect: {rotated_xy[:, 0]} vs {expected_x}"
    assert np.allclose(rotated_xy[:, 1], expected_y), f"Y coordinates incorrect: {rotated_xy[:, 1]} vs {expected_y}"
    assert np.allclose(rotated_t, events_t), "Timestamps should be unchanged"
    assert np.allclose(rotated_p, events_p), "Polarities should be unchanged"
    
    print("  ✓ 90° rotation transforms correctly")


def test_rotate_180deg_symmetry():
    """Test that 180° rotation is equivalent to two 90° rotations."""
    print("Testing 180° rotation symmetry...")
    
    height, width = 128, 128
    events_xy = np.array([[64, 32], [10, 20], [100, 100]], dtype=np.int32)
    events_t = np.array([1000, 2000, 3000], dtype=np.float32)
    events_p = np.array([1, -1, 1], dtype=np.float32)
    
    # Method 1: Direct 180° rotation
    rotated_180, t1, p1 = rotate_events_180deg(
        events_xy, events_t, events_p, height, width
    )
    
    # Method 2: Two 90° rotations
    temp_xy, temp_t, temp_p = rotate_events_90deg(
        events_xy, events_t, events_p, height, width
    )
    rotated_90_90, t2, p2 = rotate_events_90deg(
        temp_xy, temp_t, temp_p, height, width
    )
    
    assert np.allclose(rotated_180, rotated_90_90), "180° should equal two 90° rotations"
    assert np.allclose(t1, t2), "Timestamps should match"
    assert np.allclose(p1, p2), "Polarities should match"
    
    print("  ✓ 180° rotation is symmetric")


def test_rotate_360deg_identity():
    """Test that four 90° rotations return to original position."""
    print("Testing 360° rotation (identity)...")
    
    height, width = 128, 128
    events_xy = np.array([[64, 32], [10, 20], [100, 100]], dtype=np.int32)
    events_t = np.array([1000, 2000, 3000], dtype=np.float32)
    events_p = np.array([1, -1, 1], dtype=np.float32)
    
    # Apply four 90° rotations
    xy = events_xy.copy()
    t = events_t.copy()
    p = events_p.copy()
    
    for i in range(4):
        xy, t, p = rotate_events_90deg(xy, t, p, height, width)
    
    assert np.allclose(xy, events_xy), "Four 90° rotations should return to original"
    assert np.allclose(t, events_t), "Timestamps should be unchanged"
    assert np.allclose(p, events_p), "Polarities should be unchanged"
    
    print("  ✓ 360° rotation returns to identity")


def test_out_of_bounds_filtering():
    """Test that events outside image bounds are discarded."""
    print("Testing out-of-bounds filtering...")
    
    height, width = 128, 128
    
    # Create events that will go out of bounds when rotated
    # Event at (0, 0) rotated 90° -> (0, 127) - should stay
    # Event at (127, 0) rotated 90° -> (0, 0) - should stay
    # Event at (0, 127) rotated 90° -> (127, 127) - should stay
    events_xy = np.array([[0, 0], [127, 0], [0, 127]], dtype=np.int32)
    events_t = np.array([1000, 2000, 3000], dtype=np.float32)
    events_p = np.array([1, -1, 1], dtype=np.float32)
    
    rotated_xy, rotated_t, rotated_p = rotate_events_90deg(
        events_xy, events_t, events_p, height, width
    )
    
    # All should be in bounds for these corner cases
    assert len(rotated_xy) == 3, "All corner events should remain in bounds"
    assert np.all(rotated_xy[:, 0] >= 0) and np.all(rotated_xy[:, 0] < width), "X in bounds"
    assert np.all(rotated_xy[:, 1] >= 0) and np.all(rotated_xy[:, 1] < height), "Y in bounds"
    
    print("  ✓ Out-of-bounds filtering works correctly")


def test_timestamp_preservation():
    """Test that timestamps are not modified by rotation."""
    print("Testing timestamp preservation...")
    
    height, width = 128, 128
    events_xy = np.array([[64, 32]], dtype=np.int32)
    events_t = np.array([123456.789], dtype=np.float32)
    events_p = np.array([1], dtype=np.float32)
    
    for angle in [0, 90, 180, 270]:
        rotated_xy, rotated_t, rotated_p = rotate_events(
            events_xy, events_t, events_p, angle, height, width
        )
        assert np.allclose(rotated_t, events_t), f"Timestamps should be preserved for {angle}°"
        assert np.allclose(rotated_p, events_p), f"Polarities should be preserved for {angle}°"
    
    print("  ✓ Timestamps and polarities preserved across all rotations")


def test_main_rotate_function():
    """Test the main rotate_events dispatcher function."""
    print("Testing main rotate_events function...")
    
    height, width = 128, 128
    events_xy = np.array([[64, 32]], dtype=np.int32)
    events_t = np.array([1000], dtype=np.float32)
    events_p = np.array([1], dtype=np.float32)
    
    # Test 0° (identity)
    rotated_0, _, _ = rotate_events(events_xy, events_t, events_p, 0, height, width)
    assert np.allclose(rotated_0, events_xy), "0° should return unchanged coordinates"
    
    # Test that unsupported angles raise error
    try:
        rotate_events(events_xy, events_t, events_p, 45, height, width)
        assert False, "Should raise ValueError for unsupported angle"
    except ValueError as e:
        assert "Unsupported rotation angle" in str(e)
    
    print("  ✓ Main rotate_events function works correctly")


def test_empty_events():
    """Test rotation with empty event arrays."""
    print("Testing empty event handling...")
    
    height, width = 128, 128
    events_xy = np.array([], dtype=np.int32).reshape(0, 2)
    events_t = np.array([], dtype=np.float32)
    events_p = np.array([], dtype=np.float32)
    
    for angle in [0, 90, 180, 270]:
        rotated_xy, rotated_t, rotated_p = rotate_events(
            events_xy, events_t, events_p, angle, height, width
        )
        assert len(rotated_xy) == 0, f"Empty events should remain empty for {angle}°"
        assert len(rotated_t) == 0
        assert len(rotated_p) == 0
    
    print("  ✓ Empty event arrays handled correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Event Rotation Unit Tests")
    print("=" * 60)
    print()
    
    try:
        test_rotate_90deg_correctness()
        test_rotate_180deg_symmetry()
        test_rotate_360deg_identity()
        test_out_of_bounds_filtering()
        test_timestamp_preservation()
        test_main_rotate_function()
        test_empty_events()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Unexpected error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
