#!/usr/bin/env python3
"""
Test script to verify rotation augmentation correctness across different resolutions.
Tests both square (DVSGesture) and non-square (HMDB/UCF101) resolutions.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.event_augmentation import rotate_events


def test_rotation_square_resolution():
    """Test all rotations work correctly for square resolution (DVSGesture: 128x128)"""
    print("Testing Square Resolution (128 × 128 - DVSGesture)")
    print("=" * 60)
    
    height, width = 128, 128
    
    # Create test events at known positions
    events_xy = np.array([
        [0, 0],           # Top-left corner
        [127, 0],         # Top-right corner
        [0, 127],         # Bottom-left corner
        [127, 127],       # Bottom-right corner
        [64, 64],         # Center
    ], dtype=np.float32)
    
    events_t = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    events_p = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    
    print(f"Original events:\n{events_xy}\n")
    
    for angle in [0, 90, 180, 270]:
        try:
            rot_xy, rot_t, rot_p = rotate_events(
                events_xy, events_t, events_p, angle, height, width
            )
            print(f"✅ {angle}° rotation: {len(rot_xy)} events preserved")
            print(f"   Sample rotated coords: {rot_xy[:2]}")
        except Exception as e:
            print(f"❌ {angle}° rotation FAILED: {e}")
    
    print()


def test_rotation_nonsquare_resolution():
    """Test rotation restrictions for non-square resolution (HMDB/UCF101: 180x240)"""
    print("Testing Non-Square Resolution (180 × 240 - HMDB/UCF101)")
    print("=" * 60)
    
    height, width = 180, 240
    
    # Create test events
    events_xy = np.array([
        [0, 0],           # Top-left corner
        [239, 0],         # Top-right corner
        [0, 179],         # Bottom-left corner
        [239, 179],       # Bottom-right corner
        [120, 90],        # Center
    ], dtype=np.float32)
    
    events_t = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    events_p = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    
    print(f"Original events:\n{events_xy}\n")
    
    # Test 0° and 180° (should work)
    for angle in [0, 180]:
        try:
            rot_xy, rot_t, rot_p = rotate_events(
                events_xy, events_t, events_p, angle, height, width
            )
            print(f"✅ {angle}° rotation: PASSED ({len(rot_xy)} events)")
            print(f"   Sample coords: {rot_xy[:2]}")
        except Exception as e:
            print(f"❌ {angle}° rotation FAILED (unexpected!): {e}")
    
    # Test 90° and 270° (should fail with clear error message)
    for angle in [90, 270]:
        try:
            rot_xy, rot_t, rot_p = rotate_events(
                events_xy, events_t, events_p, angle, height, width
            )
            print(f"❌ {angle}° rotation: INCORRECTLY ALLOWED (should have failed!)")
        except ValueError as e:
            print(f"✅ {angle}° rotation: Correctly blocked")
            print(f"   Error message: {str(e)[:100]}...")
        except Exception as e:
            print(f"⚠️  {angle}° rotation: Failed with unexpected error: {e}")
    
    print()


def test_180_degree_correctness():
    """Verify 180° rotation is mathematically correct"""
    print("Verifying 180° Rotation Correctness")
    print("=" * 60)
    
    for (height, width, name) in [(128, 128, "Square"), (180, 240, "Non-Square")]:
        print(f"\n{name} ({height} × {width}):")
        
        # Test specific points
        events_xy = np.array([
            [0, 0],                    # Corner
            [width-1, height-1],       # Opposite corner
            [width//2, height//2],     # Center
        ], dtype=np.float32)
        
        events_t = np.array([0, 1, 2], dtype=np.float32)
        events_p = np.array([1, 0, 1], dtype=np.float32)
        
        rot_xy, _, _ = rotate_events(events_xy, events_t, events_p, 180, height, width)
        
        # Expected: (x, y) -> (width-1-x, height-1-y)
        expected_xy = np.array([
            [width-1, height-1],
            [0, 0],
            [width//2, height//2],  # Center stays at center
        ], dtype=np.float32)
        
        matches = np.allclose(rot_xy, expected_xy, atol=0.1)
        
        if matches:
            print(f"  ✅ 180° rotation is mathematically correct")
        else:
            print(f"  ❌ 180° rotation is INCORRECT!")
            print(f"     Expected: {expected_xy}")
            print(f"     Got:      {rot_xy}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ROTATION AUGMENTATION VERIFICATION")
    print("=" * 60 + "\n")
    
    test_rotation_square_resolution()
    test_rotation_nonsquare_resolution()
    test_180_degree_correctness()
    
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
