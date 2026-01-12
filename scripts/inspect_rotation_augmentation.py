"""
Quick inspection script to verify rotation augmentation in preprocessed HDF5 files.

Usage:
    python3 scripts/inspect_rotation_augmentation.py <path_to_h5_file>
"""

import sys
import h5py
import numpy as np


def inspect_hdf5_rotation(h5_path):
    """Inspect rotation augmentation in HDF5 file."""
    print("=" * 60)
    print(f"Inspecting: {h5_path}")
    print("=" * 60)
    print()
    
    with h5py.File(h5_path, 'r') as h5f:
        # Read metadata
        num_samples = h5f['labels'].shape[0]
        rotation_enabled = h5f.attrs.get('rotation_enabled', False)
        rotation_angles_list = list(h5f.attrs.get('rotation_angles_list', [0]))
        
        print(f"Total samples: {num_samples}")
        print(f"Rotation augmentation: {'ENABLED' if rotation_enabled else 'DISABLED'}")
        print(f"Configured rotation angles: {rotation_angles_list}")
        print()
        
        if 'rotation_angles' in h5f:
            rotation_angles = h5f['rotation_angles'][:]
            
            # Count distribution
            unique_angles, counts = np.unique(rotation_angles, return_counts=True)
            
            print("Rotation Distribution:")
            for angle, count in zip(unique_angles, counts):
                percentage = (count / num_samples) * 100
                print(f"  {angle:3d}°: {count:5d} samples ({percentage:5.1f}%)")
            print()
            
            # Show first 10 samples
            print("First 10 samples:")
            print(f"{'Index':<8} {'Label':<8} {'Rotation':<10} {'#Intervals':<12}")
            print("-" * 40)
            for i in range(min(10, num_samples)):
                label = h5f['labels'][i]
                rotation = rotation_angles[i]
                num_intervals = h5f['num_intervals'][i]
                print(f"{i:<8} {label:<8} {rotation:3d}°{'':<6} {num_intervals:<12}")
            
            if num_samples > 10:
                print(f"... ({num_samples - 10} more samples)")
        else:
            print("⚠ No rotation_angles dataset found (old format or disabled)")
        
        print()
        print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/inspect_rotation_augmentation.py <path_to_h5_file>")
        print()
        print("Example:")
        print("  python3 scripts/inspect_rotation_augmentation.py precomputed_data/dvsgesture/train.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    inspect_hdf5_rotation(h5_path)
