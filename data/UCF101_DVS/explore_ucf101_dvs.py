"""
Exploration script for UCF101-DVS dataset.

This dataset appears to contain preprocessed graph features stored in .mat files
rather than raw event data. This script examines the structure of these files.

Usage:
    python explore_ucf101_dvs.py --dataset_dir ~/Downloads/UCF101_DVS
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

# Try to import scipy for loading .mat files
try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Install with: pip install scipy")


def explore_dataset_structure(dataset_dir):
    """Explore the overall dataset structure."""
    
    dataset_dir = Path(dataset_dir).expanduser()
    
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Exploring UCF101-DVS Dataset Structure")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_dir}")
    
    # Find all action class directories
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"\nNumber of action classes: {len(class_dirs)}")
    
    # Count files per class
    file_counts = {}
    total_files = 0
    
    for class_dir in class_dirs:
        mat_files = list(class_dir.glob('*.mat'))
        file_counts[class_dir.name] = len(mat_files)
        total_files += len(mat_files)
    
    print(f"Total .mat files: {total_files}")
    
    print(f"\nAction classes and file counts:")
    for i, (class_name, count) in enumerate(sorted(file_counts.items())):
        print(f"  {i:3d}. {class_name:30s} ({count:4d} files)")
    
    # Check file naming pattern
    if class_dirs:
        sample_class = class_dirs[0]
        sample_files = list(sample_class.glob('*.mat'))[:5]
        if sample_files:
            print(f"\nSample filenames from '{sample_class.name}':")
            for f in sample_files:
                print(f"  {f.name}")
    
    return class_dirs


def explore_mat_file(file_path):
    """Explore the contents of a .mat file."""
    
    if not HAS_SCIPY:
        print("✗ Cannot load .mat file without scipy")
        return None
    
    print(f"\n{'='*60}")
    print(f"Exploring .mat file structure")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / 1024:.2f} KB")
    
    try:
        # Load .mat file
        mat_data = loadmat(str(file_path))
        
        # Filter out metadata keys (those starting with '__')
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        print(f"\nData keys: {data_keys}")
        
        # Examine each data field
        for key in data_keys:
            data = mat_data[key]
            print(f"\n--- Key: '{key}' ---")
            print(f"  Type: {type(data)}")
            
            if isinstance(data, np.ndarray):
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                
                # If it's numeric, show statistics
                if np.issubdtype(data.dtype, np.number):
                    print(f"  Min: {data.min()}")
                    print(f"  Max: {data.max()}")
                    print(f"  Mean: {data.mean():.4f}")
                    print(f"  Std: {data.std():.4f}")
                
                # Show a sample of the data
                if data.size < 100:
                    print(f"  Data preview:\n{data}")
                else:
                    print(f"  Data preview (first few elements):")
                    if data.ndim == 1:
                        print(f"    {data[:10]}")
                    elif data.ndim == 2:
                        print(f"    {data[:5, :min(10, data.shape[1])]}")
                    else:
                        print(f"    Shape too complex, showing flattened first 20: {data.flat[:20]}")
            
            elif isinstance(data, dict):
                print(f"  Nested dict with keys: {list(data.keys())}")
            
            else:
                print(f"  Value: {data}")
        
        return mat_data
        
    except Exception as e:
        print(f"✗ Error loading .mat file: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_dataset_format(dataset_dir):
    """Analyze whether this is graph features or raw events."""
    
    dataset_dir = Path(dataset_dir).expanduser()
    
    print(f"\n{'='*60}")
    print(f"Dataset Format Analysis")
    print(f"{'='*60}")
    
    # Sample a few files from different classes
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    if not class_dirs:
        print("No class directories found!")
        return
    
    sample_files = []
    for class_dir in class_dirs[:3]:  # Sample from first 3 classes
        mat_files = list(class_dir.glob('*.mat'))
        if mat_files:
            sample_files.append(mat_files[0])
    
    if not sample_files or not HAS_SCIPY:
        print("Cannot analyze format without sample files or scipy")
        return
    
    print(f"Analyzing {len(sample_files)} sample files...\n")
    
    common_keys = None
    all_shapes = defaultdict(list)
    
    for sample_file in sample_files:
        try:
            mat_data = loadmat(str(sample_file))
            data_keys = set([k for k in mat_data.keys() if not k.startswith('__')])
            
            if common_keys is None:
                common_keys = data_keys
            else:
                common_keys = common_keys.intersection(data_keys)
            
            # Collect shapes
            for key in data_keys:
                if isinstance(mat_data[key], np.ndarray):
                    all_shapes[key].append(mat_data[key].shape)
        
        except Exception as e:
            print(f"  Warning: Could not load {sample_file.name}: {e}")
    
    print(f"Common keys across all files: {sorted(common_keys) if common_keys else 'None'}")
    
    print(f"\nShape summary:")
    for key in sorted(all_shapes.keys()):
        shapes = all_shapes[key]
        print(f"  '{key}': {shapes}")
    
    # Determine if this looks like graph features or raw events
    print(f"\n{'='*60}")
    print(f"Format Determination:")
    print(f"{'='*60}")
    
    if common_keys:
        has_graph_keywords = any(keyword in str(common_keys).lower() 
                                for keyword in ['graph', 'edge', 'node', 'feature', 'adj'])
        has_event_keywords = any(keyword in str(common_keys).lower() 
                                for keyword in ['event', 'x', 'y', 't', 'p', 'polarity'])
        
        if has_graph_keywords:
            print("✓ This appears to be GRAPH FEATURES (preprocessed)")
            print("  - Data is likely already encoded as graph representations")
            print("  - Not suitable for standard event-based preprocessing")
        elif has_event_keywords:
            print("✓ This appears to be RAW EVENTS")
            print("  - Data contains raw event coordinates")
            print("  - Can be processed like HMDB-DVS")
        else:
            print("? Format unclear - examine sample file manually")
            print(f"  Keys found: {common_keys}")


def main():
    parser = argparse.ArgumentParser(description='Explore UCF101-DVS dataset structure')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='~/Downloads/UCF101_DVS',
        help='Path to UCF101-DVS dataset directory'
    )
    parser.add_argument(
        '--sample_file',
        type=str,
        default=None,
        help='Specific .mat file to analyze (optional)'
    )
    
    args = parser.parse_args()
    
    if not HAS_SCIPY:
        print("\n✗ scipy is required to read .mat files")
        print("Install with: pip install scipy")
        sys.exit(1)
    
    # Explore dataset structure
    class_dirs = explore_dataset_structure(args.dataset_dir)
    
    if class_dirs is None:
        return
    
    # Find a sample file
    if args.sample_file:
        sample_file = Path(args.sample_file).expanduser()
    else:
        # Use first .mat file from first class
        sample_file = next(class_dirs[0].glob('*.mat'), None)
    
    if sample_file and sample_file.exists():
        explore_mat_file(sample_file)
    else:
        print(f"\n✗ No sample file found")
    
    # Analyze format
    analyze_dataset_format(args.dataset_dir)
    
    print(f"\n{'='*60}")
    print("Exploration complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
