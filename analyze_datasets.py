"""
Analyze and compare DVSGesture and Custom Gesture precomputed datasets.

This script:
1. Analyzes filtering logic for dynamic background sequences
2. Verifies downsample logic correctness
3. Compares statistics between DVSGesture and Custom Gesture:
   - Sequence length (number of intervals)
   - Vectors per sample
   - Vectors per interval
   - Event counts and densities
"""

import h5py
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json


def analyze_filtering_logic():
    """Check the filtering logic implementation for dynamic background sequences."""
    print("=" * 80)
    print("1. FILTERING LOGIC ANALYSIS")
    print("=" * 80)
    
    # Read the config to see what filter is applied
    import yaml
    config_path = '/fs/nexus-scratch/haowenyu/GestureRecognitionNew/configs/custom_gesture_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    filter_config = config['PRECOMPUTING']['filter']
    print(f"\nCurrent filter configuration:")
    print(f"  View: {filter_config['view']}")
    print(f"  Lighting: {filter_config['lighting']}")
    print(f"  Background: {filter_config['background']}")
    
    if filter_config['background'] == 'STATIC':
        print("\n✓ Filter is correctly set to STATIC (removes DYNAMIC backgrounds)")
    elif filter_config['background'] == 'DYNAMIC':
        print("\n⚠ WARNING: Filter is set to DYNAMIC (keeps only dynamic backgrounds)")
    elif filter_config['background'] in ['both', 'all']:
        print("\n⚠ WARNING: Filter is set to 'both' (no filtering applied)")
    
    # Test the filtering function logic
    print("\n" + "-" * 80)
    print("Testing filter_sequences logic:")
    print("-" * 80)
    
    # Import and test the parse_sequence_name function
    import sys
    sys.path.insert(0, '/fs/nexus-scratch/haowenyu/GestureRecognitionNew')
    from preprocess_custom_gesture_ultra import parse_sequence_name
    
    # Test cases
    test_sequences = [
        "sequence_haowen1_TOP_STATIC_LIGHT_wine_glass",
        "sequence_haowen1_SIDE_DYNAMIC_DARK_bottle",
        "sequence_haowen2_TOP_STATIC_DARK_coffee_mug",
        "sequence_haowen2_SIDE_DYNAMIC_LIGHT_pasta_server",
    ]
    
    print("\nTest sequence parsing:")
    for seq in test_sequences:
        metadata = parse_sequence_name(seq)
        print(f"  {seq}")
        print(f"    → person: {metadata['person_id']}, view: {metadata['view']}, "
              f"background: {metadata['background']}, lighting: {metadata['lighting']}, "
              f"class: {metadata['class_name']}")
        
        # Check if it would pass the filter
        if filter_config['background'] == 'STATIC':
            passes = metadata['background'] == 'STATIC'
            print(f"    → {'✓ PASSES' if passes else '✗ FILTERED OUT'} (background={metadata['background']})")
    
    print("\n" + "=" * 80)


def analyze_downsample_logic():
    """Verify the downsample logic correctness."""
    print("\n" + "=" * 80)
    print("2. DOWNSAMPLE LOGIC ANALYSIS")
    print("=" * 80)
    
    # Read the downsample config
    import yaml
    config_path = '/fs/nexus-scratch/haowenyu/GestureRecognitionNew/configs/custom_gesture_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    downsample_config = config['PRECOMPUTING']['downsample']
    print(f"\nDownsample configuration:")
    print(f"  Enabled: {downsample_config['enabled']}")
    print(f"  Factor: {downsample_config['factor']}")
    print(f"  dt_us (refractory period): {downsample_config['dt_us']} microseconds")
    
    original_height = config['PRECOMPUTING']['height']
    original_width = config['PRECOMPUTING']['width']
    factor = downsample_config['factor']
    
    print(f"\nSpatial downsampling:")
    print(f"  Original resolution: {original_width} × {original_height}")
    print(f"  Downsampled resolution: {original_width // factor} × {original_height // factor}")
    print(f"  Reduction factor: {factor}x")
    
    # Check the bit-shift logic
    print(f"\nBit-shift verification (factor={factor}):")
    shift = int(np.log2(factor))
    print(f"  Shift amount: {shift} bits")
    print(f"  Example: pixel (640, 480) → ({640 >> shift}, {480 >> shift})")
    
    # Verify bit-shift matches division
    test_coords = [(640, 480), (320, 240), (160, 120), (639, 479)]
    print(f"\n  Test coordinates:")
    for x, y in test_coords:
        lr_x_shift = x >> shift
        lr_y_shift = y >> shift
        lr_x_div = x // factor
        lr_y_div = y // factor
        match = (lr_x_shift == lr_x_div) and (lr_y_shift == lr_y_div)
        print(f"    ({x:3d}, {y:3d}) → shift: ({lr_x_shift:3d}, {lr_y_shift:3d}), "
              f"division: ({lr_x_div:3d}, {lr_y_div:3d}) {'✓' if match else '✗ MISMATCH!'}")
    
    print(f"\nRefractory filtering:")
    print(f"  Each pixel can fire at most once every {downsample_config['dt_us']} μs")
    print(f"  This preserves temporal geometry while reducing event rate")
    
    # Check if the grid size is correct
    grid_w = (original_width + factor - 1) // factor
    grid_h = (original_height + factor - 1) // factor
    print(f"\nGrid allocation:")
    print(f"  Grid size: {grid_w} × {grid_h}")
    print(f"  Expected size: {original_width // factor} × {original_height // factor}")
    if grid_w == original_width // factor and grid_h == original_height // factor:
        print(f"  ✓ Grid size matches expected downsampled resolution")
    else:
        print(f"  ⚠ Grid size differs (ceil division for safety)")
    
    print("\n" + "=" * 80)


def compare_dataset_statistics():
    """Compare statistics between DVSGesture and Custom Gesture datasets."""
    print("\n" + "=" * 80)
    print("3. DATASET STATISTICS COMPARISON")
    print("=" * 80)
    
    dvsgesture_path = '/fs/nexus-scratch/haowenyu/GestureRecognitionNew/precomputed_data/dvsgesture'
    custom_gesture_path = '/fs/nexus-projects/DVS_Actions/precomputed_data/custom_gesture_downsampled_dynamic_removed'
    
    datasets = {
        'DVSGesture': dvsgesture_path,
        'CustomGesture': custom_gesture_path,
    }
    
    stats = {}
    
    for dataset_name, base_path in datasets.items():
        print(f"\n{'-' * 80}")
        print(f"Analyzing {dataset_name}")
        print(f"{'-' * 80}")
        
        stats[dataset_name] = {}
        
        for split in ['train', 'validation']:
            h5_path = os.path.join(base_path, f'{split}.h5')
            
            if not os.path.exists(h5_path):
                print(f"  ⚠ {split}.h5 not found at {h5_path}")
                continue
            
            print(f"\n{split.upper()} split:")
            
            with h5py.File(h5_path, 'r') as h5f:
                num_samples = h5f['labels'].shape[0]
                num_intervals_array = h5f['num_intervals'][:]
                
                # Collect statistics
                total_intervals = num_intervals_array.sum()
                mean_intervals = num_intervals_array.mean()
                std_intervals = num_intervals_array.std()
                min_intervals = num_intervals_array.min()
                max_intervals = num_intervals_array.max()
                
                # Count total vectors
                total_vectors = 0
                vectors_per_sample = []
                vectors_per_interval_all = []
                
                for idx in range(num_samples):
                    sample_group = h5f[f'sample_{idx:06d}']
                    num_vectors_per_interval = sample_group['num_vectors_per_interval'][:]
                    
                    sample_total_vectors = num_vectors_per_interval.sum()
                    total_vectors += sample_total_vectors
                    vectors_per_sample.append(sample_total_vectors)
                    vectors_per_interval_all.extend(num_vectors_per_interval.tolist())
                
                vectors_per_sample = np.array(vectors_per_sample)
                vectors_per_interval_all = np.array(vectors_per_interval_all)
                
                # Calculate statistics
                avg_vectors_per_sample = vectors_per_sample.mean()
                std_vectors_per_sample = vectors_per_sample.std()
                
                avg_vectors_per_interval = vectors_per_interval_all.mean()
                std_vectors_per_interval = vectors_per_interval_all.std()
                median_vectors_per_interval = np.median(vectors_per_interval_all)
                
                # Get metadata
                encoding_dim = h5f.attrs.get('encoding_dim', 'N/A')
                height = h5f.attrs.get('height', 'N/A')
                width = h5f.attrs.get('width', 'N/A')
                ratio_of_vectors = h5f.attrs.get('ratio_of_vectors', 'N/A')
                accumulation_interval_ms = h5f.attrs.get('accumulation_interval_ms', 'N/A')
                
                # Print results
                print(f"  Metadata:")
                print(f"    Resolution: {width} × {height}")
                print(f"    Encoding dim: {encoding_dim}")
                print(f"    Accumulation interval: {accumulation_interval_ms} ms")
                print(f"    Ratio of vectors: {ratio_of_vectors}")
                
                print(f"\n  Sample statistics:")
                print(f"    Total samples: {num_samples}")
                print(f"    Total intervals: {total_intervals}")
                print(f"    Total vectors: {total_vectors}")
                
                print(f"\n  Intervals per sample:")
                print(f"    Mean: {mean_intervals:.2f} ± {std_intervals:.2f}")
                print(f"    Range: [{min_intervals}, {max_intervals}]")
                
                print(f"\n  Vectors per sample:")
                print(f"    Mean: {avg_vectors_per_sample:.2f} ± {std_vectors_per_sample:.2f}")
                print(f"    Range: [{vectors_per_sample.min():.0f}, {vectors_per_sample.max():.0f}]")
                
                print(f"\n  Vectors per interval:")
                print(f"    Mean: {avg_vectors_per_interval:.2f} ± {std_vectors_per_interval:.2f}")
                print(f"    Median: {median_vectors_per_interval:.2f}")
                print(f"    Range: [{vectors_per_interval_all.min():.0f}, {vectors_per_interval_all.max():.0f}]")
                
                # Check for zero-vector intervals
                zero_intervals = (vectors_per_interval_all == 0).sum()
                if zero_intervals > 0:
                    print(f"    ⚠ WARNING: {zero_intervals} intervals with 0 vectors "
                          f"({zero_intervals/len(vectors_per_interval_all)*100:.2f}%)")
                
                # Store for comparison
                stats[dataset_name][split] = {
                    'num_samples': num_samples,
                    'total_intervals': total_intervals,
                    'total_vectors': total_vectors,
                    'mean_intervals_per_sample': mean_intervals,
                    'mean_vectors_per_sample': avg_vectors_per_sample,
                    'mean_vectors_per_interval': avg_vectors_per_interval,
                    'median_vectors_per_interval': median_vectors_per_interval,
                    'resolution': (width, height),
                    'encoding_dim': encoding_dim,
                    'ratio_of_vectors': ratio_of_vectors,
                }
    
    # Comparative analysis
    print(f"\n{'=' * 80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'=' * 80}")
    
    if 'DVSGesture' in stats and 'CustomGesture' in stats:
        for split in ['train', 'validation']:
            if split in stats['DVSGesture'] and split in stats['CustomGesture']:
                print(f"\n{split.upper()} split comparison:")
                
                dvs = stats['DVSGesture'][split]
                custom = stats['CustomGesture'][split]
                
                print(f"  Samples: DVS={dvs['num_samples']}, Custom={custom['num_samples']}")
                
                print(f"\n  Intervals per sample:")
                print(f"    DVS: {dvs['mean_intervals_per_sample']:.2f}")
                print(f"    Custom: {custom['mean_intervals_per_sample']:.2f}")
                print(f"    Ratio (Custom/DVS): {custom['mean_intervals_per_sample']/dvs['mean_intervals_per_sample']:.2f}x")
                
                print(f"\n  Vectors per sample:")
                print(f"    DVS: {dvs['mean_vectors_per_sample']:.2f}")
                print(f"    Custom: {custom['mean_vectors_per_sample']:.2f}")
                print(f"    Ratio (Custom/DVS): {custom['mean_vectors_per_sample']/dvs['mean_vectors_per_sample']:.2f}x")
                
                print(f"\n  Vectors per interval:")
                print(f"    DVS: {dvs['mean_vectors_per_interval']:.2f}")
                print(f"    Custom: {custom['mean_vectors_per_interval']:.2f}")
                print(f"    Ratio (Custom/DVS): {custom['mean_vectors_per_interval']/dvs['mean_vectors_per_interval']:.2f}x")
                
                print(f"\n  Resolution:")
                print(f"    DVS: {dvs['resolution'][0]} × {dvs['resolution'][1]}")
                print(f"    Custom: {custom['resolution'][0]} × {custom['resolution'][1]}")
                
                print(f"\n  Preprocessing settings:")
                print(f"    DVS ratio_of_vectors: {dvs['ratio_of_vectors']}")
                print(f"    Custom ratio_of_vectors: {custom['ratio_of_vectors']}")
    
    print("\n" + "=" * 80)


def generate_recommendations():
    """Generate recommendations based on the analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nBased on the analysis, here are potential issues and recommendations:")
    
    print("\n1. If vectors per interval are significantly higher in Custom dataset:")
    print("   → Consider increasing ratio_of_vectors downsampling (currently 0.05)")
    print("   → Or increase train_ratio_of_vectors in training config (currently 0.5)")
    
    print("\n2. If sequence length (intervals) is much longer:")
    print("   → Model may struggle with longer temporal dependencies")
    print("   → Consider increasing num_layers or d_state in the model")
    print("   → Or reduce accumulation_interval_ms to create shorter sequences")
    
    print("\n3. If downsampling logic has issues:")
    print("   → Check that bit-shift matches division (verified above)")
    print("   → Ensure refractory period (dt_us) is appropriate for your event rate")
    print("   → Verify grid size covers all downsampled coordinates")
    
    print("\n4. If filtering logic has issues:")
    print("   → Ensure parse_sequence_name correctly extracts metadata")
    print("   → Verify filter configuration matches your intent")
    print("   → Check that all sequences have the expected naming format")
    
    print("\n5. If many intervals have 0 vectors:")
    print("   → This indicates very sparse events after downsampling")
    print("   → Consider reducing downsampling factor or dt_us")
    print("   → Or handle empty intervals in the model/dataloader")
    
    print("\n" + "=" * 80)


def main():
    """Run all analyses."""
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS AND COMPARISON")
    print("=" * 80)
    
    # 1. Check filtering logic
    analyze_filtering_logic()
    
    # 2. Verify downsample logic
    analyze_downsample_logic()
    
    # 3. Compare statistics
    compare_dataset_statistics()
    
    # 4. Generate recommendations
    generate_recommendations()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
