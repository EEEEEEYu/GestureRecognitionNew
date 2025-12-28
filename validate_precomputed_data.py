#!/usr/bin/env python3
"""
Validate precomputed HDF5 data for integrity.

Checks:
1. Complex tensors are not all zeros
2. No NaN or Inf values
3. Reasonable value ranges
4. Data consistency across samples
"""

import h5py
import numpy as np
import argparse
from tqdm import tqdm
import os


def validate_sample(h5f, sample_idx, verbose=False):
    """Validate a single sample for data integrity."""
    issues = []
    
    sample_group = h5f[f'sample_{sample_idx:06d}']
    num_intervals = len([k for k in sample_group.keys() if k.startswith('interval_')])
    
    total_vectors = 0
    all_magnitudes = []
    
    for interval_idx in range(num_intervals):
        interval_group = sample_group[f'interval_{interval_idx:03d}']
        
        # Load data
        real_part = interval_group['real'][:]
        imag_part = interval_group['imag'][:]
        event_coords = interval_group['event_coords'][:]
        
        num_vectors = real_part.shape[0]
        total_vectors += num_vectors
        
        if num_vectors == 0:
            continue  # Skip empty intervals
        
        # Check for NaN
        if np.isnan(real_part).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: NaN in real part")
        if np.isnan(imag_part).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: NaN in imag part")
        if np.isnan(event_coords).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: NaN in event_coords")
        
        # Check for Inf
        if np.isinf(real_part).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Inf in real part")
        if np.isinf(imag_part).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Inf in imag part")
        if np.isinf(event_coords).any():
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Inf in event_coords")
        
        # Check for all zeros (embeddings should not be all zero)
        magnitude = np.sqrt(real_part**2 + imag_part**2)
        all_magnitudes.extend(magnitude.flatten())
        
        if np.all(magnitude < 1e-10):
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: All embeddings are zero")
        
        # Check coordinate ranges
        if num_vectors > 0:
            x_coords = event_coords[:, 0]
            y_coords = event_coords[:, 1]
            t_coords = event_coords[:, 2]
            p_coords = event_coords[:, 3]
            
            # Check x, y are in valid range [0, 127]
            if x_coords.min() < 0 or x_coords.max() > 128:
                issues.append(f"Sample {sample_idx}, interval {interval_idx}: X coords out of range [{x_coords.min():.1f}, {x_coords.max():.1f}]")
            if y_coords.min() < 0 or y_coords.max() > 128:
                issues.append(f"Sample {sample_idx}, interval {interval_idx}: Y coords out of range [{y_coords.min():.1f}, {y_coords.max():.1f}]")
            
            # Check polarity is 0 or 1
            unique_p = np.unique(p_coords)
            if not np.all(np.isin(unique_p, [0, 1])):
                issues.append(f"Sample {sample_idx}, interval {interval_idx}: Invalid polarity values {unique_p}")
        
        # Check shape consistency
        if real_part.shape != imag_part.shape:
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Shape mismatch real={real_part.shape}, imag={imag_part.shape}")
        
        if real_part.shape[0] != event_coords.shape[0]:
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Vectors and coords mismatch {real_part.shape[0]} != {event_coords.shape[0]}")
        
        if event_coords.shape[1] != 4:
            issues.append(f"Sample {sample_idx}, interval {interval_idx}: Event coords should be [N, 4], got {event_coords.shape}")
    
    # Check overall statistics
    if total_vectors > 0 and len(all_magnitudes) > 0:
        mean_mag = np.mean(all_magnitudes)
        std_mag = np.std(all_magnitudes)
        
        if verbose:
            print(f"  Sample {sample_idx}: {total_vectors} vectors, mag μ={mean_mag:.4f} σ={std_mag:.4f}")
        
        # Very low magnitude might indicate issues
        if mean_mag < 1e-6:
            issues.append(f"Sample {sample_idx}: Very low mean magnitude {mean_mag:.4e}")
    
    return issues, total_vectors, all_magnitudes


def validate_hdf5_file(h5_path, max_samples=None, verbose=False):
    """Validate an HDF5 file."""
    print(f"\nValidating: {h5_path}")
    print(f"File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
    
    with h5py.File(h5_path, 'r') as h5f:
        num_samples = h5f['labels'].shape[0]
        print(f"Total samples: {num_samples}")
        
        # Check metadata
        print("\nMetadata attributes:")
        for key, value in h5f.attrs.items():
            print(f"  {key}: {value}")
        
        # Validate samples
        samples_to_check = num_samples if max_samples is None else min(num_samples, max_samples)
        print(f"\nValidating {samples_to_check} samples...")
        
        all_issues = []
        total_vectors_all = 0
        all_magnitudes_global = []
        
        for idx in tqdm(range(samples_to_check)):
            issues, total_vectors, magnitudes = validate_sample(h5f, idx, verbose=verbose)
            all_issues.extend(issues)
            total_vectors_all += total_vectors
            all_magnitudes_global.extend(magnitudes)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Validation Summary")
        print(f"{'='*60}")
        
        print(f"Samples checked: {samples_to_check}")
        print(f"Total vectors: {total_vectors_all}")
        print(f"Average vectors per sample: {total_vectors_all / samples_to_check:.1f}")
        
        if len(all_magnitudes_global) > 0:
            mag_array = np.array(all_magnitudes_global)
            print(f"\nEmbedding Magnitude Statistics:")
            print(f"  Mean: {mag_array.mean():.4f}")
            print(f"  Std: {mag_array.std():.4f}")
            print(f"  Min: {mag_array.min():.4f}")
            print(f"  Max: {mag_array.max():.4f}")
            print(f"  Median: {np.median(mag_array):.4f}")
            
            # Check for zeros
            zero_count = np.sum(mag_array < 1e-10)
            zero_pct = 100 * zero_count / len(mag_array)
            print(f"  Near-zero (< 1e-10): {zero_count} ({zero_pct:.2f}%)")
            
            if zero_pct > 5:
                all_issues.append(f"WARNING: {zero_pct:.1f}% of embeddings are near-zero")
        
        # Print issues
        if all_issues:
            print(f"\n{'='*60}")
            print(f"ISSUES FOUND: {len(all_issues)}")
            print(f"{'='*60}")
            for issue in all_issues[:20]:  # Show first 20
                print(f"  ❌ {issue}")
            if len(all_issues) > 20:
                print(f"  ... and {len(all_issues) - 20} more issues")
            return False
        else:
            print(f"\n✅ No issues found! Data is valid.")
            return True


def main():
    parser = argparse.ArgumentParser(description='Validate precomputed HDF5 data')
    parser.add_argument(
        '--precomputed_dir',
        type=str,
        default='precomputed_data/dvsgesture',
        help='Directory containing precomputed HDF5 files'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to check (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-sample statistics'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("HDF5 Data Validation")
    print("="*60)
    
    # Check train file
    train_path = os.path.join(args.precomputed_dir, 'train.h5')
    if os.path.exists(train_path):
        train_valid = validate_hdf5_file(train_path, args.max_samples, args.verbose)
    else:
        print(f"❌ Train file not found: {train_path}")
        return 1
    
    # Check validation file
    val_path = os.path.join(args.precomputed_dir, 'validation.h5')
    if os.path.exists(val_path):
        val_valid = validate_hdf5_file(val_path, args.max_samples, args.verbose)
    else:
        print(f"❌ Validation file not found: {val_path}")
        return 1
    
    # Final result
    print(f"\n{'='*60}")
    if train_valid and val_valid:
        print("✅ ALL DATA VALID - Ready for training!")
        print("="*60)
        return 0
    else:
        print("❌ VALIDATION FAILED - Fix issues before training")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit(main())
