#!/usr/bin/env python3
"""
Verify integrity of precomputed HDF5 files.
Checks for: empty data, NaN values, all-zero tensors, and data corruption.
"""

import h5py
import numpy as np
import os

def verify_hdf5_integrity(h5_path):
    """Verify integrity of a precomputed HDF5 file."""
    
    print("="*80)
    print(f"Verifying: {h5_path}")
    print("="*80)
    
    if not os.path.exists(h5_path):
        print(f"❌ File does not exist: {h5_path}")
        return False
    
    issues = []
    warnings = []
    
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Check metadata
            num_samples = h5f['labels'].shape[0]
            print(f"\n📊 Dataset Overview:")
            print(f"  Total samples: {num_samples}")
            print(f"  File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
            
            if num_samples == 0:
                issues.append("No samples in dataset!")
                return False
            
            # Check metadata arrays
            print(f"\n🔍 Checking metadata arrays...")
            labels = h5f['labels'][:]
            num_intervals = h5f['num_intervals'][:]
            
            if np.any(np.isnan(labels)):
                issues.append("NaN found in labels")
            if np.any(np.isnan(num_intervals)):
                issues.append("NaN found in num_intervals")
            
            print(f"  ✅ Labels: {len(labels)} entries, range [{labels.min()}, {labels.max()}]")
            print(f"  ✅ Num intervals: mean={num_intervals.mean():.1f}, range [{num_intervals.min()}, {num_intervals.max()}]")
            
            # Check each sample
            print(f"\n🔍 Checking sample data...")
            empty_samples = []
            nan_samples = []
            zero_samples = []
            valid_samples = 0
            
            total_vectors = 0
            total_intervals = 0
            
            for sample_idx in range(num_samples):
                sample_group = h5f[f'sample_{sample_idx:06d}']
                num_vectors_per_interval = sample_group['num_vectors_per_interval'][:]
                
                num_intervals_for_sample = len(num_vectors_per_interval)
                total_intervals += num_intervals_for_sample
                
                sample_has_issue = False
                sample_vectors = 0
                
                for interval_idx in range(num_intervals_for_sample):
                    interval_group = sample_group[f'interval_{interval_idx:03d}']
                    
                    # Load data
                    real = interval_group['real'][:]
                    imag = interval_group['imag'][:]
                    event_coords = interval_group['event_coords'][:]
                    
                    # Check for NaN
                    if np.any(np.isnan(real)) or np.any(np.isnan(imag)) or np.any(np.isnan(event_coords)):
                        if sample_idx not in nan_samples:
                            nan_samples.append(sample_idx)
                        sample_has_issue = True
                    
                    # Check for all-zero (only problematic if interval should have data)
                    num_vectors_in_interval = real.shape[0]
                    sample_vectors += num_vectors_in_interval
                    
                    if num_vectors_in_interval > 0:
                        if np.all(real == 0) and np.all(imag == 0):
                            if sample_idx not in zero_samples:
                                zero_samples.append(sample_idx)
                    else:
                        if sample_idx not in empty_samples:
                            empty_samples.append(sample_idx)
                
                total_vectors += sample_vectors
                
                if not sample_has_issue and sample_vectors > 0:
                    valid_samples += 1
            
            # Report findings
            print(f"\n📈 Statistics:")
            print(f"  Total intervals: {total_intervals}")
            print(f"  Total vectors: {total_vectors:,}")
            print(f"  Average vectors/sample: {total_vectors/num_samples:.1f}")
            print(f"  Average vectors/interval: {total_vectors/total_intervals:.1f}")
            
            print(f"\n✅ Valid samples: {valid_samples}/{num_samples} ({valid_samples/num_samples*100:.1f}%)")
            
            if empty_samples:
                warnings.append(f"{len(empty_samples)} samples have empty intervals (normal for background-heavy sequences)")
                print(f"  ⚠️  {len(empty_samples)} samples with empty intervals: {empty_samples[:5]}{'...' if len(empty_samples) > 5 else ''}")
            
            if zero_samples:
                issues.append(f"{len(zero_samples)} samples have all-zero data")
                print(f"  ❌ {len(zero_samples)} samples with all-zero data: {zero_samples[:5]}{'...' if len(zero_samples) > 5 else ''}")
            
            if nan_samples:
                issues.append(f"{len(nan_samples)} samples contain NaN values")
                print(f"  ❌ {len(nan_samples)} samples with NaN: {nan_samples[:5]}{'...' if len(nan_samples) > 5 else ''}")
            
            # Sample a few random samples for detailed check
            print(f"\n🔬 Detailed check of random samples...")
            sample_indices = np.random.choice(num_samples, min(5, num_samples), replace=False)
            
            for sample_idx in sample_indices:
                sample_group = h5f[f'sample_{sample_idx:06d}']
                num_vectors_per_interval = sample_group['num_vectors_per_interval'][:]
                
                # Check first interval
                if len(num_vectors_per_interval) > 0 and num_vectors_per_interval[0] > 0:
                    interval_group = sample_group['interval_000']
                    real = interval_group['real'][:]
                    imag = interval_group['imag'][:]
                    
                    magnitude = np.abs(real + 1j * imag)
                    mag_mean = magnitude.mean()
                    mag_std = magnitude.std()
                    mag_max = magnitude.max()
                    
                    print(f"  Sample {sample_idx:03d}, Interval 0:")
                    print(f"    Vectors: {real.shape[0]}, Encoding dim: {real.shape[1]}")
                    print(f"    Magnitude: mean={mag_mean:.3f}, std={mag_std:.3f}, max={mag_max:.3f}")
                    
                    if mag_mean < 0.001:
                        warnings.append(f"Sample {sample_idx} has very low magnitude (might be issue)")
                    elif mag_mean > 100:
                        warnings.append(f"Sample {sample_idx} has very high magnitude (might be issue)")
    
    except Exception as e:
        issues.append(f"Error reading file: {e}")
        print(f"\n❌ Error: {e}")
        return False
    
    # Final verdict
    print(f"\n" + "="*80)
    if issues:
        print(f"❌ VERIFICATION FAILED - {len(issues)} issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    elif warnings:
        print(f"⚠️  VERIFICATION PASSED WITH WARNINGS - {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print(f"\n✅ Data appears valid but review warnings above")
        return True
    else:
        print(f"✅ VERIFICATION PASSED - No issues found!")
        print(f"   Data is valid and ready for training.")
        return True


def main():
    base_dir = "/fs/nexus-scratch/haowenyu/GestureRecognitionNew/precomputed_data/custom_gesture"
    
    print("\n" + "="*80)
    print("PRECOMPUTED DATA INTEGRITY VERIFICATION")
    print("="*80)
    
    results = {}
    
    for split in ['train', 'validation']:
        h5_path = os.path.join(base_dir, f'{split}.h5')
        print(f"\n")
        results[split] = verify_hdf5_integrity(h5_path)
        print()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for split, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {split:12s}: {status}")
    
    if all_passed:
        print(f"\n✅ All precomputed files are valid and ready for training!")
    else:
        print(f"\n❌ Some files have issues - review details above")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
