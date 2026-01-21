"""
Test script comparing sampling strategies - UPDATED FOR KERNEL-AWARE SAMPLING
"""

import sys
import os
import numpy as np
import torch
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dvsgesture.dataset import DVSGesture
from utils.density_adaptive_spatial_striding import adaptive_spatial_sampling


def load_intervals(dataset_dir, accumulation_interval_ms=200.0, num_samples=10):
    """Load intervals separately."""
    dataset = DVSGesture(
        dataset_dir=dataset_dir,
        purpose='train',
        height=128,
        width=128,
        use_flip_augmentation=False,
        accumulation_interval_ms=accumulation_interval_ms
    )
    
    intervals = []
    # Try to get enough non-empty intervals
    sample_idx = 0
    while len(intervals) < num_samples and sample_idx < len(dataset):
        sample = dataset[sample_idx]
        sample_idx += 1
        for j in range(len(sample['events_xy_sliced'])):
            interval = {
                'events_xy': sample['events_xy_sliced'][j],
                'events_t': sample['events_t_sliced'][j],
                'events_p': sample['events_p_sliced'][j],
            }
            if len(interval['events_t']) > 0:
                intervals.append(interval)
                if len(intervals) >= num_samples:
                    break
    
    return intervals


def test_adaptive_striding(intervals, kernel_size=17, overlap_factor=0.2):
    """Adaptive striding: Returns queries based on kernel coverage."""
    print("\n" + "="*60)
    print(f"Adaptive Kernel-Aware Sampling (K={kernel_size}, Overlap={overlap_factor})")
    print("="*60)
    
    total_events, total_queries, total_unique = 0, 0, 0
    times = []
    query_counts = []
    
    # Calculate stride for reference
    stride = int(kernel_size * (1 - overlap_factor))
    print(f"  Calculated Stride: {stride} pixels")
    print(f"  Grid Cell Area: {stride}x{stride} = {stride**2} pixels")
    
    for interval in intervals:
        events_xy, events_t, events_p = interval['events_xy'], interval['events_t'], interval['events_p']
        num_events = len(events_t)
        
        start = time.time()
        
        # New Utility Call
        indices = adaptive_spatial_sampling(
            events_t, events_xy[:, 1], events_xy[:, 0], events_p,
            height=128, width=128,
            kernel_size=kernel_size,
            overlap_factor=overlap_factor,
            sort_by_time=True
        )
        
        times.append(time.time() - start)
        
        num_queries = len(indices)
        unique = len(np.unique(events_xy[indices], axis=0)) if num_queries > 0 else 0
        
        total_events += num_events
        total_queries += num_queries
        total_unique += unique
        query_counts.append(num_queries)
    
    print(f"  Intervals: {len(intervals)}")
    print(f"  Total events: {total_events}")
    print(f"  Total queries: {total_queries}")
    print(f"  Avg queries/interval: {total_queries/len(intervals):.1f}")
    if query_counts:
        print(f"  Query count range: [{min(query_counts)}, {max(query_counts)}]")
    print(f"  Avg unique pixels/interval: {total_unique/len(intervals):.1f}")
    print(f"  Avg time/interval: {np.mean(times)*1000:.2f}ms")
    compression = total_events/total_queries if total_queries > 0 else 0
    print(f"  Compression: {compression:.2f}x")
    return total_queries


def main():
    print("="*60)
    print("Sampling Verification")
    print("="*60)
    
    with open('configs/config_dvsgesture.yaml', 'r') as f:
        # Simple loader to avoid custom tag errors if present
        config = yaml.safe_load(f)
    
    precompute_cfg = config['PRECOMPUTING']
    dataset_dir = precompute_cfg['dataset_dir']
    
    print(f"\nDataset: {dataset_dir}")
    print(f"Loading intervals...")
    
    intervals = load_intervals(dataset_dir, precompute_cfg['accumulation_interval_ms'], num_samples=50)
    
    print(f"Loaded {len(intervals)} intervals")
    if not intervals:
        print("No intervals loaded! Check dataset path.")
        return

    print(f"Avg events/interval: {np.mean([len(iv['events_t']) for iv in intervals]):.0f}")
    
    # Test with different overlaps to verify density increase
    q_low = test_adaptive_striding(intervals, kernel_size=17, overlap_factor=0.0)
    q_mid = test_adaptive_striding(intervals, kernel_size=17, overlap_factor=0.5)
    q_high = test_adaptive_striding(intervals, kernel_size=17, overlap_factor=0.8)
    
    print("\n" + "="*60)
    print("OVERLAP FACTOR SCALING CHECK")
    print("="*60)
    print(f"Overlap 0.0 (Stride 17): {q_low} queries")
    print(f"Overlap 0.5 (Stride 8):  {q_mid} queries ({q_mid/q_low:.1f}x)")
    print(f"Overlap 0.8 (Stride 3):  {q_high} queries ({q_high/q_low:.1f}x)")
    
    if q_high > q_mid > q_low:
         print("\nSUCCESS: Increasing overlap factor consistently increases query count (spatial density).")
    else:
         print("\nWARNING: Query counts did not monotonic increase. Check logic.")

if __name__ == '__main__':
    main()
