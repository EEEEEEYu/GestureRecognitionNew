"""
Test script comparing sampling strategies - CORRECTED VERSION
Each method uses its natural query selection, not forced to same count.
"""

import sys
import os
import numpy as np
import torch
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.DVSGesture import DVSGesture
from data.SparseVKMEncoder import VecKMSparse
from utils.simple_density_sampling import simple_density_sample
from utils.density_adaptive_spatial_striding import sample_veckm_queries


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
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        for j in range(len(sample['events_xy_sliced'])):
            interval = {
                'events_xy': sample['events_xy_sliced'][j],
                'events_t': sample['events_t_sliced'][j],
                'events_p': sample['events_p_sliced'][j],
            }
            if len(interval['events_t']) > 0:
                intervals.append(interval)
    
    return intervals


def test_random(intervals, ratio):
    """Random sampling: select ratio × num_events queries."""
    print("\n" + "="*60)
    print("Random Sampling")
    print("="*60)
    
    total_events, total_queries, total_unique = 0, 0, 0
    times = []
    
    for interval in intervals:
        events_xy, events_t = interval['events_xy'], interval['events_t']
        num_events = len(events_t)
        num_queries = max(1, int(num_events * ratio))
        
        start = time.time()
        indices = np.random.choice(num_events, num_queries, replace=False)
        times.append(time.time() - start)
        
        unique = len(np.unique(events_xy[indices], axis=0))
        total_events += num_events
        total_queries += num_queries
        total_unique += unique
    
    print(f"  Intervals: {len(intervals)}")
    print(f"  Total events: {total_events}")
    print(f"  Total queries: {total_queries}")
    print(f"  Avg queries/interval: {total_queries/len(intervals):.1f}")
    print(f"  Avg unique pixels/interval: {total_unique/len(intervals):.1f}")
    print(f"  Avg time/interval: {np.mean(times)*1000:.2f}ms")
    print(f"  Compression: {total_events/total_queries:.2f}x")
    return total_queries


def test_simple_density(intervals, ratio, grid_size=16):
    """Simple density: select ratio × num_events queries, weighted by density."""
    print("\n" + "="*60)
    print("Simple Density Sampling")
    print("="*60)
    
    total_events, total_queries, total_unique = 0, 0, 0
    times = []
    
    for interval in intervals:
        events_xy, events_t, events_p = interval['events_xy'], interval['events_t'], interval['events_p']
        num_events = len(events_t)
        num_queries = max(1, int(num_events * ratio))
        
        start = time.time()
        indices = simple_density_sample(
            events_xy, events_t, events_p,
            num_samples=num_queries,
            grid_size=grid_size,
            height=128, width=128
        )
        times.append(time.time() - start)
        
        unique = len(np.unique(events_xy[indices], axis=0))
        total_events += num_events
        total_queries += len(indices)
        total_unique += unique
    
    print(f"  Intervals: {len(intervals)}")
    print(f"  Total events: {total_events}")
    print(f"  Total queries: {total_queries}")
    print(f"  Avg queries/interval: {total_queries/len(intervals):.1f}")
    print(f"  Avg unique pixels/interval: {total_unique/len(intervals):.1f}")
    print(f"  Avg time/interval: {np.mean(times)*1000:.2f}ms")
    print(f"  Compression: {total_events/total_queries:.2f}x")
    return total_queries


def test_adaptive_striding(intervals, kernel_size=17, stride_factor=0.5):
    """Adaptive striding: returns VARIABLE number of queries based on spatial structure."""
    print("\n" + "="*60)
    print("Adaptive Striding Sampling (Natural Query Count)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_events, total_queries, total_unique = 0, 0, 0
    times = []
    query_counts = []
    
    for interval in intervals:
        events_xy, events_t = interval['events_xy'], interval['events_t']
        num_events = len(events_t)
        
        start = time.time()
        t_tensor = torch.from_numpy(events_t).float().to(device)
        y_tensor = torch.from_numpy(events_xy[:, 1]).float().to(device)
        x_tensor = torch.from_numpy(events_xy[:, 0]).float().to(device)
        
        # Let it return natural number of queries (don't force top_k)
        qy, qx, qt = sample_veckm_queries(
            t_tensor, y_tensor, x_tensor,
            H=128, W=128,
            encoding_kernel_size=kernel_size,
            stride_factor=stride_factor,
            top_k=99999  # Let it return all spatial peaks
        )
        
        # Map to events
        indices = []
        qy_np, qx_np, qt_np = qy.cpu().numpy(), qx.cpu().numpy(), qt.cpu().numpy()
        
        for i in range(len(qy_np)):
            mask = (events_xy[:, 0] == qx_np[i]) & (events_xy[:, 1] == qy_np[i])
            pix_indices = np.where(mask)[0]
            if len(pix_indices) > 0:
                closest = np.argmin(np.abs(events_t[pix_indices] - qt_np[i]))
                indices.append(pix_indices[closest])
        
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
    print(f"  Query count range: [{min(query_counts)}, {max(query_counts)}]")
    print(f"  Avg unique pixels/interval: {total_unique/len(intervals):.1f}")
    print(f"  Avg time/interval: {np.mean(times)*1000:.2f}ms")
    print(f"  Compression: {total_events/total_queries:.2f}x" if total_queries > 0 else "  N/A")
    return total_queries


def main():
    print("="*60)
    print("Sampling Comparison: Natural Query Counts")
    print("="*60)
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    precompute_cfg = config['PRECOMPUTING']
    dataset_dir = '/fs/nexus-projects/DVS_Actions/DVSGestureData'
    
    print(f"\nDataset: {dataset_dir}")
    print(f"Loading intervals...")
    
    intervals = load_intervals(dataset_dir, precompute_cfg['accumulation_interval_ms'], num_samples=10)
    ratio = precompute_cfg['ratio_of_vectors']
    
    print(f"Loaded {len(intervals)} intervals")
    print(f"Avg events/interval: {np.mean([len(iv['events_t']) for iv in intervals]):.0f}")
    print(f"Target ratio for random/density: {ratio*100}%")
    
    # Test all methods
    q_random = test_random(intervals, ratio)
    q_density = test_simple_density(intervals, ratio, grid_size=16)
    q_adaptive = test_adaptive_striding(intervals, kernel_size=17, stride_factor=0.1)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Random:          {q_random:7d} queries total")
    print(f"Simple Density:  {q_density:7d} queries total")
    print(f"Adaptive:        {q_adaptive:7d} queries total (natural count)")
    print(f"\nAdaptive uses {q_random/q_adaptive:.1f}x FEWER queries than random!")


if __name__ == '__main__':
    main()
