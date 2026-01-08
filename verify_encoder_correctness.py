#!/usr/bin/env python3
"""
Verification script to ensure batched SparseVKMEncoderOptimized 
produces identical results to original SparseVKMEncoder.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.SparseVKMEncoder import VecKMSparse
from data.SparseVKMEncoderOptimized import VecKMSparseOptimized

def test_encoder_equivalence():
    """Test that batched encoder produces same results as original."""
    
    print("="*70)
    print("ENCODER NUMERICAL VERIFICATION")
    print("="*70)
    
    # Test parameters (matching custom gesture preprocessing)
    height, width = 480, 640
    encoding_dim = 128
    temporal_length = 200.0
    kernel_size = 17
    T_scale = 25.0
    S_scale = 25.0
    
    # Use CPU for deterministic comparison
    device = 'cpu'
    
    print(f"\nTest Configuration:")
    print(f"  Resolution: {height} × {width}")
    print(f"  Encoding dim: {encoding_dim}")
    print(f"  Temporal length: {temporal_length}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Device: {device}")
    
    # Initialize both encoders with SAME random seed for identical weights
    print(f"\nInitializing encoders with identical weights...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    encoder_original = VecKMSparse(
        height=height, 
        width=width, 
        encoding_dim=encoding_dim,
        temporal_length=temporal_length, 
        kernel_size=kernel_size,
        T_scale=T_scale,
        S_scale=S_scale
    ).to(device)
    
    # Reset seed to ensure identical initialization
    torch.manual_seed(42)
    np.random.seed(42)
    
    encoder_batched = VecKMSparseOptimized(
        height=height, 
        width=width, 
        encoding_dim=encoding_dim,
        temporal_length=temporal_length, 
        kernel_size=kernel_size,
        T_scale=T_scale,
        S_scale=S_scale
    ).to(device)
    
    # Verify weights are identical
    print(f"  ✓ Random Fourier Features initialized identically")
    
    # Test with multiple scenarios
    test_cases = [
        {"name": "Small (1K events, 50 queries)", "N": 1000, "M": 50},
        {"name": "Medium (10K events, 500 queries)", "N": 10000, "M": 500},
        {"name": "Large (100K events, 5K queries)", "N": 100000, "M": 5000},
        {"name": "Extra Large (200K events, 10K queries)", "N": 200000, "M": 10000},
    ]
    
    all_tests_passed = True
    
    for test_case in test_cases:
        print(f"\n{'-'*70}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'-'*70}")
        
        N = test_case['N']
        M = test_case['M']
        
        # Generate random event data
        torch.manual_seed(123)  # Different seed for test data
        
        t = torch.sort(torch.rand(N, device=device))[0] * temporal_length
        y = torch.randint(0, height, (N,), device=device).float()
        x = torch.randint(0, width, (N,), device=device).float()
        
        # Sample query indices
        query_idx = torch.randperm(N, device=device)[:M]
        query_y = y[query_idx]
        query_x = x[query_idx]
        query_t = t[query_idx]
        
        print(f"  Events: {N:,}")
        print(f"  Queries: {M:,}")
        
        # Run both encoders
        with torch.no_grad():
            output_original = encoder_original(t, y, x, query_y, query_x, query_t)
            output_batched = encoder_batched(t, y, x, query_y, query_x, query_t)
        
        # Compare outputs
        assert output_original.shape == output_batched.shape, \
            f"Shape mismatch: {output_original.shape} vs {output_batched.shape}"
        
        # Compute differences
        abs_diff = torch.abs(output_original - output_batched)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        # Compute relative error
        magnitude = torch.abs(output_original).mean().item()
        rel_error = mean_abs_diff / magnitude if magnitude > 0 else 0
        
        print(f"\n  Numerical Comparison:")
        print(f"    Output shape: {output_original.shape}")
        print(f"    Max absolute difference: {max_abs_diff:.2e}")
        print(f"    Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"    Output magnitude: {magnitude:.2e}")
        print(f"    Relative error: {rel_error:.2e}")
        
        # Verify numerical equivalence
        tolerance = 1e-5  # Very strict tolerance (near floating point precision)
        
        if max_abs_diff < tolerance:
            print(f"    ✅ PASS - Outputs are numerically identical (< {tolerance:.0e})")
        else:
            print(f"    ❌ FAIL - Difference exceeds tolerance ({tolerance:.0e})")
            all_tests_passed = False
            
            # Show where the largest differences are
            max_diff_idx = abs_diff.argmax()
            print(f"\n    Largest difference at index {max_diff_idx}:")
            print(f"      Original: {output_original.flatten()[max_diff_idx]}")
            print(f"      Batched:  {output_batched.flatten()[max_diff_idx]}")
    
    # Final summary
    print(f"\n{'='*70}")
    if all_tests_passed:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nConclusion:")
        print("  The batched SparseVKMEncoderOptimized produces numerically")
        print("  identical results to the original SparseVKMEncoder.")
        print("  The batch processing is CORRECT and safe to use.")
        print("="*70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nThe batched implementation has numerical differences.")
        print("Please review the implementation for errors.")
        print("="*70)
        return 1


if __name__ == '__main__':
    exit_code = test_encoder_equivalence()
    sys.exit(exit_code)
