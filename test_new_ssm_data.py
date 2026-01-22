"""
Standalone test script to verify the data generation format for new_ssm.py
This mimics the data format from SparseVKMEncoder.py without requiring model dependencies.
"""
import torch
import torch.nn.functional as F

def generate_batch_data(batch_size=4, num_segments=8, encoding_dim=64, device='cpu'):
    """
    Generate synthetic batch data that mimics SparseVKMEncoder output.
    
    Args:
        batch_size: Number of samples in the batch
        num_segments: Number of temporal segments per sample (T)
        encoding_dim: Dimension of complex embeddings (D)
        device: torch device
        
    Returns:
        Dictionary with keys:
        - 'segments_complex': List[List[Tensor]] - Complex embeddings per segment
        - 'segments_coords': List[List[Tensor]] - Spatial coordinates per segment
    """
    batch = {
        'segments_complex': [],
        'segments_coords': []
    }
    
    for b in range(batch_size):
        sample_segments_complex = []
        sample_segments_coords = []
        
        for seg in range(num_segments):
            # Varying number of events per segment (between 50-200)
            num_events = torch.randint(50, 200, (1,)).item()
            
            # Generate complex embeddings (mimicking VecKMSparse output)
            # VecKMSparse returns complex numbers from exp(i * phase) operations
            real_part = torch.randn(num_events, encoding_dim, device=device)
            imag_part = torch.randn(num_events, encoding_dim, device=device)
            complex_emb = torch.complex(real_part, imag_part)
            
            # Generate coordinates (y, x) in pixel space
            # DVS346 resolution: 346x260 (width x height)
            coords = torch.stack([
                torch.randint(0, 260, (num_events,), device=device, dtype=torch.float32),  # y
                torch.randint(0, 346, (num_events,), device=device, dtype=torch.float32)   # x
            ], dim=1)
            
            sample_segments_complex.append(complex_emb)
            sample_segments_coords.append(coords)
        
        batch['segments_complex'].append(sample_segments_complex)
        batch['segments_coords'].append(sample_segments_coords)
    
    return batch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- Generate Synthetic Batch Data ---
    batch_size = 4
    num_segments_per_sample = 8
    encoding_dim = 64
    
    batch = generate_batch_data(
        batch_size=batch_size,
        num_segments=num_segments_per_sample,
        encoding_dim=encoding_dim,
        device=device
    )
    
    print(f"\n{'='*60}")
    print(f"Batch Data Structure (mimicking SparseVKMEncoder output)")
    print(f"{'='*60}")
    
    print(f"\nBatch configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Segments per sample: {num_segments_per_sample}")
    print(f"  Encoding dimension: {encoding_dim}")
    
    print(f"\nData structure:")
    print(f"  batch['segments_complex']: List[List[Tensor]]")
    print(f"    Outer list length: {len(batch['segments_complex'])} (batch size)")
    print(f"    Inner list length: {len(batch['segments_complex'][0])} (num segments)")
    
    print(f"\n  batch['segments_coords']: List[List[Tensor]]")
    print(f"    Outer list length: {len(batch['segments_coords'])} (batch size)")
    print(f"    Inner list length: {len(batch['segments_coords'][0])} (num segments)")
    
    print(f"\nSample segment details (batch=0, segment=0):")
    sample_complex = batch['segments_complex'][0][0]
    sample_coords = batch['segments_coords'][0][0]
    
    print(f"  Complex embeddings:")
    print(f"    Shape: {sample_complex.shape}")
    print(f"    Dtype: {sample_complex.dtype}")
    print(f"    Real part range: [{sample_complex.real.min():.3f}, {sample_complex.real.max():.3f}]")
    print(f"    Imag part range: [{sample_complex.imag.min():.3f}, {sample_complex.imag.max():.3f}]")
    
    print(f"\n  Coordinates:")
    print(f"    Shape: {sample_coords.shape}")
    print(f"    Dtype: {sample_coords.dtype}")
    print(f"    Y range: [{sample_coords[:, 0].min():.1f}, {sample_coords[:, 0].max():.1f}]")
    print(f"    X range: [{sample_coords[:, 1].min():.1f}, {sample_coords[:, 1].max():.1f}]")
    
    print(f"\nAll segments (events per segment):")
    for b_idx in range(batch_size):
        event_counts = [batch['segments_complex'][b_idx][s].shape[0] 
                       for s in range(num_segments_per_sample)]
        print(f"  Sample {b_idx}: {event_counts}")
    
    # Verify data format matches model expectations
    print(f"\n{'='*60}")
    print(f"Verification: Data Format Matches Model Requirements")
    print(f"{'='*60}")
    
    print("\nExpected by NestedEventMamba.forward():")
    print("  batch['segments_complex'][b][s]: Tensor[N_events, encoding_dim] (cfloat)")
    print("  batch['segments_coords'][b][s]: Tensor[N_events, 2] (float)")
    
    print("\nGenerated:")
    all_valid = True
    for b in range(batch_size):
        for s in range(num_segments_per_sample):
            z = batch['segments_complex'][b][s]
            c = batch['segments_coords'][b][s]
            
            # Check shapes
            if z.dtype != torch.cfloat:
                print(f"  ✗ Sample {b}, Segment {s}: Complex dtype mismatch")
                all_valid = False
            if z.shape[1] != encoding_dim:
                print(f"  ✗ Sample {b}, Segment {s}: Complex dim mismatch")
                all_valid = False
            if c.shape[1] != 2:
                print(f"  ✗ Sample {b}, Segment {s}: Coords dim mismatch")
                all_valid = False
            if z.shape[0] != c.shape[0]:
                print(f"  ✗ Sample {b}, Segment {s}: Shape mismatch")
                all_valid = False
    
    if all_valid:
        print("  ✓ All segments have correct format!")
        
        # Test the concatenation that happens in the model
        print("\nTest model operations:")
        test_z = batch['segments_complex'][0][0]
        test_c = batch['segments_coords'][0][0]
        
        # Normalize coords (as done in model)
        c_norm = test_c / torch.tensor([346, 260], device=test_c.device)
        print(f"  ✓ Coordinate normalization: {c_norm.shape}")
        
        # Concatenate real, imag, coords (as done in model)
        feat = torch.cat([test_z.real, test_z.imag, c_norm], dim=-1)
        expected_feat_dim = encoding_dim * 2 + 2
        print(f"  ✓ Feature concatenation: {feat.shape} (expected: [N, {expected_feat_dim}])")
        
        if feat.shape[1] == expected_feat_dim:
            print(f"\n✓ Data format is CORRECT and ready for NestedEventMamba!")
        else:
            print(f"\n✗ Feature dimension mismatch!")
    else:
        print("\n✗ Data format has issues!")
    
    print(f"\n{'='*60}")
