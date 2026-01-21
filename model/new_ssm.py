import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2

class ComplexPolarInput(nn.Module):
    """
    Splits Complex inputs into Magnitude and Phase streams, 
    concatenating them with (x,y) spatial context.
    """
    def __init__(self, encoding_dim, hidden_dim):
        super().__init__()
        # Mag Stream: |z| (dim) + x (1) + y (1)
        self.mag_proj = nn.Linear(encoding_dim + 2, hidden_dim)
        
        # Phase Stream: cos(angle) (dim) + sin(angle) (dim) + x (1) + y (1)
        self.phase_proj = nn.Linear(encoding_dim * 2 + 2, hidden_dim)
        
        self.norm_m = nn.LayerNorm(hidden_dim)
        self.norm_p = nn.LayerNorm(hidden_dim)

    def forward(self, z, x_norm, y_norm):
        # z: [Batch, N, D] (Complex)
        # x_norm, y_norm: [Batch, N, 1] (already normalized to [0, 1])
        
        # 1. Magnitude Path
        mag = z.abs()
        # Normalize magnitude vector to unit length for stability
        mag = F.normalize(mag, p=2, dim=-1)
        mag_in = torch.cat([mag, x_norm, y_norm], dim=-1)
        feat_mag = self.norm_m(F.gelu(self.mag_proj(mag_in)))
        
        # 2. Phase Path
        phase = z.angle()
        # Use cos/sin to avoid discontinuity at -pi/pi
        phase_in = torch.cat([torch.cos(phase), torch.sin(phase), x_norm, y_norm], dim=-1)
        feat_phase = self.norm_p(F.gelu(self.phase_proj(phase_in)))
        
        return feat_mag, feat_phase

class LocalHilbertAggregator(nn.Module):
    """
    The 'Space' Block.
    Processes scattered events within a segment using a shared Mamba 
    scanning in spatial orders (XY, YX).
    """
    def __init__(self, dim, resolution):
        super().__init__()
        self.width, self.height = resolution
        # Shared SSM for spatial scanning
        self.spatial_ssm = Mamba2(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, coords):
        # x: [Batch, N_events, Dim]
        # coords: [Batch, N_events, 2] (x, y)
        
        # 1. Generate Sort Indices for XY and YX orders
        # (Simplified: In practice, use your lexsort logic here)
        # We process the sequence twice (forward XY, and maybe sort by Y)
        # For efficiency in this snippet, we just assume input is sorted by time
        # and we let Mamba handle it, OR we explicitly sort indices here.
        
        # First scan: Y-major order (y * W + x)
        sort_idx_yx = torch.argsort(coords[..., 1] * self.width + coords[..., 0], dim=1)
        x_sorted_yx = torch.gather(x, 1, sort_idx_yx.unsqueeze(-1).expand_as(x))
        
        # 2. Spatial Scan in YX order
        out_yx = self.spatial_ssm(x_sorted_yx)
        
        # Second scan: X-major order (x * W + y) for bidirectional processing
        sort_idx_xy = torch.argsort(coords[..., 0] * self.height + coords[..., 1], dim=1)
        x_sorted_xy = torch.gather(x, 1, sort_idx_xy.unsqueeze(-1).expand_as(x))
        out_xy = self.spatial_ssm(x_sorted_xy)
        
        # Unsort both outputs back to original order
        inverse_idx_yx = torch.argsort(sort_idx_yx, dim=1)
        out_yx_unsorted = torch.gather(out_yx, 1, inverse_idx_yx.unsqueeze(-1).expand_as(out_yx))
        
        inverse_idx_xy = torch.argsort(sort_idx_xy, dim=1)
        out_xy_unsorted = torch.gather(out_xy, 1, inverse_idx_xy.unsqueeze(-1).expand_as(out_xy))
        
        # Combine both directional scans
        out = (out_yx_unsorted + out_xy_unsorted) / 2
        
        # 3. Multi-scale Pooling (Condense N events -> 1 Segment Vector)
        # Combine mean and max pooling for richer feature representation
        mean_pool = torch.mean(out, dim=1)  # [B*T, Dim]
        max_pool = torch.max(out, dim=1)[0]  # [B*T, Dim]
        segment_feat = (mean_pool + max_pool) / 2 
        
        return self.norm(segment_feat)

class SymmetricPolarGating(nn.Module):
    """
    The Fusion Block.
    Uses Phase to gate Magnitude and Magnitude to gate Phase.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_p2m = nn.Linear(dim, dim) # Phase -> Mag
        self.gate_m2p = nn.Linear(dim, dim) # Mag -> Phase
        self.out_proj = nn.Linear(2 * dim, dim)

    def forward(self, feat_m, feat_p):
        # Compute Gates
        g_m = torch.sigmoid(self.gate_p2m(feat_p))
        g_p = torch.sigmoid(self.gate_m2p(feat_m))
        
        # Apply Gates (Symmetric)
        m_gated = feat_m * g_m
        p_gated = feat_p * g_p
        
        # Concatenate
        combined = torch.cat([m_gated, p_gated], dim=-1)
        return self.out_proj(combined)

class SpaceTimePolarSSM(nn.Module):
    """
    The Main Model.
    """
    def __init__(self, encoding_dim=64, hidden_dim=128, resolution=(346, 260), num_classes=11):
        super().__init__()
        self.width, self.height = resolution
        self.input_adapter = ComplexPolarInput(encoding_dim, hidden_dim)
        
        # Two independent Local Aggregators (One for Mag, One for Phase)
        self.local_ssm_mag = LocalHilbertAggregator(hidden_dim, resolution)
        self.local_ssm_phase = LocalHilbertAggregator(hidden_dim, resolution)
        
        # Global Time Mamba (One for each stream)
        self.global_ssm_mag = Mamba2(d_model=hidden_dim, d_state=64, expand=2)
        self.global_ssm_phase = Mamba2(d_model=hidden_dim, d_state=64, expand=2)
        
        # Fusion
        self.fusion = SymmetricPolarGating(hidden_dim)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, segments_complex, segments_coords):
        # Input is a list/tensor of segments: [Batch, T_segs, N_events, D]
        # segments_coords: [Batch, T_segs, N_events, 2] with unnormalized (x, y) in pixel coordinates
        B, T, N, D = segments_complex.shape
        
        # Flatten Batch and Time to process all segments in parallel
        z_flat = segments_complex.view(B*T, N, D)
        coords_flat = segments_coords.view(B*T, N, 2)
        
        # Normalize coordinates to [0, 1] range
        x_norm = coords_flat[..., 0:1] / self.width
        y_norm = coords_flat[..., 1:2] / self.height
        
        # 1. Project to Polar Streams
        f_mag, f_phase = self.input_adapter(z_flat, x_norm, y_norm)
        
        # 2. Local Aggregation (Spatial)
        # Returns [B*T, Hidden]
        seg_mag = self.local_ssm_mag(f_mag, coords_flat)
        seg_phase = self.local_ssm_phase(f_phase, coords_flat)
        
        # Reshape back to Sequence: [B, T, Hidden]
        seq_mag = seg_mag.view(B, T, -1)
        seq_phase = seg_phase.view(B, T, -1)
        
        # 3. Global Aggregation (Temporal)
        # Mamba scans across time T
        global_mag = self.global_ssm_mag(seq_mag)
        global_phase = self.global_ssm_phase(seq_phase)
        
        # 4. Symmetric Gating
        fused_features = self.fusion(global_mag, global_phase)
        
        # 5. Classify (Pool over time or take last state)
        final_out = fused_features.mean(dim=1) 
        return self.classifier(final_out)


if __name__ == "__main__":
    """
    Test the SpaceTimePolarSSM model with random complex encodings in XYT space.
    """
    print("=" * 60)
    print("Testing SpaceTimePolarSSM Model")
    print("=" * 60)
    
    # Model configuration
    encoding_dim = 64
    hidden_dim = 128
    resolution = (346, 260)  # DVS camera resolution (width, height)
    num_classes = 11
    batch_size = 2
    num_segments = 10
    events_per_segment = 50
    
    # Create model
    model = SpaceTimePolarSSM(
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim,
        resolution=resolution,
        num_classes=num_classes
    )
    model.eval()
    
    print(f"\nModel Configuration:")
    print(f"  Encoding Dim: {encoding_dim}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Num Classes: {num_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Generate random complex encodings in XYT space
    print(f"\nGenerating random test data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Segments per sample: {num_segments}")
    print(f"  Events per segment: {events_per_segment}")
    
    # Complex features: [Batch, T_segments, N_events, encoding_dim]
    # Simulate complex embeddings with magnitude and phase
    magnitude = torch.rand(batch_size, num_segments, events_per_segment, encoding_dim) * 2
    phase = torch.rand(batch_size, num_segments, events_per_segment, encoding_dim) * 2 * 3.14159
    segments_complex = magnitude * torch.exp(1j * phase)
    segments_complex = segments_complex.to(torch.complex64)
    
    # Coordinates: [Batch, T_segments, N_events, 2] - (x, y) in pixel space
    # Generate random coordinates within resolution
    x_coords = torch.randint(0, resolution[0], (batch_size, num_segments, events_per_segment, 1)).float()
    y_coords = torch.randint(0, resolution[1], (batch_size, num_segments, events_per_segment, 1)).float()
    segments_coords = torch.cat([x_coords, y_coords], dim=-1)
    
    print(f"\nInput Shapes:")
    print(f"  Complex features: {segments_complex.shape} (dtype: {segments_complex.dtype})")
    print(f"  Coordinates: {segments_coords.shape}")
    print(f"  Coordinate ranges: x=[{x_coords.min():.1f}, {x_coords.max():.1f}], y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            logits = model(segments_complex, segments_coords)
        
        print(f"✓ Forward pass successful!")
        print(f"\nOutput:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        
        print(f"\nPredictions:")
        for i in range(batch_size):
            print(f"  Sample {i}: Class {predicted_classes[i].item()} (confidence: {probs[i, predicted_classes[i]].item():.3f})")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)