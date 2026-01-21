import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from typing import List, Dict
from timm.layers import DropPath

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
    def __init__(self, dim, resolution, dropout=0.1, drop_path_rate=0.0):
        super().__init__()
        self.width, self.height = resolution
        # Shared SSM for spatial scanning
        self.spatial_ssm = Mamba2(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x, coords):
        # x: [Batch, N_events, Dim]
        # coords: [Batch, N_events, 2] (x, y)
        
        residual_input = x
        
        # 1. Generate Sort Indices for XY and YX orders
        # First scan: Y-major order (y * W + x)
        sort_idx_yx = torch.argsort(coords[..., 1] * self.width + coords[..., 0], dim=1)
        x_sorted_yx = torch.gather(x, 1, sort_idx_yx.unsqueeze(-1).expand_as(x))
        
        # 2. Spatial Scan in YX order
        out_yx = self.spatial_ssm(x_sorted_yx)
        out_yx = self.dropout(out_yx)
        
        # Second scan: X-major order (x * H + y) for bidirectional processing
        sort_idx_xy = torch.argsort(coords[..., 0] * self.height + coords[..., 1], dim=1)
        x_sorted_xy = torch.gather(x, 1, sort_idx_xy.unsqueeze(-1).expand_as(x))
        out_xy = self.spatial_ssm(x_sorted_xy)
        out_xy = self.dropout(out_xy)
        
        # Unsort both outputs back to original order
        inverse_idx_yx = torch.argsort(sort_idx_yx, dim=1)
        out_yx_unsorted = torch.gather(out_yx, 1, inverse_idx_yx.unsqueeze(-1).expand_as(out_yx))
        
        inverse_idx_xy = torch.argsort(sort_idx_xy, dim=1)
        out_xy_unsorted = torch.gather(out_xy, 1, inverse_idx_xy.unsqueeze(-1).expand_as(out_xy))
        
        # Combine both directional scans with residual
        out = (out_yx_unsorted + out_xy_unsorted) / 2
        
        # Apply DropPath for regularization
        out = self.drop_path(out)
        
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
    The Main Model with support for variable-length sequences.
    """
    def __init__(self, encoding_dim=64, hidden_dim=128, resolution=(346, 260), 
                 num_classes=11, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.width, self.height = resolution
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.input_adapter = ComplexPolarInput(encoding_dim, hidden_dim)
        
        # Two independent Local Aggregators (One for Mag, One for Phase)
        self.local_ssm_mag = LocalHilbertAggregator(hidden_dim, resolution, dropout, drop_path)
        self.local_ssm_phase = LocalHilbertAggregator(hidden_dim, resolution, dropout, drop_path)
        
        # Global Time Mamba (One for each stream) with dropout and drop_path
        self.global_ssm_mag = Mamba2(d_model=hidden_dim, d_state=64, expand=2)
        self.global_ssm_phase = Mamba2(d_model=hidden_dim, d_state=64, expand=2)
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path_global = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Fusion
        self.fusion = SymmetricPolarGating(hidden_dim)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, segments_complex, segments_coords):
        """
        Process a single sample with variable-length segments.
        
        Args:
            segments_complex: [T_segs, N_events, D] complex tensor
            segments_coords: [T_segs, N_events, 2] coordinates
        
        Returns:
            [hidden_dim] pooled feature vector
        """
        T, N, D = segments_complex.shape
        
        # Add batch dimension for processing: [1, T, N, D]
        z_batch = segments_complex.unsqueeze(0)
        coords_batch = segments_coords.unsqueeze(0)
        
        # Flatten to process segments: [T, N, D]
        z_flat = z_batch.view(T, N, D)
        coords_flat = coords_batch.view(T, N, 2)
        
        # Normalize coordinates to [0, 1] range
        x_norm = coords_flat[..., 0:1] / self.width
        y_norm = coords_flat[..., 1:2] / self.height
        
        # 1. Project to Polar Streams
        f_mag, f_phase = self.input_adapter(z_flat, x_norm, y_norm)
        
        # 2. Local Aggregation (Spatial)
        # Returns [T, Hidden]
        seg_mag = self.local_ssm_mag(f_mag, coords_flat)
        seg_phase = self.local_ssm_phase(f_phase, coords_flat)
        
        # Add batch dimension for global SSM: [1, T, Hidden]
        seq_mag = seg_mag.unsqueeze(0)
        seq_phase = seg_phase.unsqueeze(0)
        
        # 3. Global Aggregation (Temporal)
        # Mamba scans across time T
        global_mag = self.global_ssm_mag(seq_mag)
        global_mag = self.dropout(global_mag)
        global_mag = self.drop_path_global(global_mag)
        
        global_phase = self.global_ssm_phase(seq_phase)
        global_phase = self.dropout(global_phase)
        global_phase = self.drop_path_global(global_phase)
        
        # 4. Symmetric Gating
        fused_features = self.fusion(global_mag, global_phase)
        
        # 5. Pool over time: [1, T, Hidden] -> [Hidden]
        pooled = fused_features.squeeze(0).mean(dim=0)
        
        return pooled
    
    def forward(self, batch):
        """
        Process a batch of variable-length samples in UNIFIED FORMAT.
        
        Args:
            batch: Dictionary with UNIFIED FORMAT:
                - 'segments_complex': List[List[Tensor]] - batch of segment lists
                - 'segments_coords': List[List[Tensor]] - batch of coord lists
        
        Returns:
            logits: [batch_size, num_classes]
        """
        segments_complex_list = batch['segments_complex']
        segments_coords_list = batch['segments_coords']
        
        features_list = []
        for i, (segs_list, coords_list) in enumerate(zip(segments_complex_list, segments_coords_list)):
            # Stack segments into proper tensor format
            # segs_list is List[Tensor] where each tensor is [N_i, D]
            # Need to create [T_segments, N_max, D] with padding
            
            if len(segs_list) > 0:
                # Find max events per segment
                max_events = max(seg.shape[0] for seg in segs_list)
                T_segments = len(segs_list)
                D = segs_list[0].shape[-1]
                
                # Create padded tensors
                segments_complex_padded = torch.zeros(T_segments, max_events, D, 
                                                     dtype=segs_list[0].dtype, 
                                                     device=segs_list[0].device)
                segments_coords_padded = torch.zeros(T_segments, max_events, 2,
                                                     dtype=coords_list[0].dtype,
                                                     device=coords_list[0].device)
                
                # Fill in the actual data
                for t, (seg, coord) in enumerate(zip(segs_list, coords_list)):
                    n_events = seg.shape[0]
                    segments_complex_padded[t, :n_events] = seg
                    segments_coords_padded[t, :n_events] = coord
            else:
                # Empty sample
                segments_complex_padded = torch.zeros(1, 1, self.encoding_dim, dtype=torch.cfloat)
                segments_coords_padded = torch.zeros(1, 1, 2, dtype=torch.float32)
            
            # Extract features for this sample
            feat = self.extract_features(segments_complex_padded, segments_coords_padded)
            features_list.append(feat)
            
            # Clear GPU cache between samples to prevent OOM
            if torch.cuda.is_available() and i % 4 == 3:
                torch.cuda.empty_cache()
        
        # Stack features: [batch_size, hidden_dim]
        features = torch.stack(features_list)
        
        # Classify
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    """
    Test the SpaceTimePolarSSM model with variable-length sequences.
    """
    print("=" * 70)
    print("Testing SpaceTimePolarSSM Model with Variable-Length Support")
    print("=" * 70)
    
    # Model configuration
    encoding_dim = 64
    hidden_dim = 128
    resolution = (346, 260)  # DVS camera resolution (width, height)
    num_classes = 11
    batch_size = 3
    dropout = 0.1
    drop_path = 0.1
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model with dropout and drop_path
    model = SpaceTimePolarSSM(
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim,
        resolution=resolution,
        num_classes=num_classes,
        dropout=dropout,
        drop_path=drop_path
    )
    model = model.to(device)
    model.eval()
    
    print(f"\nModel Configuration:")
    print(f"  Encoding Dim: {encoding_dim}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Dropout: {dropout}")
    print(f"  Drop Path: {drop_path}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # ========================================================================
    # Test: Variable-length batch
    # ========================================================================
    print("\n" + "=" * 70)
    print("Variable-Length Batch Test")
    print("=" * 70)
    
    print(f"\nGenerating variable-length test data:")
    print(f"  Batch size: {batch_size}")
    
    # Create variable-length samples (different number of segments per sample)
    variable_segments = [8, 12, 10]  # Different lengths for each sample
    events_per_segment = 50
    
    segments_complex_list = []
    segments_coords_list = []
    
    for i, num_segs in enumerate(variable_segments):
        # Create complex features for this sample
        mag = torch.rand(num_segs, events_per_segment, encoding_dim) * 2
        ph = torch.rand(num_segs, events_per_segment, encoding_dim) * 2 * 3.14159
        seg_complex = mag * torch.exp(1j * ph)
        seg_complex = seg_complex.to(torch.complex64).to(device)
        
        # Create coordinates for this sample
        x_c = torch.randint(0, resolution[0], (num_segs, events_per_segment, 1)).float().to(device)
        y_c = torch.randint(0, resolution[1], (num_segs, events_per_segment, 1)).float().to(device)
        seg_coords = torch.cat([x_c, y_c], dim=-1)
        
        segments_complex_list.append(seg_complex)
        segments_coords_list.append(seg_coords)
        
        print(f"  Sample {i}: {num_segs} segments x {events_per_segment} events = {num_segs * events_per_segment} total events")
    
    # Create batch dictionary
    batch_dict = {
        'segments_complex': segments_complex_list,
        'segments_coords': segments_coords_list
    }
    
    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            logits = model(batch_dict)
        
        print(f"✓ Forward pass successful!")
        print(f"\nOutput:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        
        print(f"\nPredictions:")
        for i in range(batch_size):
            print(f"  Sample {i} ({variable_segments[i]} segs): Class {predicted_classes[i].item()} (confidence: {probs[i, predicted_classes[i]].item():.3f})")
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)