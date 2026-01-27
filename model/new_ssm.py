import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.layers import DropPath

class ConvMambaBlock(nn.Module):
    """
    The Core Building Block: Conv1d (Local) + Mamba2 (Global) + MLP.
    Designed for variable-length sequences [B, L, D].
    """
    def __init__(self, dim, d_state=32, expand=2, kernel_size=3, padding=1, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Depthwise Conv to enhance local features (spatial/temporal jitters)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, L, D]
        res = x
        x = self.norm1(x)
        
        # 1. Local + Global Branch
        # Transpose for Conv1d: [B, D, L]
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv.contiguous()

        x = self.ssm(x_conv + x) 
        x = res + self.drop_path(x)
        
        # 2. Feed-Forward Branch
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class NestedEventMamba(nn.Module):
    def __init__(
        self, 
        encoding_dim=64, 
        hidden_dim=128, 
        num_classes=11, 
        intra_window_blocks=2,
        inter_window_blocks=2,
        intra_window_d_state=32,
        inter_window_d_state=32,
        intra_window_expand=2,
        inter_window_expand=2,
        intra_window_kernel_size=11,
        inter_window_kernel_size=3,
        intra_window_padding=5,
        inter_window_padding=1,
        dropout=0.1, 
        drop_path=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection: Complex (Real/Imag) + Normalized Coords
        self.input_adapter = nn.Sequential(
            nn.Linear(encoding_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # --- Level 1: Intra-Window Shared Blocks ---
        # These process N events within a single segment.
        self.intra_window_blocks = nn.ModuleList([
            ConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=drop_path) for _ in range(intra_window_blocks)
        ])
        
        # --- Level 2: Inter-Window (Temporal) Blocks ---
        # These process T segments across the whole sequence.
        self.inter_window_blocks = nn.ModuleList([
            ConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=drop_path) for _ in range(inter_window_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, batch):
        # batch['segments_complex']: List[List[Tensor]]
        # batch['segments_coords']: List[List[Tensor]]
        
        batch_logits = []
        for z_list, c_list in zip(batch['segments_complex'], batch['segments_coords']):
            window_vectors = []
            
            for z, c in zip(z_list, c_list):
                # 1. Pre-process Window: [N, D_complex] -> [1, N, hidden_dim]
                feat = torch.cat([z.real, z.imag, c], dim=-1)
                x = self.input_adapter(feat).unsqueeze(0)
                
                # CRITICAL: Ensure contiguous memory layout for causal_conv1d
                # Variable-length sequences can create misaligned strides
                x = x.contiguous()
                
                # 2. Intra-Window Feature Extraction (Shared)
                for block in self.intra_window_blocks:
                    x = block(x)
                
                # 3. Pooling: Compress N events to 1 vector
                # Use a mix of Max and Mean for better representation
                win_vec = (x.mean(dim=1) + x.max(dim=1)[0]) / 2
                window_vectors.append(win_vec)
            
            # 4. Global Temporal Processing
            # seq: [1, T_segments, hidden_dim]
            seq = torch.cat(window_vectors, dim=0).unsqueeze(0)
            
            # Ensure contiguous for inter-window processing
            seq = seq.contiguous()
            
            for block in self.inter_window_blocks:
                seq = block(seq)
            
            # 5. Final Classification
            # Pool across T dimension
            global_feat = seq.mean(dim=1)
            batch_logits.append(self.head(global_feat))
            
        return torch.cat(batch_logits, dim=0)

def print_model_summary(model, model_name="NestedEventMamba"):
    """
    Print a detailed summary of model architecture and parameters.
    Shows hierarchical breakdown by component and total counts.
    """
    def count_parameters(module):
        """Count trainable and total parameters in a module"""
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    def format_number(num):
        """Format number with thousands separator"""
        return f"{num:,}"
    
    print(f"\n{'='*80}")
    print(f"{model_name} - Architecture Summary")
    print(f"{'='*80}\n")
    
    # Hierarchical breakdown
    print("Layer-wise Parameter Breakdown:")
    print(f"{'─'*80}")
    print(f"{'Component':<45} {'Parameters':>15} {'Trainable':>15}")
    print(f"{'─'*80}")
    
    total_params = 0
    total_trainable = 0
    
    # Input Adapter
    if hasattr(model, 'input_adapter'):
        params, trainable = count_parameters(model.input_adapter)
        total_params += params
        total_trainable += trainable
        print(f"│ Input Adapter")
        for i, layer in enumerate(model.input_adapter):
            layer_params = sum(p.numel() for p in layer.parameters())
            if layer_params > 0:
                print(f"│   ├─ [{i}] {layer.__class__.__name__:<35} {format_number(layer_params):>15}")
        print(f"│   └─ {'Subtotal':<35} {format_number(params):>15} {format_number(trainable):>15}")
        print(f"│")
    
    # Intra-Window Blocks
    if hasattr(model, 'intra_window_blocks'):
        print(f"│ Intra-Window Blocks (x{len(model.intra_window_blocks)})")
        total_intra = 0
        for i, block in enumerate(model.intra_window_blocks):
            block_params, block_trainable = count_parameters(block)
            total_intra += block_params
            print(f"│   ├─ Block {i}:")
            
            # Norm1
            norm1_params = sum(p.numel() for p in block.norm1.parameters())
            print(f"│   │   ├─ {'LayerNorm (norm1)':<32} {format_number(norm1_params):>15}")
            
            # Local Conv
            conv_params = sum(p.numel() for p in block.local_conv.parameters())
            print(f"│   │   ├─ {'Conv1d (local_conv)':<32} {format_number(conv_params):>15}")
            
            # SSM (Mamba2)
            ssm_params = sum(p.numel() for p in block.ssm.parameters())
            print(f"│   │   ├─ {'Mamba2 (ssm)':<32} {format_number(ssm_params):>15}")
            
            # Norm2
            norm2_params = sum(p.numel() for p in block.norm2.parameters())
            print(f"│   │   ├─ {'LayerNorm (norm2)':<32} {format_number(norm2_params):>15}")
            
            # MLP
            mlp_params = sum(p.numel() for p in block.mlp.parameters())
            print(f"│   │   ├─ {'MLP':<32} {format_number(mlp_params):>15}")
            
            # DropPath
            drop_params = sum(p.numel() for p in block.drop_path.parameters())
            if drop_params > 0:
                print(f"│   │   ├─ {'DropPath':<32} {format_number(drop_params):>15}")
            
            print(f"│   │   └─ {'Block Total':<32} {format_number(block_params):>15}")
        
        total_params += total_intra
        total_trainable += total_intra
        print(f"│   └─ {'All Intra-Window Blocks':<35} {format_number(total_intra):>15} {format_number(total_intra):>15}")
        print(f"│")
    
    # Inter-Window Blocks
    if hasattr(model, 'inter_window_blocks'):
        print(f"│ Inter-Window Blocks (x{len(model.inter_window_blocks)})")
        total_inter = 0
        for i, block in enumerate(model.inter_window_blocks):
            block_params, block_trainable = count_parameters(block)
            total_inter += block_params
            print(f"│   ├─ Block {i}:")
            
            # Norm1
            norm1_params = sum(p.numel() for p in block.norm1.parameters())
            print(f"│   │   ├─ {'LayerNorm (norm1)':<32} {format_number(norm1_params):>15}")
            
            # Local Conv
            conv_params = sum(p.numel() for p in block.local_conv.parameters())
            print(f"│   │   ├─ {'Conv1d (local_conv)':<32} {format_number(conv_params):>15}")
            
            # SSM (Mamba2)
            ssm_params = sum(p.numel() for p in block.ssm.parameters())
            print(f"│   │   ├─ {'Mamba2 (ssm)':<32} {format_number(ssm_params):>15}")
            
            # Norm2
            norm2_params = sum(p.numel() for p in block.norm2.parameters())
            print(f"│   │   ├─ {'LayerNorm (norm2)':<32} {format_number(norm2_params):>15}")
            
            # MLP
            mlp_params = sum(p.numel() for p in block.mlp.parameters())
            print(f"│   │   ├─ {'MLP':<32} {format_number(mlp_params):>15}")
            
            # DropPath
            drop_params = sum(p.numel() for p in block.drop_path.parameters())
            if drop_params > 0:
                print(f"│   │   ├─ {'DropPath':<32} {format_number(drop_params):>15}")
            
            print(f"│   │   └─ {'Block Total':<32} {format_number(block_params):>15}")
        
        total_params += total_inter
        total_trainable += total_inter
        print(f"│   └─ {'All Inter-Window Blocks':<35} {format_number(total_inter):>15} {format_number(total_inter):>15}")
        print(f"│")
    
    # Classification Head
    if hasattr(model, 'head'):
        params, trainable = count_parameters(model.head)
        total_params += params
        total_trainable += trainable
        print(f"│ Classification Head")
        for i, layer in enumerate(model.head):
            layer_params = sum(p.numel() for p in layer.parameters())
            if layer_params > 0:
                print(f"│   ├─ [{i}] {layer.__class__.__name__:<35} {format_number(layer_params):>15}")
        print(f"│   └─ {'Subtotal':<35} {format_number(params):>15} {format_number(trainable):>15}")
    
    print(f"{'─'*80}")
    print(f"{'TOTAL PARAMETERS':<45} {format_number(total_params):>15} {format_number(total_trainable):>15}")
    print(f"{'─'*80}")
    
    # Memory estimation
    param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"\nModel Size Estimation:")
    print(f"  Parameters (float32): ~{param_size_mb:.2f} MB")
    print(f"  Parameters (float16): ~{param_size_mb/2:.2f} MB")
    
    # Additional model info
    print(f"\nModel Configuration:")
    if hasattr(model, 'hidden_dim'):
        print(f"  Hidden dimension: {model.hidden_dim}")
    
    print(f"{'='*80}\n")
    
    return total_params, total_trainable


if __name__ == "__main__":
    """
    Test script that mimics data generated from SparseVKMEncoder.py
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # --- Model Configuration ---
    encoding_dim = 64
    hidden_dim = 128
    num_classes = 11
    
    model = NestedEventMamba(
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        intra_window_blocks=2,
        inter_window_blocks=2,
        intra_window_d_state=32,
        inter_window_d_state=32,
        intra_window_expand=2,
        inter_window_expand=1.5,
        dropout=0.1,
        drop_path=0.1
    ).to(device)
    
    # Print detailed model summary
    total_params, trainable_params = print_model_summary(model)
    
    # --- Generate Synthetic Batch Data ---
    # Mimicking SparseVKMEncoder output format:
    # - Complex embeddings: [N_events, encoding_dim] dtype=cfloat
    # - Coordinates: [N_events, 2] dtype=float (y, x)
    
    batch_size = 4
    num_segments_per_sample = 8  # T temporal segments
    
    batch = {
        'segments_complex': [],
        'segments_coords': []
    }
    
    for b in range(batch_size):
        sample_segments_complex = []
        sample_segments_coords = []
        
        for seg in range(num_segments_per_sample):
            # Varying number of events per segment (between 50-200)
            num_events = torch.randint(50, 200, (1,)).item()
            
            # Generate complex embeddings (mimicking VecKMSparse output)
            # Complex numbers have both real and imaginary parts
            real_part = torch.randn(num_events, encoding_dim, device=device)
            imag_part = torch.randn(num_events, encoding_dim, device=device)
            complex_emb = torch.complex(real_part, imag_part)
            
            # Generate coordinates (y, x) in pixel space
            # DVS346 resolution: 346x260 (as seen in forward pass normalization)
            coords = torch.stack([
                torch.randint(0, 260, (num_events,), device=device, dtype=torch.float32),  # y
                torch.randint(0, 346, (num_events,), device=device, dtype=torch.float32)   # x
            ], dim=1)
            
            sample_segments_complex.append(complex_emb)
            sample_segments_coords.append(coords)
        
        batch['segments_complex'].append(sample_segments_complex)
        batch['segments_coords'].append(sample_segments_coords)
    
    print(f"\nBatch structure:")
    print(f"  Batch size: {batch_size}")
    print(f"  Segments per sample: {num_segments_per_sample}")
    print(f"  Sample segment shapes:")
    print(f"    Complex: {batch['segments_complex'][0][0].shape} (dtype: {batch['segments_complex'][0][0].dtype})")
    print(f"    Coords: {batch['segments_coords'][0][0].shape} (dtype: {batch['segments_coords'][0][0].dtype})")
    
    # --- Forward Pass ---
    print("\n--- Running Forward Pass ---")
    model.eval()
    
    import time
    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        logits = model(batch)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
    
    print(f"\nResults:")
    print(f"  Output shape: {logits.shape}")  # Should be [batch_size, num_classes]
    print(f"  Inference time: {(end - start)*1000:.2f}ms")
    print(f"  Per-sample time: {(end - start)*1000/batch_size:.2f}ms")
    
    # Check output
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print(f"\nPredictions: {predictions.cpu().numpy()}")
    print(f"Confidences: {probs.max(dim=1)[0].cpu().numpy()}")
    
    print("\n✓ Test completed successfully!")