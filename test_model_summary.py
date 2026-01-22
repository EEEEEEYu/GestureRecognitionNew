"""
Standalone test to demonstrate the model parameter summary.
This creates a mock version of the model to show what the summary would look like.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MockMamba2(nn.Module):
    """Mock Mamba2 for testing (approximates actual parameter count)"""
    def __init__(self, d_model=128, d_state=32, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # Approximate Mamba2 structure
        d_inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
    
    def forward(self, x):
        return x

class MockDropPath(nn.Module):
    """Mock DropPath"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return x

class ConvMambaBlock(nn.Module):
    """Same as in new_ssm.py but with mock Mamba2"""
    def __init__(self, dim, d_state=32, expand=2, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.ssm = MockMamba2(d_model=dim, d_state=d_state, expand=expand)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = MockDropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.ssm(x_conv + x) 
        x = res + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MockNestedEventMamba(nn.Module):
    """Mock version of NestedEventMamba for testing"""
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
        inter_window_expand=1.5,
        dropout=0.1, 
        drop_path=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input adapter
        self.input_adapter = nn.Sequential(
            nn.Linear(encoding_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Intra-window blocks
        self.intra_window_blocks = nn.ModuleList([
            ConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=drop_path) 
            for _ in range(intra_window_blocks)
        ])
        
        # Inter-window blocks
        self.inter_window_blocks = nn.ModuleList([
            ConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=drop_path) 
            for _ in range(inter_window_blocks)
        ])
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return x

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
    print("Creating MockNestedEventMamba model for parameter analysis...")
    
    model = MockNestedEventMamba(
        encoding_dim=64,
        hidden_dim=128,
        num_classes=11,
        intra_window_blocks=2,
        inter_window_blocks=2,
        intra_window_d_state=32,
        inter_window_d_state=32,
        intra_window_expand=2,
        inter_window_expand=1.5,
        dropout=0.1,
        drop_path=0.1
    )
    
    # Print the summary
    total_params, trainable_params = print_model_summary(model, "MockNestedEventMamba")
    
    print("NOTE: This is a mock version using simplified Mamba2.")
    print("      Actual parameter counts may vary slightly with real mamba_ssm.Mamba2")
    print("\nKey insights:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Model is relatively lightweight for event-based processing")
    print(f"  • Most parameters are in the Mamba2 SSM blocks")
    print(f"  • Input adapter converts complex embeddings to hidden representation")
    print(f"  • Two-level hierarchy: intra-window (local) → inter-window (temporal)")
