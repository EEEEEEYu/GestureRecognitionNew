import torch
import torch.nn as nn
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock Mamba before importing the model
sys.modules["mamba_ssm"] = MagicMock()
class MockMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.proj(x)
sys.modules["mamba_ssm"].Mamba = MockMamba

# Add valid path for imports if needed, assuming we run from root
sys.path.append(os.getcwd())

from model.semi_optimized_new_ssm import NestedEventMamba

def test_forward_cpu():
    print("Testing NestedEventMamba (Semi-Optimized) on CPU with MockMamba...")
    device = 'cpu'
    
    # Configuration
    encoding_dim = 64
    hidden_dim = 128
    num_classes = 11
    
    # Enable Gradient Checkpointing
    model = NestedEventMamba(
        encoding_dim=encoding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        intra_window_blocks=2,
        inter_window_blocks=2,
        use_checkpointing=True
    ).to(device)
    
    # Create valid dummy data
    batch_size = 2
    num_segments = 3
    
    batch = {
        'segments_complex': [],
        'segments_coords': []
    }
    
    for _ in range(batch_size):
        seg_complex = []
        seg_coords = []
        for _ in range(num_segments):
            # Variable events
            num_events = torch.randint(10, 30, (1,)).item()
             # Complex numbers have both real and imaginary parts
            real_part = torch.randn(num_events, encoding_dim, device=device)
            imag_part = torch.randn(num_events, encoding_dim, device=device)
            c_emb = torch.complex(real_part, imag_part)
            
            coords = torch.randn(num_events, 2, device=device)
            
            seg_complex.append(c_emb)
            seg_coords.append(coords)
        
        batch['segments_complex'].append(seg_complex)
        batch['segments_coords'].append(seg_coords)
        
    # Forward
    logits = model(batch)
    print(f"Output shape: {logits.shape}")
    
    assert logits.shape == (batch_size, num_classes)
    
    # Backward check (Verification for checkpointing)
    print("Running backward pass...")
    loss = logits.sum()
    loss.backward()
    print("Backward pass successful!")

    print("Test Passed!")

if __name__ == "__main__":
    test_forward_cpu()
