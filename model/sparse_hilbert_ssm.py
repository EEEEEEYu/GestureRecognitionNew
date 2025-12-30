"""
3D Sparse Hilbert SSM Model for Event-based Gesture Recognition.

This model uses 6 different spatial-temporal traversal orders to process
event vectors in XYT space, mimicking sparse 3D Hilbert curves.

Architecture:
- 6 Mamba2 blocks (one per traversal order: XYT, XTY, YXT, YTX, TXY, TYX)
- Dropout and layer normalization for regularization
- Multi-scale pooling
- MLP classifier

Reference: mamba-ssm (https://github.com/state-spaces/mamba)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2
from typing import List, Tuple



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class SpatialTemporalOrdering(nn.Module):
    """Reorder vectors based on different XYT traversal orders."""
    
    def __init__(self):
        super().__init__()
        # Define 6 traversal orders (permutations of XYT axes)
        self.orders = {
            'XYT': (0, 1, 2),  # x, y, then time
            'XTY': (0, 2, 1),  # x, time, then y
            'YXT': (1, 0, 2),  # y, x, then time
            'YTX': (1, 2, 0),  # y, time, then x
            'TXY': (2, 0, 1),  # time, x, then y
            'TYX': (2, 1, 0),  # time, y, then x
        }
    
    def get_ordering(self, event_coords: np.ndarray, order_name: str) -> np.ndarray:
        """
        Get indices for ordering vectors based on traversal pattern.
        
        Args:
            event_coords: [N, 4] array with [x, y, t, p]
            order_name: One of 'XYT', 'XTY', 'YXT', 'YTX', 'TXY', 'TYX'
        
        Returns:
            Indices for reordering [N]
        """
        axes = self.orders[order_name]
        
        # Use lexsort for stable multi-key sorting
        # lexsort sorts by last key first, so reverse the order
        sort_keys = tuple(event_coords[:, axis] for axis in reversed(axes))
        indices = np.lexsort(sort_keys)
        
        return indices


class Mamba2Block(nn.Module):
    """
    Single Mamba2 block with input projection, SSM, and output projection.
    Includes dropout and layer norm for regularization.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint
        
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def _forward_impl(self, x):
        """Actual forward implementation."""
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x
    
    def forward(self, x):
        """
        Args:
            x: [batch, length, d_model]
        Returns:
            [batch, length, d_model]
        """
        residual = x
        
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x = self._forward_impl(x)
        
        return residual + self.drop_path(x)


class MultiDirectionalSSM(nn.Module):
    """
    Multi-directional SSM that processes sequences in 6 different orders.
    
    This captures different spatial-temporal dependencies by traversing
    the sparse 3D event space in all permutations of XYT axes.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        num_layers: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.ordering = SpatialTemporalOrdering()
        self.order_names = ['XYT', 'XTY', 'YXT', 'YTX', 'TXY', 'TYX']
        
        # Create separate Mamba2 stacks for each direction
        self.direction_models = nn.ModuleDict()
        
        # Stochastic depth decay rule: linear decay of drop_path rate
        # We process layers sequentially in each stack, so we distribute drop_prob 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]

        for order_name in self.order_names:
            layers = nn.ModuleList([
                Mamba2Block(d_model, d_state, d_conv, expand, dropout, 
                          drop_path=dp_rates[i], use_checkpoint=use_checkpoint)
                for i in range(num_layers)
            ])
            self.direction_models[order_name] = layers
        
        # Fusion layer to combine multi-directional outputs
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * len(self.order_names)),
            nn.Linear(d_model * len(self.order_names), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, vectors, event_coords):
        """
        Process vectors in multiple directions with explicit batch dimension.
        
        Args:
            vectors: [batch, length, d_model] tensor (typically batch=1)
            event_coords: [length, 4] numpy array with [x, y, t, p]
        
        Returns:
            [batch, length, d_model] tensor after multi-directional processing
        """
        # vectors should already be [B, L, C] from caller
        batch_size = vectors.shape[0]
        
        # Process in each direction
        direction_outputs = []
        
        for order_name in self.order_names:
            # Get ordering indices
            indices = self.ordering.get_ordering(event_coords, order_name)
            indices_tensor = torch.from_numpy(indices).to(vectors.device)
            
            # Reorder vectors: [B, L, C] → [B, L_reordered, C]
            x = vectors[:, indices_tensor, :]
            
            # Pass through Mamba2 layers
            for layer in self.direction_models[order_name]:
                x = layer(x)
            
            # Reverse ordering to original positions
            inverse_indices = torch.argsort(indices_tensor)
            x = x[:, inverse_indices, :]
            
            direction_outputs.append(x)
        
        # Concatenate all directions: [B, L, C*6]
        multi_dir = torch.cat(direction_outputs, dim=-1)
        
        # Fuse directions: [B, L, C*6] → [B, L, C]
        fused = self.fusion(multi_dir)
        
        return fused


class SparseHilbertSSM(nn.Module):
    """
    Complete model for event-based gesture recognition using 3D SSM.
    
    Architecture:
    1. Input projection (complex embeddings → real features)
    2. Multi-directional SSM (6 traversal orders)
    3. Multi-scale temporal pooling
    4. MLP classifier with dropout
    """
    
    def __init__(
        self,
        encoding_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_layers: int = 3,  # Renamed from num_ssm_layers
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        pooling_scales: List[int] = [1, 2, 4],
        use_checkpoint: bool = True,  # Enable by default for memory efficiency
        input_meta: dict = None,  # Optional metadata
    ):
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        
        # Input projection: complex embeddings (2 * encoding_dim) → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(2 * encoding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multi-directional SSM
        self.ssm = MultiDirectionalSSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_path=drop_path,
            num_layers=num_layers,
            use_checkpoint=use_checkpoint,
        )
        
        # Multi-scale pooling
        self.pooling_scales = pooling_scales
        pooled_dim = hidden_dim * len(pooling_scales)
        
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def multi_scale_pool(self, features):
        """
        Apply multi-scale temporal pooling.
        
        Args:
            features: [length, hidden_dim]
        
        Returns:
            [pooled_dim] pooled features
        """
        pooled_features = []
        
        for scale in self.pooling_scales:
            if scale == 1:
                # Global average pooling
                pooled = features.mean(dim=0)
            else:
                # Stride pooling + average
                length = features.shape[0]
                stride = max(1, length // scale)
                windows = []
                for i in range(0, length, stride):
                    window = features[i:min(i + stride, length)]
                    windows.append(window.mean(dim=0))
                
                if len(windows) > 0:
                    pooled = torch.stack(windows).mean(dim=0)
                else:
                    pooled = features.mean(dim=0)
            
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=0)
    
    def forward_single(self, vectors, event_coords):
        """
        Process a single sample with explicit batch dimension.
        
        Args:
            vectors: [length, encoding_dim] complex tensor
            event_coords: [length, 4] numpy array
        
        Returns:
            logits: [num_classes]
        """
        # Input projection
        # Stack real and imaginary parts: [L, encoding_dim] → [L, 2*encoding_dim]
        x = torch.cat([vectors.real, vectors.imag], dim=-1)
        
        # Add explicit batch dimension: [L, 2*encoding_dim] → [1, L, 2*encoding_dim]
        x = x.unsqueeze(0)
        
        # Input projection: [1, L, 2*encoding_dim] → [1, L, hidden_dim]
        x = self.input_proj(x)
        
        # Multi-directional SSM: [1, L, hidden_dim] → [1, L, hidden_dim]
        x = self.ssm(x, event_coords)
        
        # Remove batch dimension: [1, L, hidden_dim] → [L, hidden_dim]
        x = x.squeeze(0)
        
        # Multi-scale pooling: [L, hidden_dim] → [pooled_dim]
        pooled = self.multi_scale_pool(x)
        
        # Classification: [pooled_dim] → [num_classes]
        logits = self.classifier(pooled)
        
        return logits
    
    def forward(self, batch):
        """
        Process a batch of variable-length samples.
        
        Args:
            batch: Dictionary with:
                - vectors: List[Tensor] of shape [length_i, encoding_dim]
                - event_coords: List[ndarray] of shape [length_i, 4]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        vectors_list = batch['vectors']
        coords_list = batch['event_coords']
        
        logits_list = []
        
        for i, (vectors, coords) in enumerate(zip(vectors_list, coords_list)):
            logits = self.forward_single(vectors, coords)
            logits_list.append(logits)
            
            # Clear GPU cache between samples to prevent OOM
            if torch.cuda.is_available() and i % 4 == 3:  # Every 4 samples
                torch.cuda.empty_cache()
        
        return torch.stack(logits_list)  # [batch_size, num_classes]


# Model factory function
def create_sparse_hilbert_ssm(config):
    """Create model from config dictionary."""
    model_args = config['MODEL']['model_init_args']
    
    return SparseHilbertSSM(
        encoding_dim=model_args.get('encoding_dim', 64),
        hidden_dim=model_args.get('hidden_dim', 256),
        num_classes=model_args.get('num_classes', 11),
        num_layers=model_args.get('num_layers', 3),
        d_state=model_args.get('d_state', 64),
        d_conv=model_args.get('d_conv', 4),
        expand=model_args.get('expand', 2),
        dropout=model_args.get('dropout', 0.2),
        drop_path=model_args.get('drop_path', 0.0),
        pooling_scales=model_args.get('pooling_scales', [1, 2, 4]),
        use_checkpoint=model_args.get('use_checkpoint', True),
    )
