import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.layers import DropPath
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
import math

class LearnedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, x, mask=None):
        # x: [B, T, D]
        # mask: [B, T] (True if valid)
        
        attn_logits = self.pool_proj(x) # [B, T, 1]
        
        if mask is not None:
             mask_expanded = mask.unsqueeze(-1) # [B, T, 1]
             # mask is boolean: True = keep, False = mask out
             # We want to fill *invalid* (False) positions with -inf
             attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))
        
        weights = F.softmax(attn_logits, dim=1) # [B, T, 1]
        x_pooled = (x * weights).sum(dim=1) # [B, D]
        return x_pooled

class BidirectionalConvMambaBlock(nn.Module):
    """
    Gated Parallel Block Architecture:
    w = softmax(branch_logits)
    y = w[0] * Mixer(Norm(x)) + w[1] * MLP(Norm(x))
    x = x + DropPath(y)
    Designed for variable-length sequences [B, L, D].
    """
    def __init__(self, dim, d_state=32, expand=2, dropout=0.1, drop_path=0.1, kernel_size=3, padding=1):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        
        # --- Mixer Branch ---
        # 1. Depthwise Conv
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.ln_conv = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        
        # 2. SSM
        self.ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        # 3. Post-SSM Conv
        self.post_conv = nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1, groups=dim)
        self.ln_post = nn.LayerNorm(dim * 2)
        self.act_post = nn.SiLU()
        
        # --- MLP Branch ---
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Gating
        self.branch_logits = nn.Parameter(torch.tensor([1.0, 0.1]))

    def forward(self, x):
        # x: [B, L, D]
        
        # Shared Norm
        x_norm = self.norm(x)
        
        # --- Mixer Branch Execution ---
        # Local Conv: [B, L, D] -> [B, D, L]
        x_t = x_norm.transpose(1, 2)
        x_c = self.local_conv(x_t)
        x_c = x_c.transpose(1, 2)
        x_c = self.act1(self.ln_conv(x_c)) 
        
        x_c_in = x_c.contiguous()
        
        # Global SSM
        x_s = self.ssm(x_c_in)
        x_s = x_s + x_norm
        x_c_in_reversed = x_c_in.flip(1)
        x_norm_reversed = x_norm.flip(1)
        x_s_reversed = self.ssm(x_c_in_reversed)
        x_s_reversed = x_s_reversed + x_norm_reversed
        x_s_reversed = x_s_reversed.flip(1)
        x_s = torch.cat([x_s, x_s_reversed], dim=-1)
        
        # Post-SSM Conv
        x_s_norm = self.ln_post(x_s)
        x_s_t = x_s_norm.transpose(1, 2)
        x_mixer = self.post_conv(x_s_t)
        x_mixer = x_mixer.transpose(1, 2)
        x_mixer = self.act_post(x_mixer)
        
        # --- MLP Branch Execution ---
        x_mlp = self.mlp(x_norm)
        
        # --- Gated Fusion ---
        weights = F.softmax(self.branch_logits, dim=0)
        y = weights[0] * x_mixer + weights[1] * x_mlp
        
        x = x + self.drop_path(y)
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
        drop_path=0.1,
        use_checkpointing=False,
        use_bidirectional=True # Parameter kept for compatibility but ignored/implied True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing
        
        # Initial projection
        self.input_adapter = nn.Sequential(
            nn.Linear(encoding_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # --- Level 1: Intra-Window Shared Blocks (Parallel) ---
        self.intra_window_blocks = nn.ModuleList()
        
        original_drop_path = drop_path
        original_dropout = dropout
        
        # Total blocks for decay calculation
        total_blocks = intra_window_blocks + inter_window_blocks
        current_layer_idx = 0
        
        for i in range(intra_window_blocks):
            # Stochastic Depth decay: Linear scaling (1/N, 2/N, ... N/N)
            dpr = original_drop_path * ((current_layer_idx + 1) / total_blocks)
            
            # Always use Bidirectional
            self.intra_window_blocks.append(BidirectionalConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=dpr, kernel_size=intra_window_kernel_size, padding=intra_window_padding))
            current_layer_idx += 1
        
        # --- Level 2: Inter-Window (Flat) ---
        
        self.inter_window_blocks = nn.ModuleList()
        
        for i in range(inter_window_blocks):
            # Stochastic Depth decay: Linear scaling
            dpr = original_drop_path * ((current_layer_idx + 1) / total_blocks)
            
            # Always use Bidirectional
            self.inter_window_blocks.append(BidirectionalConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=dpr, kernel_size=inter_window_kernel_size, padding=inter_window_padding))
            current_layer_idx += 1
            
        # Learned Temporal Pooling
        self.intra_pool = LearnedPooling(hidden_dim)
        self.inter_pool = LearnedPooling(hidden_dim)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2), # As updated by user
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    @property
    def no_weight_decay(self):
        """
        Return a set of parameter names that should not be subject to weight decay.
        Includes:
        - pool_proj (Attention Poolings)
        - branch_logits (Gating)
        - SSM parameters (A_log, D, dt_bias)
        """
        nwd = set()
        for name, _ in self.named_parameters():
            if 'pool_proj' in name:
                nwd.add(name)
            if 'branch_logits' in name:
                nwd.add(name)
            if 'A_log' in name or 'D' in name or 'dt_bias' in name:
                nwd.add(name)
        return nwd

    def forward(self, batch):
        segments_complex = batch['segments_complex']
        segments_coords = batch['segments_coords']
        
        # ... (Intra-Window Logic: Unchanged conceptually) ...
        
        sample_vectors_list = []
        
        for z_list, c_list in zip(segments_complex, segments_coords):
            if len(z_list) == 0: continue
            device = z_list[0].device
            
            processed_segments = []
            for z, c in zip(z_list, c_list):
                 processed_segments.append(torch.cat([z.real, z.imag, c], dim=-1))
            
            segment_lengths = torch.tensor([s.shape[0] for s in processed_segments], device=device)
            max_events = segment_lengths.max().item()
            padded_inputs = pad_sequence(processed_segments, batch_first=True, padding_value=0.0)
            mask = torch.arange(max_events, device=device)[None, :] < segment_lengths[:, None]
            
            x = self.input_adapter(padded_inputs)
            x = x.contiguous()
            
            for block in self.intra_window_blocks:
                if self.use_checkpointing and x.requires_grad:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            
            # Intra-Window Learned Pooling
            window_vectors = self.intra_pool(x, mask)
            sample_vectors_list.append(window_vectors)
        
        # --- Inter-Window Pass (Flat) ---
        
        if not sample_vectors_list:
             # Handle empty batch case properly with correct dim
             return torch.zeros(len(segments_complex), self.head[-1].out_features, device=self.intra_window_blocks[0].local_conv.weight.device)
             
        device = sample_vectors_list[0].device
        
        padded_seqs = pad_sequence(sample_vectors_list, batch_first=True, padding_value=0.0)
        seq_lengths = torch.tensor([s.shape[0] for s in sample_vectors_list], device=device)
        max_segments = seq_lengths.max().item()
        seq_mask = torch.arange(max_segments, device=device)[None, :] < seq_lengths[:, None]
        
        seq = padded_seqs.contiguous()
        
        for block in self.inter_window_blocks:
            if self.use_checkpointing and seq.requires_grad:
                seq = checkpoint(block, seq, use_reentrant=False)
            else:
                seq = block(seq)
        
        # Inter-Window Learned Pooling
        global_feat = self.inter_pool(seq, seq_mask)
        
        logits = self.head(global_feat)
        
        return logits
