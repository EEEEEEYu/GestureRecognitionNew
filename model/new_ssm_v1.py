import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.layers import DropPath
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
import math

class BidirectionalConvMambaBlock(nn.Module):
    """
    Parallel Block Architecture:
    x = x + DropPath(Mixer(Norm(x)) + MLP(Norm(x)))
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
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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
        x_s = self.ssm(x_c_in + x_norm)
        x_c_in_reversed = x_c_in.flip(1)
        x_norm_reversed = x_norm.flip(1)
        x_s_reversed = self.ssm(x_c_in_reversed + x_norm_reversed)
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
        
        # --- Parallel Fusion ---
        x = x + self.drop_path(x_mixer + x_mlp)
        return x

class ConvMambaBlock(nn.Module):
    """
    Parallel Block Architecture:
    x = x + DropPath(Mixer(Norm(x)) + MLP(Norm(x)))
    Designed for variable-length sequences [B, L, D].
    """
    def __init__(self, dim, d_state=32, expand=2, dropout=0.1, drop_path=0.1, kernel_size=3, padding=1):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        
        # --- Mixer Branch ---
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.ln_conv = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        
        self.ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        self.post_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.ln_post = nn.LayerNorm(dim)
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

    def forward(self, x):
        # x: [B, L, D]
        
        # Shared Norm
        x_norm = self.norm(x)
        
        # --- Mixer Branch Execution ---
        # Local Conv
        x_t = x_norm.transpose(1, 2)
        x_c = self.local_conv(x_t)
        x_c = x_c.transpose(1, 2)
        x_c = self.act1(self.ln_conv(x_c)) 
        
        x_c_in = x_c.contiguous()
        
        # Global SSM
        x_s = self.ssm(x_c_in + x_norm)
        
        # Post-SSM Conv
        x_s_t = x_s.transpose(1, 2)
        x_mixer = self.post_conv(x_s_t)
        x_mixer = x_mixer.transpose(1, 2)
        x_mixer = self.act_post(self.ln_post(x_mixer))
        
        # --- MLP Branch Execution ---
        x_mlp = self.mlp(x_norm)
        
        # --- Parallel Fusion ---
        x = x + self.drop_path(x_mixer + x_mlp)
        return x

class DownsampleBlock(nn.Module):
    """
    Hierarchical Downsampling with Conv1d:
    Merges 2 temporal steps and doubles the dimension.
    [B, T, D] -> [B, T/2, 2*D]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv1d(dim, 2 * dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        if T < 2 or T % 2 != 0:
             # Pad with last element if odd or too short (ensure T >= 2 and even)
             x = torch.cat([x, x[:, -1:]], dim=1)
             
        # [B, T, D] -> [B, D, T] for Conv1d
        x = x.transpose(1, 2)
        x = self.reduction(x)
        
        # [B, 2D, T/2] -> [B, T/2, 2D]
        x = x.transpose(1, 2)
        x = self.norm(x)
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
        use_bidirectional=False # Kept for signature compatibility
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
        
        for i in range(intra_window_blocks):
            # Stochastic Depth decay
            dpr = original_drop_path * (i / (intra_window_blocks + inter_window_blocks)) 
            
            if use_bidirectional:
                 self.intra_window_blocks.append(BidirectionalConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=dpr, kernel_size=intra_window_kernel_size, padding=intra_window_padding))
            else:
                 self.intra_window_blocks.append(ConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=dpr, kernel_size=intra_window_kernel_size, padding=intra_window_padding))
        
        # --- Level 2: Inter-Window (Hierarchical) ---
        # Stage 1 Blocks -> Downsample -> Stage 2 Blocks
        
        self.stage1_blocks = nn.ModuleList()
        self.stage2_blocks = nn.ModuleList()
        
        # Split inter_window_blocks: First half Stage 1, Second half Stage 2
        # Ensure at least 1 block per stage if possible, otherwise put all in Stage 2?
        # Robust logic:
        n_stage1 = max(1, inter_window_blocks // 2)
        n_stage2 = inter_window_blocks - n_stage1
        if n_stage2 == 0: n_stage2 = 1 # Force at least one block in final stage if count allows, technically changes total blocks if input was 1
        
        # Total blocks for decay calculation
        total_blocks = intra_window_blocks + n_stage1 + n_stage2
        current_layer_idx = intra_window_blocks
        
        # Stage 1
        for i in range(n_stage1):
            dpr = original_drop_path * (current_layer_idx / total_blocks)
            if use_bidirectional:
                self.stage1_blocks.append(BidirectionalConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=dpr, kernel_size=inter_window_kernel_size, padding=inter_window_padding))
            else:
                self.stage1_blocks.append(ConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=dpr, kernel_size=inter_window_kernel_size, padding=inter_window_padding))
            current_layer_idx += 1
            
        # Downsample
        self.downsample = DownsampleBlock(hidden_dim)
        self.stage2_dim = hidden_dim * 2
        
        # Stage 2
        for i in range(n_stage2):
            dpr = original_drop_path * (current_layer_idx / total_blocks)
            if use_bidirectional:
                self.stage2_blocks.append(BidirectionalConvMambaBlock(self.stage2_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=dpr, kernel_size=inter_window_kernel_size, padding=inter_window_padding))
            else:
                self.stage2_blocks.append(ConvMambaBlock(self.stage2_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=dpr, kernel_size=inter_window_kernel_size, padding=inter_window_padding))
            current_layer_idx += 1

        # Learned Temporal Pooling Projection (for Stage 2dim)
        self.pool_proj = nn.Linear(self.stage2_dim, 1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.stage2_dim),
            nn.Dropout(0.15),
            nn.Linear(self.stage2_dim, self.stage2_dim // 2),
            nn.GELU(),
            nn.Linear(self.stage2_dim // 2, num_classes)
        )

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
            
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded.type_as(x)
            
            sum_pooled = x_masked.sum(dim=1)
            lengths = segment_lengths.unsqueeze(1).float()
            lengths = torch.clamp(lengths, min=1.0)
            mean_pooled = sum_pooled / lengths
            
            x_for_max = x.masked_fill(~mask_expanded, float('-inf'))
            max_pooled = x_for_max.max(dim=1)[0]
            
            window_vectors = (mean_pooled + max_pooled) / 2
            sample_vectors_list.append(window_vectors)
        
        # --- Inter-Window Pass (Hierarchical) ---
        
        if not sample_vectors_list:
             # Handle empty batch case properly with correct dim
             return torch.zeros(len(segments_complex), self.head[-1].out_features, device=self.intra_window_blocks[0].local_conv.weight.device)
             
        device = sample_vectors_list[0].device
        
        padded_seqs = pad_sequence(sample_vectors_list, batch_first=True, padding_value=0.0)
        seq_lengths = torch.tensor([s.shape[0] for s in sample_vectors_list], device=device)
        max_segments = seq_lengths.max().item()
        seq_mask = torch.arange(max_segments, device=device)[None, :] < seq_lengths[:, None]
        
        seq = padded_seqs.contiguous()
        
        # Stage 1
        for block in self.stage1_blocks:
            if self.use_checkpointing and seq.requires_grad:
                seq = checkpoint(block, seq, use_reentrant=False)
            else:
                seq = block(seq)
                
        # Downsample
        # Seq: [B, T, D] -> [B, T//2, 2D]
        # Mask needs update too
        B, T, D = seq.shape
        
        seq = self.downsample(seq)
        
        # Update Mask: [B, T] -> [B, T//2]
        # Considers a 2-step block valid if at least one step was valid? Or both?
        # Standard: ceil(Length / 2)
        new_seq_lengths = torch.ceil(seq_lengths.float() / 2).long()
        max_new_T = seq.shape[1]
        seq_mask = torch.arange(max_new_T, device=device)[None, :] < new_seq_lengths[:, None]
        
        # Stage 2
        for block in self.stage2_blocks:
            if self.use_checkpointing and seq.requires_grad:
                seq = checkpoint(block, seq, use_reentrant=False)
            else:
                seq = block(seq)
        
        # Learned Temporal Pooling
        # seq: [B, T/2, 2D]
        attn_logits = self.pool_proj(seq) # [B, T/2, 1]
        
        seq_mask_expanded = seq_mask.unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(~seq_mask_expanded, float('-inf'))
        
        weights = F.softmax(attn_logits, dim=1)
        global_feat = (seq * weights).sum(dim=1)
        
        logits = self.head(global_feat)
        
        return logits