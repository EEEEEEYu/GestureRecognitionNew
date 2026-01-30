import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.layers import DropPath
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

class BidirectionalConvMambaBlock(nn.Module):
    """
    Optimized Building Block: 
    1. x = x + DropPath(Mixer(Norm(x)))
    2. x = x + DropPath(MLP(Norm(x)))
    Designed for variable-length sequences [B, L, D].
    """
    def __init__(self, dim, d_state=32, expand=2, dropout=0.1, drop_path=0.1, kernel_size=3, padding=1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        
        # 1. Depthwise Conv to enhance local features (spatial/temporal jitters)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.ln_conv = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        
        # 2. SSM
        self.ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        # 3. Post-SSM Conv
        # After bidirectional concatenation, input channels are 2*dim, output should be dim to match residual
        self.post_conv = nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1, groups=dim)
        self.ln_post = nn.LayerNorm(dim * 2)
        self.act_post = nn.SiLU()
        
        # 4. Feed-Forward
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, L, D] (Channel Last)
        
        # --- Branch 1: Mixer (Conv + SSM + PostConv) ---
        residual = x
        x_norm = self.norm1(x)
        
        # 1. Local Conv: [B, L, D] -> [B, D, L]
        x_t = x_norm.transpose(1, 2)
        x_c = self.local_conv(x_t)
        x_c = x_c.transpose(1, 2)
        x_c = self.act1(self.ln_conv(x_c)) 
        
        x_c_in = x_c.contiguous()
        
        # 2. Global SSM
        x_s = self.ssm(x_c_in + x_norm)
        x_c_in_reversed = x_c_in.flip(1)
        x_norm_reversed = x_norm.flip(1)
        x_s_reversed = self.ssm(x_c_in_reversed + x_norm_reversed)
        x_s_reversed = x_s_reversed.flip(1)
        x_s = torch.cat([x_s, x_s_reversed], dim=-1)
        
        # 3. Post-SSM Conv
        x_s_norm = self.ln_post(x_s)
        x_s_t = x_s_norm.transpose(1, 2)
        x_out = self.post_conv(x_s_t)
        x_out = x_out.transpose(1, 2)
        x_out = self.act_post(x_out)
        
        # Residual connection for Mixer
        x = residual + self.drop_path(x_out)
        
        # --- Branch 2: Feed-Forward ---
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvMambaBlock(nn.Module):
    """
    Optimized Building Block: 
    1. x = x + DropPath(Mixer(Norm(x)))
    2. x = x + DropPath(MLP(Norm(x)))
    Designed for variable-length sequences [B, L, D].
    """
    def __init__(self, dim, d_state=32, expand=2, dropout=0.1, drop_path=0.1, kernel_size=3, padding=1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        
        # 1. Depthwise Conv to enhance local features (spatial/temporal jitters)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.ln_conv = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        
        # 2. SSM
        self.ssm = Mamba(d_model=dim, d_state=d_state, expand=expand)
        
        # 3. Post-SSM Conv
        self.post_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.ln_post = nn.LayerNorm(dim)
        self.act_post = nn.SiLU()
        
        # 4. Feed-Forward
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, L, D] (Channel Last)
        
        # --- Branch 1: Mixer (Conv + SSM + PostConv) ---
        residual = x
        x_norm = self.norm1(x)
        
        # 1. Local Conv: [B, L, D] -> [B, D, L]
        x_t = x_norm.transpose(1, 2)
        x_c = self.local_conv(x_t)
        x_c = x_c.transpose(1, 2)
        x_c = self.act1(self.ln_conv(x_c)) 
        
        x_c_in = x_c.contiguous()
        
        # 2. Global SSM
        # Research-backed: Should we add 'x' (residual) here? 
        # Current plan: Remove 'x' input to SSM to let Conv features dominate.
        x_s = self.ssm(x_c_in + x_norm)
        
        # 3. Post-SSM Conv
        x_s_t = x_s.transpose(1, 2)
        x_out = self.post_conv(x_s_t)
        x_out = x_out.transpose(1, 2)
        x_out = self.act_post(self.ln_post(x_out))
        
        # Residual connection for Mixer
        x = residual + self.drop_path(x_out)
        
        # --- Branch 2: Feed-Forward ---
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
        drop_path=0.1,
        use_checkpointing=False,
        use_bidirectional=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing
        
        # Initial projection: Complex (Real/Imag) + Normalized Coords
        self.input_adapter = nn.Sequential(
            nn.Linear(encoding_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # --- Level 1: Intra-Window Shared Blocks ---
        # These process N events within a single segment.
        if use_bidirectional:
            self.intra_window_blocks = nn.ModuleList([
                BidirectionalConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=drop_path, kernel_size=intra_window_kernel_size, padding=intra_window_padding) for _ in range(intra_window_blocks)
            ])
        else:
            self.intra_window_blocks = nn.ModuleList([
                ConvMambaBlock(hidden_dim, d_state=intra_window_d_state, expand=intra_window_expand, dropout=dropout, drop_path=drop_path, kernel_size=intra_window_kernel_size, padding=intra_window_padding) for _ in range(intra_window_blocks)
            ])
        
        # --- Level 2: Inter-Window (Temporal) Blocks ---
        # These process T segments across the whole sequence.
        if use_bidirectional:
            self.inter_window_blocks = nn.ModuleList([
                BidirectionalConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=drop_path, kernel_size=inter_window_kernel_size, padding=inter_window_padding) for _ in range(inter_window_blocks)
            ])
        else:
            self.inter_window_blocks = nn.ModuleList([
                ConvMambaBlock(hidden_dim, d_state=inter_window_d_state, expand=inter_window_expand, dropout=dropout, drop_path=drop_path, kernel_size=inter_window_kernel_size, padding=inter_window_padding) for _ in range(inter_window_blocks)
            ])
        
        # Learned Temporal Pooling Projection
        self.pool_proj = nn.Linear(hidden_dim, 1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, batch):
        # batch['segments_complex']: List[List[Tensor]] [B, T, N, D]
        # batch['segments_coords']: List[List[Tensor]] [B, T, N, 2]
        
        segments_complex = batch['segments_complex']
        segments_coords = batch['segments_coords']
        
        # To avoid OOM, we process each sample in the batch independently for the intra-window stage
        # This keeps the peak memory usage lower (proportional to single sample size, not batch size)
        
        sample_vectors_list = []
        
        for z_list, c_list in zip(segments_complex, segments_coords):
            # Processing one sample (which has T segments)
            
            if len(z_list) == 0:
                # Handle empty sample if necessary (though usually filtered out)
                 continue
                 
            device = z_list[0].device
            
            # 1. Prepare segments for this sample
            # [T_segments, N_events, D]
            
            # Concatenate features per segment
            processed_segments = []
            for z, c in zip(z_list, c_list):
                 processed_segments.append(torch.cat([z.real, z.imag, c], dim=-1))
            
            # Find max events for this sample
            segment_lengths = torch.tensor([s.shape[0] for s in processed_segments], device=device)
            max_events = segment_lengths.max().item()
            
            # Pad segments within this sample
            padded_inputs = pad_sequence(processed_segments, batch_first=True, padding_value=0.0)
            
            # Create mask for this sample
            mask = torch.arange(max_events, device=device)[None, :] < segment_lengths[:, None]
            
            # 2. Intra-Window Pass (Batched over T segments)
            x = self.input_adapter(padded_inputs)
            x = x.contiguous()
            
            for block in self.intra_window_blocks:
                if self.use_checkpointing and x.requires_grad:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            
            # 3. Masked Pooling
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded.type_as(x)
            
            # Sum pooling
            sum_pooled = x_masked.sum(dim=1)
            
            # Mean pooling
            lengths = segment_lengths.unsqueeze(1).float()
            lengths = torch.clamp(lengths, min=1.0)
            mean_pooled = sum_pooled / lengths
            
            # Max pooling
            x_for_max = x.masked_fill(~mask_expanded, float('-inf'))
            max_pooled = x_for_max.max(dim=1)[0]
            
            # Combined Vector: [T_segments, hidden_dim]
            window_vectors = (mean_pooled + max_pooled) / 2
            
            sample_vectors_list.append(window_vectors)
        
        # --- 4. Inter-Window Pass (Batched over BatchSize) ---
        # Now we have a list of [T_i, hidden_dim] tensors, one per sample
        
        if not sample_vectors_list:
             return torch.zeros(len(segments_complex), self.head[-1].out_features, device=self.intra_window_blocks[0].local_conv.weight.device)
             
        device = sample_vectors_list[0].device
        
        # Pad temporal dimension
        padded_seqs = pad_sequence(sample_vectors_list, batch_first=True, padding_value=0.0)
        
        # Create temporal mask
        # [BatchSize]
        seq_lengths = torch.tensor([s.shape[0] for s in sample_vectors_list], device=device)
        max_segments = seq_lengths.max().item()
        
        # [BatchSize, MaxSegments]
        seq_mask = torch.arange(max_segments, device=device)[None, :] < seq_lengths[:, None]
        
        seq = padded_seqs.contiguous()
        
        for block in self.inter_window_blocks:
            if self.use_checkpointing and seq.requires_grad:
                seq = checkpoint(block, seq, use_reentrant=False)
            else:
                seq = block(seq)
                
        # 5. Learned Temporal Pooling with Attention
        # seq: [B, T, D]
        # weights: [B, T, 1]
        
        attn_logits = self.pool_proj(seq) # [B, T, 1]
        
        # Apply mask to attention logits
        seq_mask_expanded = seq_mask.unsqueeze(-1) # [B, T, 1]
        attn_logits = attn_logits.masked_fill(~seq_mask_expanded, float('-inf'))
        
        weights = F.softmax(attn_logits, dim=1)
        
        # Weighted Pooling
        # [B, T, D] * [B, T, 1] -> [B, D]
        global_feat = (seq * weights).sum(dim=1)
        
        # 6. Classification
        logits = self.head(global_feat)
        
        return logits