import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VecKMSparse(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        encoding_dim: int,
        temporal_length: float,
        kernel_size: int = 17,
        T_scale: float = 3.0,
        S_scale: float = 3.0, 
    ):
        super().__init__()
        # Convert to proper types (handles OmegaConf interpolation)
        self.height = int(height)
        self.width = int(width)
        self.encoding_dim = int(encoding_dim)
        self.temporal_length = float(temporal_length)
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        
        # Random Fourier Features
        T_scale = float(T_scale)
        S_scale = float(S_scale)
        self.register_buffer('T', torch.randn(1, self.encoding_dim) * T_scale)
        self.register_buffer('X', torch.randn(1, self.encoding_dim) * S_scale)
        self.register_buffer('Y', torch.randn(1, self.encoding_dim) * S_scale)
        
        # Precompute Spatial Kernel Weights
        self.register_buffer('kernel_weights', self._precompute_kernel())

    def _precompute_kernel(self):
        r = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        dy, dx = torch.meshgrid(r, r, indexing='ij')
        
        norm_dy = dy / self.radius
        norm_dx = dx / self.radius
        
        phase_y = norm_dy.unsqueeze(-1) @ self.Y # type: ignore
        phase_x = norm_dx.unsqueeze(-1) @ self.X # type: ignore
        
        return torch.exp(1j * (phase_x + phase_y))

    @torch.no_grad
    def forward(self, t, y, x, query_y, query_x, query_t):
        """
        Args:
            t, y, x: Raw event stream (N,) - used to build the map
            query_y, query_x: Spatial locations of queries (M,)
            query_t: Timestamp of the queries (M,) - used for re-centering
        """
        device = t.device
        
        # ------------------------------------------------------------------
        # PHASE 1: Dense Accumulation (Absolute Time Encoding)
        # ------------------------------------------------------------------
        
        # 1. Normalize Time 
        # Use relative time for numerical stability (bit-exact invariance)
        t_ref = t.min().detach()
        t_centered = t - t_ref
        
        t_norm = t_centered / self.temporal_length 
        
        # 2. Compute Temporal Embeddings: exp(i * t * T)
        # Shape: (N, D)
        temp_emb = torch.exp(1j * (t_norm.unsqueeze(1) @ self.T))
        
        # 3. Scatter-Add to Grid
        grid_flat = torch.zeros(self.height * self.width, self.encoding_dim, 
                                dtype=torch.cfloat, device=device)
        count_flat = torch.zeros(self.height * self.width, 1, 
                                 dtype=torch.float, device=device)
        
        flat_indices = y.long() * self.width + x.long()
        
        grid_flat.index_add_(0, flat_indices, temp_emb)
        
        ones = torch.ones(t.shape[0], 1, device=device)
        count_flat.index_add_(0, flat_indices, ones)
        
        grid = grid_flat.view(self.height, self.width, self.encoding_dim)
        counts = count_flat.view(self.height, self.width, 1)

        # ------------------------------------------------------------------
        # PHASE 2: Sparse Convolution (Gather Neighbors)
        # ------------------------------------------------------------------
        
        # Pad grid
        padded_grid = F.pad(grid.permute(2, 0, 1), 
                            (self.radius, self.radius, self.radius, self.radius)).permute(1, 2, 0)
        padded_counts = F.pad(counts.permute(2, 0, 1), 
                              (self.radius, self.radius, self.radius, self.radius)).permute(1, 2, 0)
        
        # Adjust query coords for padding
        q_y = query_y.long() + self.radius
        q_x = query_x.long() + self.radius
        
        num_queries = query_y.shape[0]
        out_emb = torch.zeros(num_queries, self.encoding_dim, dtype=torch.cfloat, device=device)
        out_cnt = torch.zeros(num_queries, 1, dtype=torch.float, device=device)
        
        K = self.kernel_size
        
        # Iterate over Kernel offsets
        for dy_idx in range(K):
            offset_y = dy_idx - self.radius
            for dx_idx in range(K):
                offset_x = dx_idx - self.radius
                
                w_spatial = self.kernel_weights[dy_idx, dx_idx] 
                
                gathered_feat = padded_grid[q_y + offset_y, q_x + offset_x]
                gathered_cnt = padded_counts[q_y + offset_y, q_x + offset_x]
                
                out_emb += gathered_feat * w_spatial
                out_cnt += gathered_cnt

        # ------------------------------------------------------------------
        # PHASE 3: Temporal Re-centering & Normalization
        # ------------------------------------------------------------------
        # Algorithm Line 20: emb_k = emb_k * exp(-i * (t_k/dt) * T) / cnt_k
        
        # 1. Compute re-centering factor for the queries
        # We must use the same reference time
        qt_centered = query_t - t_ref
        qt_norm = qt_centered / self.temporal_length
        # Note the negative sign (-1j) for subtraction
        recenter_factor = torch.exp(-1j * (qt_norm.unsqueeze(1) @ self.T))
        
        # 2. Apply re-centering
        out_emb = out_emb * recenter_factor
        
        # 3. Normalize by count
        return out_emb * math.sqrt(self.encoding_dim) / out_cnt.clamp(min=1)

# --- Verification Script ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fix torch random seed
    torch.manual_seed(42)
    
    # Init
    model = VecKMSparse(height=240, width=320, encoding_dim=64, temporal_length=100.0).to(device)
    
    # Fake Data (100k events)
    N = 300000
    t = torch.sort(torch.rand(N))[0] * 100 + 100.0
    y = torch.randint(0, 240, (N,))
    x = torch.randint(0, 320, (N,))
    
    # Sample Queries from Events (Subset of events)
    num_queries = 10000
    indices = torch.randperm(N)[:num_queries]
    
    qy = y[indices]
    qx = x[indices]
    qt = t[indices]
    
    import time
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    # Pass qt to the forward pass
    res = model(t.to(device), y.to(device), x.to(device), 
                qy.to(device), qx.to(device), qt.to(device))
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    
    print(f"Time: {end - start:.4f}s")
    print(f"Output shape: {res.shape}") # Should be (5000, 64)
    print(f"Output: {res}")