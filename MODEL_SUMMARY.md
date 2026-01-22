# NestedEventMamba Model Summary

## Overview
The `NestedEventMamba` model is a hierarchical SSM-based architecture for event-based gesture recognition. It processes variable-length event sequences through a two-level hierarchy:
1. **Intra-Window Processing**: Local feature extraction within temporal segments
2. **Inter-Window Processing**: Global temporal modeling across segments

---

## Architecture Breakdown

### Total Parameters: **~1.19M** (1,187,595)
- **Model Size**: ~4.53 MB (float32) / ~2.27 MB (float16)
- **All parameters are trainable**

---

## Component-wise Parameter Distribution

### 1. Input Adapter: **33,280 parameters** (2.8%)
Transforms complex embeddings and coordinates into hidden representation.

| Layer | Parameters |
|-------|------------|
| Linear (130 → 128) | 16,768 |
| GELU | 0 |
| Linear (128 → 128) | 16,512 |
| **Subtotal** | **33,280** |

**Input**: Complex embeddings [N, 64] + Coordinates [N, 2]  
**Processing**: Concatenates [real, imag, normalized_coords] → [N, 130]  
**Output**: [N, 128] hidden features

---

### 2. Intra-Window Blocks (×2): **630,400 parameters** (53.1%)
Process events within individual temporal segments.

#### Each Block Contains:
| Component | Parameters | Purpose |
|-----------|------------|---------|
| LayerNorm (norm1) | 256 | Normalize inputs |
| Conv1d (local_conv) | 512 | Depthwise convolution for local patterns |
| **Mamba2 (ssm)** | **182,464** | **State-space modeling (main computation)** |
| LayerNorm (norm2) | 256 | Normalize before MLP |
| MLP (4× expansion) | 131,712 | Feed-forward network |
| DropPath | 0 | Stochastic depth regularization |
| **Per-block Total** | **315,200** | |

**Configuration**:
- `d_state = 32` (SSM state dimension)
- `expand = 2` (SSM expansion factor)
- Each block processes: [B, N_events, 128] → [B, N_events, 128]

---

### 3. Inter-Window Blocks (×2): **514,688 parameters** (43.3%)
Process temporal relationships across segments.

#### Each Block Contains:
| Component | Parameters | Purpose |
|-----------|------------|---------|
| LayerNorm (norm1) | 256 | Normalize inputs |
| Conv1d (local_conv) | 512 | Depthwise convolution for local patterns |
| **Mamba2 (ssm)** | **124,608** | **State-space modeling (main computation)** |
| LayerNorm (norm2) | 256 | Normalize before MLP |
| MLP (4× expansion) | 131,712 | Feed-forward network |
| DropPath | 0 | Stochastic depth regularization |
| **Per-block Total** | **257,344** | |

**Configuration**:
- `d_state = 32` (SSM state dimension)
- `expand = 1.5` (SSM expansion factor - smaller than intra-window)
- Each block processes: [B, T_segments, 128] → [B, T_segments, 128]

---

### 4. Classification Head: **9,227 parameters** (0.8%)
Final classification layer.

| Layer | Parameters |
|-------|------------|
| LayerNorm | 256 |
| Dropout (0.3) | 0 |
| Linear (128 → 64) | 8,256 |
| GELU | 0 |
| Linear (64 → 11) | 715 |
| **Subtotal** | **9,227** |

**Output**: 11 gesture classes

---

## Key Observations

### Parameter Distribution
```
Input Adapter:         33,280 ( 2.8%) ▓
Intra-Window Blocks:  630,400 (53.1%) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
Inter-Window Blocks:  514,688 (43.3%) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
Classification Head:    9,227 ( 0.8%) ▓
```

### Mamba2 SSM Dominance
- **Intra-Window Mamba2**: 182,464 params × 2 = 364,928 (30.7%)
- **Inter-Window Mamba2**: 124,608 params × 2 = 249,216 (21.0%)
- **Total Mamba2 params**: 614,144 (51.7% of entire model!)

The SSM blocks are the computational core of the model, handling the temporal dependencies.

### MLP Contribution
- **Intra-Window MLPs**: 131,712 × 2 = 263,424 (22.2%)
- **Inter-Window MLPs**: 131,712 × 2 = 263,424 (22.2%)
- **Total MLP params**: 526,848 (44.4%)

MLPs provide the non-linear transformation capacity.

---

## Design Rationale

### Why Different Expansion Factors?
- **Intra-Window** (`expand=2`): Larger capacity for extracting rich local features from raw events
- **Inter-Window** (`expand=1.5`): Smaller capacity since working with already-processed segment representations

### Model Efficiency
- **Compact size**: ~1.2M parameters is very lightweight by modern standards
- **Efficient inference**: Can process variable-length sequences without padding overhead
- **Memory-efficient**: SSM architecture is more efficient than attention mechanisms for long sequences

### Trade-offs
- **Local vs Global**: 53% intra-window vs 43% inter-window shows balanced capacity
- **Feature extraction vs Classification**: 99% feature learning vs 1% classification
- **Computation vs Storage**: SSM provides good compute-memory trade-off

---

## Comparison to Other Architectures

| Model Type | Typical Parameters | NestedEventMamba |
|------------|-------------------|------------------|
| ResNet-18 | ~11M | 1.2M (10× smaller) |
| ViT-Tiny | ~6M | 1.2M (5× smaller) |
| MobileNetV2 | ~3.5M | 1.2M (3× smaller) |
| **NestedEventMamba** | **1.2M** | **Baseline** |

The model is designed to be efficient while maintaining strong representation capacity through SSM blocks.

---

## Usage Example

```python
from model.new_ssm import NestedEventMamba, print_model_summary

# Create model
model = NestedEventMamba(
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

# Print detailed summary
total_params, trainable_params = print_model_summary(model)
```

---

## Model Architecture Flow

```
Input: Event Stream Segments
        ↓
┌─────────────────────────┐
│   Input Adapter         │  [N, 130] → [N, 128]
│   (33K params)          │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ For each segment:       │
│                         │
│  ┌──────────────────┐   │
│  │ Intra-Block 1    │   │  [N, 128] → [N, 128]
│  │ (315K params)    │   │
│  └──────────────────┘   │
│          ↓              │
│  ┌──────────────────┐   │
│  │ Intra-Block 2    │   │  [N, 128] → [N, 128]
│  │ (315K params)    │   │
│  └──────────────────┘   │
│          ↓              │
│  [ Pooling: N → 1 ]     │
└─────────────────────────┘
        ↓
Segment Representations
        ↓
┌─────────────────────────┐
│  ┌──────────────────┐   │
│  │ Inter-Block 1    │   │  [T, 128] → [T, 128]
│  │ (257K params)    │   │
│  └──────────────────┘   │
│          ↓              │
│  ┌──────────────────┐   │
│  │ Inter-Block 2    │   │  [T, 128] → [T, 128]
│  │ (257K params)    │   │
│  └──────────────────┘   │
└─────────────────────────┘
        ↓
  [ Pooling: T → 1 ]
        ↓
┌─────────────────────────┐
│  Classification Head    │  [128] → [11]
│   (9K params)           │
└─────────────────────────┘
        ↓
   Class Logits
```

---

*Generated from NestedEventMamba architecture analysis*
