# Controlling Second-Stage ratio_of_vectors

## Summary

You now have **two ways** to control the second-stage `ratio_of_vectors` (training-time downsampling):

1. **❌ Old Way**: Hardcode in training script
2. **✅ New Way**: Configure in `config.yaml` 

---

## Configuration in config.yaml

The second-stage downsampling is now configured in the `DATA` section:

```yaml
DATA:
  dataset:
    file_name: DVSGesturePrecomputed
    class_name: DVSGesturePrecomputed
    dataset_init_args:
      precomputed_dir: /fs/nexus-scratch/.../precomputed_data/dvsgesture
      height: 128
      width: 128
      num_classes: 11
      # ⭐ Second-stage downsampling (training-time augmentation)
      train_ratio_of_vectors: 0.8  # Train: use 80% of precomputed vectors
      val_ratio_of_vectors: 1.0    # Val: use 100% (all) vectors
      use_flip_augmentation: False
```

---

## Two-Stage Downsampling Overview

### Stage 1: Preprocessing (One-time)
**Location:** `PRECOMPUTING.ratio_of_vectors`
```yaml
PRECOMPUTING:
  ratio_of_vectors: 0.1  # 10% of events → vectors
```
This runs once during `./run_preprocess.sh`

### Stage 2: Training (Per epoch)
**Location:** `DATA.dataset.dataset_init_args.train_ratio_of_vectors`
```yaml
DATA:
  dataset:
    dataset_init_args:
      train_ratio_of_vectors: 0.8  # Use 80% of precomputed vectors
      val_ratio_of_vectors: 1.0    # Use all vectors for validation
```
This applies every epoch for data augmentation.

---

## Usage in Training Script

### Method 1: Helper Functions (Recommended)

```python
import yaml
from data import create_dvsgesture_dataloaders

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create dataloaders (automatically uses config ratios)
dataloaders = create_dvsgesture_dataloaders(config)

train_loader = dataloaders['train']      # Uses train_ratio_of_vectors=0.8
val_loader = dataloaders['validation']    # Uses val_ratio_of_vectors=1.0

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        vectors = batch['vectors']
        event_coords = batch['event_coords']
        labels = batch['labels']
        # ... training code ...
```

### Method 2: Manual Creation

```python
import yaml
from data import DVSGesturePrecomputed, collate_fn
from torch.utils.data import DataLoader

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

cfg = config['DATA']['dataset']['dataset_init_args']

# Training dataset
train_dataset = DVSGesturePrecomputed(
    precomputed_dir=cfg['precomputed_dir'],
    purpose='train',
    ratio_of_vectors=cfg['train_ratio_of_vectors'],  # 0.8 from config
    height=cfg['height'],
    width=cfg['width'],
)

# Validation dataset
val_dataset = DVSGesturePrecomputed(
    precomputed_dir=cfg['precomputed_dir'],
    purpose='validation',
    ratio_of_vectors=cfg['val_ratio_of_vectors'],  # 1.0 from config
    height=cfg['height'],
    width=cfg['width'],
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
```

---

## Recommended Values

### train_ratio_of_vectors
- **1.0**: No augmentation, use all precomputed vectors
- **0.8-0.9**: Mild augmentation (recommended for most cases)
- **0.5-0.7**: Moderate augmentation
- **0.3-0.4**: Strong augmentation

### val_ratio_of_vectors
- **1.0**: Always use all vectors for validation (recommended)
- Lower values only if you want to speed up validation

---

## Example Configurations

### Conservative (Less Augmentation)
```yaml
train_ratio_of_vectors: 0.9  # Use 90%
val_ratio_of_vectors: 1.0    # Use 100%
```

### Balanced (Recommended)
```yaml
train_ratio_of_vectors: 0.8  # Use 80%
val_ratio_of_vectors: 1.0    # Use 100%
```

### Aggressive (More Augmentation)
```yaml
train_ratio_of_vectors: 0.5  # Use 50%
val_ratio_of_vectors: 1.0    # Use 100%
```

---

## Testing Helper Functions

Test that the helper functions work correctly:

```bash
cd /fs/nexus-scratch/haowenyu/GestureRecognitionNew
mamba run -n torch python data/create_datasets.py --config configs/config.yaml
```

Expected output:
```
Creating datasets from config...

Datasets created:
  train: 1077 samples
    ratio_of_vectors: 0.8
  validation: 333 samples
    ratio_of_vectors: 1.0

Creating dataloaders from config...

Dataloaders created:
  train: 34 batches
    batch_size: 32
  validation: 6 batches
    batch_size: 64

Testing batch loading...
train batch:
  vectors: 32 samples
  event_coords: 32 samples
  labels shape: torch.Size([32])
  ...
```

---

## Summary

| Parameter | Location | Purpose | Value |
|-----------|----------|---------|-------|
| **First-stage** | `PRECOMPUTING.ratio_of_vectors` | Preprocessing: events→vectors | 0.1 (10%) |
| **Second-stage (train)** | `DATA.dataset.dataset_init_args.train_ratio_of_vectors` | Training augmentation | 0.8 (80%) |
| **Second-stage (val)** | `DATA.dataset.dataset_init_args.val_ratio_of_vectors` | Validation | 1.0 (100%) |

**Key Point**: You control the second-stage ratio by editing `train_ratio_of_vectors` and `val_ratio_of_vectors` in your `config.yaml` file! 🎯
