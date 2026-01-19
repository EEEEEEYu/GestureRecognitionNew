# UCF101-DVS Dataset Integration

## Summary

Successfully implemented UCF101-DVS dataset integration. Despite initial concerns about graph features, the dataset actually contains **raw events** stored in `.mat` (MATLAB) format, making it fully compatible with the DVSGesture/HMDB-DVS preprocessing pipeline.

## Dataset Details

- **101 action classes** (ApplyEyeMakeup, Basketball, Swimming, etc.)
- **13,523 total samples** (10,781 train / 2,742 validation with 80/20 split)
- **File format**: `.mat` files with keys: `x`, `y`, `ts` (timestamps), `pol` (polarity)
- **Resolution**: 240×180 pixels
- **Same format as HMDB-DVS**, just different container format

## Implemented Components

### 1. Exploration Script
**File**: `data/UCF101_DVS/explore_ucf101_dvs.py`

Analyzes .mat file structure and confirms raw event format:
```bash
python data/UCF101_DVS/explore_ucf101_dvs.py --dataset_dir ~/Downloads/UCF101_DVS
```

### 2. Dataset Loader
**File**: `data/UCF101_DVS/dataset.py`

PyTorch Dataset class that:
- Loads events from `.mat` files using scipy
- Temporal slicing at configurable intervals
- Optional flip augmentation
- Returns same format as DVSGesture/HMDB-DVS

### 3. Preprocessing Script
**File**: `data/UCF101_DVS/preprocess.py`

Complete preprocessing pipeline adapted from HMDB-DVS:
- Denoising (optional spatial filtering)
- Sampling (random or grid decimation)
- VecKM encoding to HDF5
- Checkpoint support

### 4. Visualization Script
**File**: `data/UCF101_DVS/visualize_events.py`

Self-contained event visualization tool:
```bash  
python data/UCF101_DVS/visualize_events.py --sample_idx 0 --num_frames 10
```

## Key Findings

✅ **Not graph features** - Contains raw DVS events  
✅ **Compatible pipeline** - Can use same preprocessing as HMDB-DVS
✅ **240×180 resolution** - Same as HMDB-DVS  
✅ **Rich dataset** - 101 classes, 13K+ samples

## Usage

1. **Test dataset loader**:
```bash
python data/UCF101_DVS/dataset.py
```

2. **Create config file**: `configs/config_ucf101.yaml` (copy from `config_hmdb.yaml` and update paths)

3. **Run preprocessing**:
```bash
bash run_preprocess_ucf101.sh
```

4. **Visualize samples**:
```bash
python data/UCF101_DVS/visualize_events.py --sample_idx 0
```
