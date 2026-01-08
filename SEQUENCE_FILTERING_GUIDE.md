# Sequence Filtering Guide for Custom Gesture Dataset

## Overview

The preprocessing pipeline now supports **flexible sequence filtering** based on recording conditions. This allows you to select any combination of:
- **Camera View**: TOP, SIDE, or both
- **Lighting**: LIGHT, DARK, or both
- **Background**: STATIC, DYNAMIC, or both

## Why Use Filtering?

You mentioned suspecting that **dynamic background sequences could be too hard for the model**. With this filtering system, you can:

1. **Test with static backgrounds only** → 50% smaller dataset, potentially easier for the model
2. **Test specific combinations** → e.g., only TOP view with STATIC background
3. **Isolate difficult conditions** → e.g., train on LIGHT only, test on DARK
4. **Reduce preprocessing time** → Filter before preprocessing to save time

## How to Use

### 1. Edit the Configuration File

Open `configs/custom_gesture_config.yaml` and find the `PRECOMPUTING.filter` section:

```yaml
PRECOMPUTING:
  # ... other settings ...
  
  filter:
    view: both         # Options: "TOP", "SIDE", "both", or ["TOP", "SIDE"]
    lighting: both     # Options: "LIGHT", "DARK", "both", or ["LIGHT", "DARK"]
    background: both   # Options: "STATIC", "DYNAMIC", "both", or ["STATIC", "DYNAMIC"]
```

### 2. Common Filtering Examples

#### Example 1: Static Background Only (Recommended Test)
```yaml
filter:
  view: both
  lighting: both
  background: STATIC  # Reduces dataset by 50%
```

#### Example 2: TOP View with LIGHT Conditions
```yaml
filter:
  view: TOP
  lighting: LIGHT
  background: both    # Reduces dataset by 75%
```

#### Example 3: Multiple Values (List)
```yaml
filter:
  view: ["TOP", "SIDE"]  # Same as "both"
  lighting: LIGHT
  background: ["STATIC"]  # Same as "STATIC"
```

#### Example 4: No Filtering (Use All Data)
```yaml
filter:
  view: both
  lighting: both
  background: both    # Uses all 1024 sequences
```

### 3. Test Your Filters BEFORE Preprocessing

**IMPORTANT**: Always test your filters first using the verification script:

```bash
mamba activate torch
python test_sequence_filters.py
```

This will show you:
- How many sequences will be selected
- Distribution by view/lighting/background
- Distribution by person and class
- Sample sequences that match your filter

### 4. Run Preprocessing

Once you're satisfied with the filter settings:

```bash
mamba activate torch
python preprocess_custom_gesture_ultra.py --config configs/custom_gesture_config.yaml
```

## Dataset Statistics

**Full Dataset** (no filtering):
- Total sequences: **1,024**
- View distribution: 50% TOP, 50% SIDE
- Background distribution: 50% STATIC, 50% DYNAMIC
- Lighting distribution: 50% LIGHT, 50% DARK
- Persons: 8 people (128 sequences each)
- Classes: 16 classes (64 sequences each)

**With `background: STATIC`**:
- Total sequences: **512** (50% reduction)
- All persons still represented (64 sequences each)
- All classes still represented (32 sequences each)

## Sequence Naming Convention

Sequences follow this naming pattern:
```
sequence_{person}_{view}_{background}_{lighting}_{class_name}
```

Example:
```
sequence_haowen1_SIDE_STATIC_DARK_knife_bread
         ^^^^^^^ ^^^^ ^^^^^^ ^^^^ ^^^^^^^^^^^^
         person  view   bg   light  class
```

## What Gets Stored in HDF5

The preprocessing now stores the following metadata for each sequence:
- `sequence_names`: Full sequence name
- `person_ids`: Person identifier
- `views`: Camera view (TOP/SIDE)
- `backgrounds`: Background type (STATIC/DYNAMIC)
- `lightings`: Lighting condition (LIGHT/DARK)
- `labels`: Class label (1-16)
- `class_names`: Class name string
- `file_paths`: Original file path

This allows you to analyze results by condition after training!

## Recommended Testing Strategy

If you suspect dynamic backgrounds are too difficult:

1. **Start with static only**:
   ```yaml
   filter:
     background: STATIC
   ```
   
2. **Preprocess and train** on this subset

3. **Compare performance** to your current model

4. **If performance improves significantly**, consider:
   - Training longer on static-only data
   - Using dynamic backgrounds only for validation
   - Implementing progressive training (static → dynamic)

## Notes

- **Checkpointing still works**: If preprocessing is interrupted, it will resume from where it left off
- **Filters apply to both train and validation splits**: The validation split strategy respects your filters
- **No need to re-preprocess everything**: Just delete the old HDF5 files and checkpoint.json, then run preprocessing again with new filters
- **Sequence names are case-sensitive**: Use "STATIC" not "static", "TOP" not "top"

## Quick Reference

| Filter Setting | Result | Dataset Size |
|---------------|--------|--------------|
| `background: both` | All sequences | 1024 (100%) |
| `background: STATIC` | Static only | 512 (50%) |
| `background: DYNAMIC` | Dynamic only | 512 (50%) |
| `view: TOP` | Top camera only | 512 (50%) |
| `view: SIDE` | Side camera only | 512 (50%) |
| `lighting: LIGHT` | Light conditions | 512 (50%) |
| `lighting: DARK` | Dark conditions | 512 (50%) |
| `background: STATIC`<br>`view: TOP` | Static + Top | 256 (25%) |

## Files Modified

1. **`configs/custom_gesture_config.yaml`**: Added `PRECOMPUTING.filter` section
2. **`preprocess_custom_gesture_ultra.py`**: 
   - Added filter logic
   - Stores sequence metadata in HDF5
   - Shows filter settings in statistics
3. **`test_sequence_filters.py`**: New verification script

---

**Questions?** The implementation is complete and ready to use. Just edit the config and run the test script!
