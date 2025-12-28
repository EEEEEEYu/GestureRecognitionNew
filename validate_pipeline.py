#!/usr/bin/env python3
"""
Quick validation script to test the complete data pipeline.

This script:
1. Loads configuration
2. Creates dataloaders
3. Tests batch loading
4. Validates data format
5. Shows statistics

Run: mamba run -n torch python validate_pipeline.py
"""

import yaml
import torch
import numpy as np
from data import create_dvsgesture_dataloaders


def validate_batch(batch, purpose):
    """Validate a single batch structure and contents."""
    print(f"\n{'='*60}")
    print(f"Validating {purpose} batch")
    print(f"{'='*60}")
    
    # Check required keys
    required_keys = ['vectors', 'event_coords', 'labels', 'num_vectors_per_sample',
                     'num_vectors_per_interval', 'file_paths', 'num_intervals']
    
    for key in required_keys:
        if key not in batch:
            print(f"❌ Missing key: {key}")
            return False
        else:
            print(f"✅ {key}: present")
    
    # Validate types and shapes
    vectors = batch['vectors']
    coords = batch['event_coords']
    labels = batch['labels']
    
    batch_size = len(vectors)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    
    # Check each sample
    for i in range(min(3, batch_size)):  # Check first 3 samples
        v = vectors[i]
        c = coords[i]
        
        print(f"\nSample {i}:")
        print(f"  Vectors shape: {v.shape}")
        print(f"  Vectors dtype: {v.dtype}")
        print(f"  Coords shape: {c.shape}")
        print(f"  Coords dtype: {c.dtype}")
        
        # Validate alignment
        if v.shape[0] != c.shape[0]:
            print(f"  ❌ Shape mismatch! vectors[0]={v.shape[0]}, coords[0]={c.shape[0]}")
            return False
        else:
            print(f"  ✅ Shapes aligned: {v.shape[0]} vectors")
        
        # Validate coords format
        if c.shape[1] != 4:
            print(f"  ❌ Coords should be [N, 4], got {c.shape}")
            return False
        else:
            print(f"  ✅ Coords format correct: [N, 4]")
        
        # Check coord ranges
        x_coords = c[:, 0]
        y_coords = c[:, 1]
        t_coords = c[:, 2]
        p_coords = c[:, 3]
        
        print(f"  Coord ranges:")
        print(f"    x: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
        print(f"    y: [{y_coords.min():.1f}, {y_coords.max():.1f}]")
        print(f"    t: [{t_coords.min():.2f}, {t_coords.max():.2f}] ms")
        print(f"    p: {np.unique(p_coords)}")
    
    print(f"\n✅ {purpose} batch validation PASSED")
    return True


def main():
    print("="*60)
    print("DVSGesture Data Pipeline Validation")
    print("="*60)
    
    # Load config
    print("\n1. Loading configuration...")
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Print key configuration
    print("\nKey Configuration:")
    print(f"  Precomputed dir: {config['DATA']['dataset']['dataset_init_args']['precomputed_dir']}")
    print(f"  Train ratio: {config['DATA']['dataset']['dataset_init_args']['train_ratio_of_vectors']}")
    print(f"  Val ratio: {config['DATA']['dataset']['dataset_init_args']['val_ratio_of_vectors']}")
    print(f"  Batch size (train): {config['DATA']['dataloader']['batch_size']}")
    print(f"  Batch size (val): {config['DATA']['dataloader']['test_batch_size']}")
    
    # Create dataloaders
    print("\n2. Creating dataloaders...")
    try:
        dataloaders = create_dvsgesture_dataloaders(config)
        print(f"✅ Dataloaders created")
        print(f"  Train: {len(dataloaders['train'])} batches")
        print(f"  Validation: {len(dataloaders['validation'])} batches")
    except FileNotFoundError as e:
        print(f"❌ Precomputed data not found: {e}")
        print("\n⚠️  You need to run preprocessing first:")
        print("   ./run_preprocess.sh")
        return 1
    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test train loader
    print("\n3. Testing train dataloader...")
    try:
        train_loader = dataloaders['train']
        for batch in train_loader:
            if not validate_batch(batch, 'train'):
                return 1
            break
    except Exception as e:
        print(f"❌ Train dataloader failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test validation loader
    print("\n4. Testing validation dataloader...")
    try:
        val_loader = dataloaders['validation']
        for batch in val_loader:
            if not validate_batch(batch, 'validation'):
                return 1
            break
    except Exception as e:
        print(f"❌ Validation dataloader failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL VALIDATIONS PASSED!")
    print("="*60)
    print("\nData Pipeline Status:")
    print("  ✅ Configuration valid")
    print("  ✅ Dataloaders working")
    print("  ✅ Batch structure correct")
    print("  ✅ Vector-coordinate alignment verified")
    print("  ✅ Data formats correct")
    
    print("\n🚀 Your pipeline is READY FOR TRAINING!")
    
    print("\nNext steps:")
    print("  1. Implement your model architecture")
    print("  2. Update MODEL section in config.yaml with your model")
    print("  3. Start training!")
    
    return 0


if __name__ == '__main__':
    exit(main())
