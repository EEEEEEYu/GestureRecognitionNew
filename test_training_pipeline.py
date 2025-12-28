#!/usr/bin/env python3
"""
Test the complete training pipeline for DVSGesture.

This script:
1. Validates precomputed data
2. Creates dataloaders
3. Initializes model
4. Tests forward pass
5. Optionally runs a few training steps
"""

import torch
import yaml
import argparse
from pathlib import Path

# Validation and data loading
from validate_precomputed_data import validate_hdf5_file
from data import create_dvsgesture_dataloaders

# Model
from model.sparse_hilbert_ssm import SparseHilbertSSM


def test_data_validation(precomputed_dir, max_samples=10):
    """Test 1: Validate precomputed data."""
    print("\n" + "="*60)
    print("TEST 1: Data Validation")
    print("="*60)
    
    train_path = Path(precomputed_dir) / 'train.h5'
    val_path = Path(precomputed_dir) / 'validation.h5'
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return False
    
    print(f"\nValidating {max_samples} samples from train set...")
    train_valid = validate_hdf5_file(str(train_path), max_samples=max_samples, verbose=False)
    
    if not train_valid:
        return False
    
    print(f"\nValidating {max_samples} samples from val set...")
    val_valid = validate_hdf5_file(str(val_path), max_samples=max_samples, verbose=False)
    
    return train_valid and val_valid


def test_dataloader(config):
    """Test 2: Create and test dataloaders."""
    print("\n" + "="*60)
    print("TEST 2: Dataloader Creation")
    print("="*60)
    
    try:
        loaders = create_dvsgesture_dataloaders(config)
        train_loader = loaders['train']
        val_loader = loaders['validation']
        
        print(f"✅ Dataloaders created successfully")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        
        # Test loading one batch
        print(f"\nTesting batch loading...")
        for batch in train_loader:
            vectors = batch['vectors']
            coords = batch['event_coords']
            labels = batch['labels']
            
            print(f"✅ Batch loaded successfully")
            print(f"  Batch size: {len(vectors)}")
            print(f"  Labels: {labels}")
            print(f"  Sample 0 vectors shape: {vectors[0].shape}")
            print(f"  Sample 0 coords shape: {coords[0].shape}")
            
            return batch
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_creation(config):
    """Test 3: Create model."""
    print("\n" + "="*60)
    print("TEST 3: Model Creation")
    print("="*60)
    
    try:
        # Resolve num_classes from config (handle YAML interpolation)
        num_classes_raw = config['MODEL']['model_init_args']['num_classes']
        if isinstance(num_classes_raw, str) and num_classes_raw.startswith('${'):
            # It's a reference, resolve it
            num_classes = config['DATA']['dataset']['dataset_init_args']['num_classes']
        else:
            num_classes = num_classes_raw
        
        model = SparseHilbertSSM(
            encoding_dim=config['MODEL']['model_init_args']['encoding_dim'],
            hidden_dim=config['MODEL']['model_init_args']['hidden_dim'],
            num_classes=num_classes,  # Use resolved value
            num_ssm_layers=config['MODEL']['model_init_args']['num_layers'],
            d_state=config['MODEL']['model_init_args']['d_state'],
            d_conv=config['MODEL']['model_init_args']['d_conv'],
            expand=config['MODEL']['model_init_args']['expand'],
            dropout=config['MODEL']['model_init_args']['dropout'],
            pooling_scales=config['MODEL']['model_init_args']['pooling_scales'],
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB (fp32)")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, batch, device='cpu'):
    """Test 4: Forward pass."""
    print("\n" + "="*60)
    print("TEST 4: Forward Pass")
    print("="*60)
    
    try:
        model = model.to(device)
        model.eval()
        
        # Move batch to device if needed
        if device != 'cpu':
            batch_device = {
                'vectors': [v.to(device) for v in batch['vectors']],
                'event_coords': batch['event_coords'],  # Keep on CPU (numpy)
                'labels': batch['labels'].to(device),
            }
        else:
            batch_device = batch
        
        print(f"Running forward pass on {device}...")
        with torch.no_grad():
            logits = model(batch_device)
        
        print(f"✅ Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: [{len(batch['vectors'])}, {model.num_classes}]")
        
        # Check outputs
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        print(f"\n  Predictions: {preds.cpu().numpy()}")
        print(f"  Ground truth: {batch['labels'].cpu().numpy()}")
        print(f"  Max probabilities: {probs.max(dim=1)[0].cpu().numpy()}")
        
        return logits
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_backward_pass(model, batch, device='cpu'):
    """Test 5: Backward pass."""
    print("\n" + "="*60)
    print("TEST 5: Backward Pass")
    print("="*60)
    
    try:
        model = model.to(device)
        model.train()
        
        # Move batch to device if needed
        if device != 'cpu':
            batch_device = {
                'vectors': [v.to(device) for v in batch['vectors']],
                'event_coords': batch['event_coords'],
                'labels': batch['labels'].to(device),
            }
        else:
            batch_device = batch
        
        # Forward pass
        logits = model(batch_device)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, batch_device['labels'])
        
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        print(f"Running backward pass...")
        loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        print(f"✅ Backward pass successful!")
        print(f"  Gradient norm (mean): {sum(grad_norms) / len(grad_norms):.6f}")
        print(f"  Gradient norm (max): {max(grad_norms):.6f}")
        print(f"  Gradient norm (min): {min(grad_norms):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test DVSGesture training pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--max_val_samples', type=int, default=10,
                      help='Number of samples to validate')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--skip_validation', action='store_true',
                      help='Skip data validation (faster)')
    args = parser.parse_args()
    
    print("="*60)
    print("DVSGesture Training Pipeline Test")
    print("="*60)
    
    # Load config
    print("\nLoading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded")
    
    # Test 1: Validate data
    if not args.skip_validation:
        precomputed_dir = config['DATA']['dataset']['dataset_init_args']['precomputed_dir']
        if not test_data_validation(precomputed_dir, args.max_val_samples):
            print("\n❌ Data validation failed. Fix issues before proceeding.")
            return 1
    else:
        print("\n⚠️  Skipping data validation")
    
    # Test 2: Dataloader
    batch = test_dataloader(config)
    if batch is None:
        return 1
    
    # Test 3: Model creation
    model = test_model_creation(config)
    if model is None:
        return 1
    
    # Test 4: Forward pass
    logits = test_forward_pass(model, batch, args.device)
    if logits is None:
        return 1
    
    # Test 5: Backward pass
    if not test_backward_pass(model, batch, args.device):
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour pipeline is ready for training!")
    print("\nNext steps:")
    print("  1. Run full preprocessing: ./run_preprocess.sh")
    print("  2. Start training: python main.py --config configs/config.yaml")
    
    return 0


if __name__ == '__main__':
    exit(main())
