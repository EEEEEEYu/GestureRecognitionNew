"""
Helper function to create DVSGesture datasets from config.

This module provides a convenient function to create train/val/test datasets
from a configuration dictionary, automatically handling the second-stage
ratio_of_vectors parameter.
"""

from data import DVSGesturePrecomputed, collate_fn
from typing import Dict, Tuple
from torch.utils.data import DataLoader


def create_dvsgesture_datasets(config: Dict, purposes: list = ['train', 'validation']) -> Dict:
    """
    Create DVSGesture datasets from configuration.
    
    Args:
        config: Configuration dictionary loaded from YAML
        purposes: List of dataset purposes to create (e.g., ['train', 'validation'])
    
    Returns:
        Dictionary mapping purpose to dataset instance
        
    Example:
        ```python
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        datasets = create_dvsgesture_datasets(config)
        train_dataset = datasets['train']
        val_dataset = datasets['validation']
        ```
    """
    dataset_config = config['DATA']['dataset']['dataset_init_args']
    
    # Common parameters
    precomputed_dir = dataset_config['precomputed_dir']
    height = dataset_config.get('height', 128)
    width = dataset_config.get('width', 128)
    use_flip_augmentation = dataset_config.get('use_flip_augmentation', False)
    use_position_encoding = dataset_config.get('use_position_encoding', False)
    
    # Second-stage ratios (different for train/val)
    train_ratio = dataset_config.get('train_ratio_of_vectors', 0.8)
    val_ratio = dataset_config.get('val_ratio_of_vectors', 1.0)
    
    datasets = {}
    
    for purpose in purposes:
        # Determine ratio_of_vectors based on purpose
        if purpose == 'train':
            ratio_of_vectors = train_ratio
        else:  # validation, test
            ratio_of_vectors = val_ratio
        
        datasets[purpose] = DVSGesturePrecomputed(
            precomputed_dir=precomputed_dir,
            purpose=purpose,
            ratio_of_vectors=ratio_of_vectors,
            use_flip_augmentation=use_flip_augmentation if purpose == 'train' else False,
            height=height,
            width=width,
            use_position_encoding=use_position_encoding,
        )
    
    return datasets


def create_dvsgesture_dataloaders(
    config: Dict,
    purposes: list = ['train', 'validation']
) -> Dict:
    """
    Create DVSGesture dataloaders from configuration.
    
    Args:
        config: Configuration dictionary loaded from YAML
        purposes: List of dataset purposes to create
    
    Returns:
        Dictionary mapping purpose to DataLoader instance
        
    Example:
        ```python
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        dataloaders = create_dvsgesture_dataloaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['validation']
        
        for batch in train_loader:
            vectors = batch['vectors']
            event_coords = batch['event_coords']
            labels = batch['labels']
            # ... training code ...
        ```
    """
    # Create datasets
    datasets = create_dvsgesture_datasets(config, purposes)
    
    # Get dataloader config
    dataloader_config = config['DATA']['dataloader']
    
    dataloaders = {}
    
    for purpose in purposes:
        # Determine parameters based on purpose
        if purpose == 'train':
            batch_size = dataloader_config.get('batch_size', 32)
            shuffle = dataloader_config.get('shuffle_train', True)
        elif purpose == 'validation':
            batch_size = dataloader_config.get('test_batch_size', 64)
            shuffle = dataloader_config.get('shuffle_val', False)
        else:  # test
            batch_size = dataloader_config.get('test_batch_size', 64)
            shuffle = dataloader_config.get('shuffle_test', False)
        
        dataloaders[purpose] = DataLoader(
            datasets[purpose],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=dataloader_config.get('num_workers', 4),
            pin_memory=dataloader_config.get('pin_memory', True),
            persistent_workers=dataloader_config.get('persistent_workers', True),
            drop_last=dataloader_config.get('drop_last', False) if purpose == 'train' else False,
            multiprocessing_context=dataloader_config.get('multiprocessing_context', None),
            collate_fn=collate_fn,
        )
    
    return dataloaders


# Quick usage example
if __name__ == '__main__':
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating datasets from config...")
    datasets = create_dvsgesture_datasets(config)
    
    print(f"\nDatasets created:")
    for purpose, dataset in datasets.items():
        print(f"  {purpose}: {len(dataset)} samples")
        print(f"    ratio_of_vectors: {dataset.ratio_of_vectors}")
    
    print("\nCreating dataloaders from config...")
    dataloaders = create_dvsgesture_dataloaders(config)
    
    print(f"\nDataloaders created:")
    for purpose, loader in dataloaders.items():
        print(f"  {purpose}: {len(loader)} batches")
        print(f"    batch_size: {loader.batch_size}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for purpose, loader in dataloaders.items():
        for batch in loader:
            print(f"\n{purpose} batch:")
            print(f"  vectors: {len(batch['vectors'])} samples")
            print(f"  event_coords: {len(batch['event_coords'])} samples")
            print(f"  labels shape: {batch['labels'].shape}")
            print(f"  num_vectors_per_sample: {batch['num_vectors_per_sample'][:3]}...")
            break
