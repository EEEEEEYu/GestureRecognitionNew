"""
Dataloader for precomputed DVSGesture dataset stored in HDF5 format.

This dataloader:
1. Loads precomputed complex tensors from HDF5 files
2. Implements second-stage downsampling with ratio_of_vectors for data augmentation
3. Supports train/test modes with optional data augmentation
"""

import torch
import torch.utils.data as data
import h5py
import numpy as np
import os
from typing import Optional, Tuple, List
import random


class Augmentor:
    """
    Robust Augmentation for Event lists and Point Clouds.
    """
    def __init__(
        self, 
        jitter_std: float = 0.0,
        drop_rate: float = 0.0,
        time_scale_min: float = 1.0,
        time_scale_max: float = 1.0
    ):
        self.jitter_std = jitter_std
        self.drop_rate = drop_rate
        self.time_scale_min = time_scale_min
        self.time_scale_max = time_scale_max
        
    def __call__(self, vectors: torch.Tensor, event_coords: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply augmentations.
        
        Args:
            vectors: [N, C] complex tensor
            event_coords: [N, 4] numpy array [x, y, t, p]
            
        Returns:
            Augmented vectors and coords
        """
        if len(vectors) == 0:
            return vectors, event_coords
            
        # 1. Temporal Scaling (Traveral Speed Augmentation)
        # Scales the 't' coordinate, which affects relative temporal ordering sorting
        if self.time_scale_min != 1.0 or self.time_scale_max != 1.0:
            scale = random.uniform(self.time_scale_min, self.time_scale_max)
            # t is at index 2
            event_coords[:, 2] *= scale
            
        # 2. Coordinate Jittering (Traversal Order Augmentation)
        # Adds noise to x,y,t to perturb Hilbert curve sorting order
        if self.jitter_std > 0:
            noise = np.random.normal(0, self.jitter_std, event_coords[:, :3].shape).astype(event_coords.dtype)
            event_coords[:, :3] += noise
            
        # 3. Event Drop (Robustness to Occlusion/Noise)
        # Randomly drops entire vectors from the sequence
        if self.drop_rate > 0:
            num_events = len(vectors)
            keep_prob = 1.0 - self.drop_rate
            # Create mask
            mask = torch.rand(num_events) < keep_prob
            
            # Ensure we don't drop everything
            if mask.sum() == 0:
                mask[0] = True
                
            vectors = vectors[mask]
            event_coords = event_coords[mask.numpy()]
            
        return vectors, event_coords


class DVSGesturePrecomputed(data.Dataset):
    def __init__(
        self,
        precomputed_dir: str,
        purpose: str = 'train',
        ratio_of_vectors: float = 1.0,
        use_flip_augmentation: bool = False,
        aug_jitter_std: float = 0.0,
        aug_drop_rate: float = 0.0,
        aug_time_scale_min: float = 1.0,
        aug_time_scale_max: float = 1.0,
        height: int = 128,
        width: int = 128,
        use_position_encoding: bool = False,
    ):
        """
        DVS Gesture Precomputed Dataset Loader.
        
        Args:
            precomputed_dir: Directory containing precomputed HDF5 files
            purpose: 'train' or 'validation'
            ratio_of_vectors: Second-stage downsampling ratio (0.0-1.0)
                             - Positive values: sample this percentage of vectors
                             - 1.0: use all precomputed vectors
                             This provides data augmentation during training
            use_flip_augmentation: Whether to apply spatial flip augmentation
            height: Image height (for flip augmentation)
            width: Image width (for flip augmentation)
        """
        assert purpose in ('train', 'validation', 'test'), f"Invalid purpose: {purpose}"
        
        self.precomputed_dir = precomputed_dir
        self.purpose = purpose
        self.ratio_of_vectors = ratio_of_vectors
        self.use_flip_augmentation = use_flip_augmentation
        self.use_position_encoding = use_position_encoding
        
        # Initialize augmenter for training
        self.augmentor = None
        if purpose == 'train':
            self.augmentor = Augmentor(
                jitter_std=aug_jitter_std,
                drop_rate=aug_drop_rate,
                time_scale_min=aug_time_scale_min,
                time_scale_max=aug_time_scale_max
            )
            
        self.height = height
        self.width = width
        
        # Map test to validation file (if test split doesn't exist)
        file_purpose = 'validation' if purpose == 'test' else purpose
        self.h5_path = os.path.join(precomputed_dir, f'{file_purpose}.h5')
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Precomputed file not found: {self.h5_path}")
        
        # Open HDF5 file and load metadata
        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_samples = h5f['labels'].shape[0]
            
            # Store metadata
            self.accumulation_interval_ms = h5f.attrs['accumulation_interval_ms']
            self.precompute_ratio = h5f.attrs['ratio_of_vectors']
            self.encoding_dim = h5f.attrs['encoding_dim']
            self.temporal_length = h5f.attrs['temporal_length']
            
            # Load rotation metadata if available
            self.rotation_enabled = h5f.attrs.get('rotation_enabled', False)
            self.rotation_angles_list = list(h5f.attrs.get('rotation_angles_list', [0]))
            
            # Load all metadata into memory for fast access
            self.labels = h5f['labels'][:].astype(np.int64)
            self.file_paths = [fp.decode('utf-8') if isinstance(fp, bytes) else fp 
                             for fp in h5f['file_paths'][:]]
            self.num_intervals = h5f['num_intervals'][:]
            self.rotation_angles = h5f['rotation_angles'][:].astype(np.int32) if 'rotation_angles' in h5f else np.zeros(self.num_samples, dtype=np.int32)
        
        print(f"Loaded {self.num_samples} samples from {self.h5_path}")
        print(f"  Encoding dim: {self.encoding_dim}")
        print(f"  Precompute ratio (1st stage): {self.precompute_ratio}")
        print(f"  Training ratio (2nd stage): {self.ratio_of_vectors}")
        if self.rotation_enabled:
            print(f"  Rotation augmentation: enabled with angles {self.rotation_angles_list}")
            print(f"  Rotation distribution: {dict(zip(*np.unique(self.rotation_angles, return_counts=True)))}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a precomputed sample.
        
        Returns:
            Dictionary containing:
                - 'vectors': Complex tensor of shape [total_vectors, encoding_dim]
                - 'event_coords': Event coordinates array [total_vectors, 4] with columns [x, y, t, p]
                - 'num_vectors_per_interval': List of vector counts per interval
                - 'label': Class label
                - 'file_path': Original file path
        """
        with h5py.File(self.h5_path, 'r') as h5f:
            sample_group = h5f[f'sample_{idx:06d}']
            num_intervals = self.num_intervals[idx]
            
            # Load all intervals
            all_vectors = []
            all_event_coords = []
            num_vectors_per_interval = []
            
            for interval_idx in range(num_intervals):
                interval_group = sample_group[f'interval_{interval_idx:03d}']
                
                # Load real and imaginary parts
                real_part = torch.from_numpy(interval_group['real'][:])
                imag_part = torch.from_numpy(interval_group['imag'][:])
                
                # Load event coordinates [num_vectors, 4] with columns [x, y, t, p]
                event_coords = interval_group['event_coords'][:]
                
                # Reconstruct complex tensor
                vectors = torch.complex(real_part, imag_part)
                
                # Apply second-stage downsampling if needed
                if self.ratio_of_vectors < 1.0 and len(vectors) > 0:
                    num_to_sample = max(1, int(len(vectors) * self.ratio_of_vectors))
                    
                    # Random sampling for training augmentation
                    if num_to_sample < len(vectors):
                        indices = torch.randperm(len(vectors))[:num_to_sample]
                        indices = torch.sort(indices)[0]  # Sort for better cache locality
                        
                        # Apply same sampling to both vectors and coordinates
                        vectors = vectors[indices]
                        event_coords = event_coords[indices.numpy()]
                
                all_vectors.append(vectors)
                all_event_coords.append(event_coords)
                num_vectors_per_interval.append(len(vectors))
            
                # Concatenate all intervals into single tensors
            if len(all_vectors) > 0 and sum(num_vectors_per_interval) > 0:
                vectors_concatenated = torch.cat(all_vectors, dim=0)
                event_coords_concatenated = np.concatenate(all_event_coords, axis=0)
                
                # Apply Robust Augmentations (Jitter, Drop, TimeScale)
                # This modifies event coordinates and potentially drops vectors
                if self.augmentor is not None:
                    vectors_concatenated, event_coords_concatenated = self.augmentor(
                        vectors_concatenated, event_coords_concatenated
                    )
                    
            else:
                # Handle empty case
                vectors_concatenated = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
                event_coords_concatenated = np.zeros((0, 4), dtype=np.float32)
        
        # Get label
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        # Apply position encoding if enabled
        if self.use_position_encoding and len(vectors_concatenated) > 0:
            # Normalize coordinates to [0, 1]
            coords_tensor = torch.from_numpy(event_coords_concatenated).float()
            x_norm = coords_tensor[:, 0] / self.width  # x at index 0
            y_norm = coords_tensor[:, 1] / self.height  # y at index 1
            
            # Normalize time relative to the interval
            t_coords = coords_tensor[:, 2]  # t at index 2
            t_min = t_coords.min()
            t_max = t_coords.max()
            t_range = t_max - t_min
            t_norm = (t_coords - t_min) / (t_range + 1e-6) if t_range > 0 else torch.zeros_like(t_coords)
            
            # Convert complex vectors to real: [real, imag]
            vectors_real = torch.view_as_real(vectors_concatenated).reshape(vectors_concatenated.shape[0], -1)
            
            # Concatenate: [real_part (encoding_dim), imag_part (encoding_dim), x_norm, y_norm, t_norm]
            # Shape: (N, encoding_dim*2 + 3)
            position_features = torch.stack([x_norm, y_norm, t_norm], dim=1)  # (N, 3)
            vectors_concatenated = torch.cat([vectors_real, position_features], dim=1)  # (N, encoding_dim*2+3)
        
        return {
            'vectors': vectors_concatenated,
            'event_coords': event_coords_concatenated,  # [total_vectors, 4] with [x, y, t, p]
            'num_vectors_per_interval': num_vectors_per_interval,
            'label': label,
            'file_path': file_path,
            'num_intervals': num_intervals,
        }
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Get information about a sample without loading the vectors.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with sample metadata
        """
        return {
            'label': self.labels[idx],
            'file_path': self.file_paths[idx],
            'num_intervals': self.num_intervals[idx],
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for batching precomputed samples.
    
    Since different samples may have different numbers of vectors,
    we need to handle variable-length sequences.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary with vectors and event coordinates
    """
    # Extract components
    vectors_list = [sample['vectors'] for sample in batch]
    event_coords_list = [sample['event_coords'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    num_vectors_per_sample = [sample['vectors'].shape[0] for sample in batch]
    
    # Option 1: Return as list (for variable-length processing)
    # This is useful if your model can handle variable-length inputs
    
    # Option 2: Pad to common length (uncomment if needed)
    # max_vectors = max(num_vectors_per_sample)
    # encoding_dim = vectors_list[0].shape[1] if len(vectors_list[0]) > 0 else 64
    # 
    # padded_vectors = torch.zeros(len(batch), max_vectors, encoding_dim, dtype=torch.cfloat)
    # padded_coords = torch.zeros(len(batch), max_vectors, 4, dtype=torch.float32)
    # mask = torch.zeros(len(batch), max_vectors, dtype=torch.bool)
    # 
    # for i, (vectors, coords) in enumerate(zip(vectors_list, event_coords_list)):
    #     if len(vectors) > 0:
    #         padded_vectors[i, :len(vectors)] = vectors
    #         padded_coords[i, :len(vectors)] = torch.from_numpy(coords)
    #         mask[i, :len(vectors)] = True
    
    return {
        'vectors': vectors_list,  # List of tensors with different lengths
        'event_coords': event_coords_list,  # List of arrays [num_vectors, 4] with [x, y, t, p]
        'labels': labels,  # [batch_size]
        'num_vectors_per_sample': num_vectors_per_sample,  # List of ints
        'num_vectors_per_interval': [sample['num_vectors_per_interval'] for sample in batch],
        'file_paths': [sample['file_path'] for sample in batch],
        'num_intervals': [sample['num_intervals'] for sample in batch],
    }


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed_dir', type=str, required=True)
    parser.add_argument('--purpose', type=str, default='train')
    args = parser.parse_args()
    
    # Create dataset
    dataset = DVSGesturePrecomputed(
        precomputed_dir=args.precomputed_dir,
        purpose=args.purpose,
        ratio_of_vectors=0.8,  # Use 80% of precomputed vectors
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Vectors shape: {sample['vectors'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Num intervals: {sample['num_intervals']}")
    print(f"  Num vectors per interval: {sample['num_vectors_per_interval']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    print(f"\nTesting dataloader with batch_size=4...")
    for batch in dataloader:
        print(f"Batch:")
        print(f"  Num samples: {len(batch['labels'])}")
        print(f"  Labels: {batch['labels']}")
        print(f"  Num vectors per sample: {batch['num_vectors_per_sample']}")
        break
