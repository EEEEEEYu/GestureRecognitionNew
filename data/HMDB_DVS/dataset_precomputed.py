"""
Dataloader for precomputed HMDB-DVS dataset stored in HDF5 format.

This dataloader:
1. Loads precomputed complex tensors from HDF5 files
2. Implements second-stage downsampling for data augmentation
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


class HMDB_DVS_Precomputed(data.Dataset):
    def __init__(
        self,
        dataset_dir: str,  # This should be the precomputed data directory
        purpose: str = 'train',
        use_flip_augmentation: bool = False,
        aug_jitter_std: float = 0.0,
        aug_drop_rate: float = 0.0,
        aug_time_scale_min: float = 1.0,
        aug_time_scale_max: float = 1.0,
        height: int = 180,
        width: int = 240,
        num_classes: int = 51,
        accumulation_interval_ms: float = 200.0,
        train_split: float = 0.8,
    ):
        """
        HMDB-DVS Precomputed Dataset Loader.
        
        Args:
            dataset_dir: Directory containing precomputed HDF5 files
            purpose: 'train' or 'validation'
            use_flip_augmentation: Whether to apply spatial flip augmentation
            height: Image height (for normalization)
            width: Image width (for normalization)
            num_classes: Number of classes (51 for HMDB-DVS)
            accumulation_interval_ms: Time interval for event accumulation (not used, loaded from H5)
            train_split: Train split ratio (not used, loaded from H5)
        """
        assert purpose in ('train', 'validation', 'test'), f"Invalid purpose: {purpose}"
        
        self.dataset_dir = dataset_dir
        self.purpose = purpose
        self.use_flip_augmentation = use_flip_augmentation
        self.num_classes = num_classes
        
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
        self.h5_path = os.path.join(dataset_dir, f'{file_purpose}.h5')
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Precomputed file not found: {self.h5_path}")
        
        # Open HDF5 file and load metadata
        with h5py.File(self.h5_path, 'r') as h5f:
            self.num_samples = h5f['labels'].shape[0]
            
            # Store metadata
            self.accumulation_interval_ms = h5f.attrs['accumulation_interval_ms']
            # self.precompute_ratio = h5f.attrs['ratio_of_vectors'] # Removed
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
            
            # Filter out empty samples (samples with no vectors in any interval)
            print(f"Filtering empty samples from {self.num_samples} total samples...")
            valid_indices = []
            for idx in range(self.num_samples):
                sample_group = h5f[f'sample_{idx:06d}']
                num_intervals = self.num_intervals[idx]
                
                # Check if at least one interval has vectors
                has_vectors = False
                for interval_idx in range(num_intervals):
                    interval_group = sample_group[f'interval_{interval_idx:03d}']
                    if interval_group['real'].shape[0] > 0:  # Has at least one vector
                        has_vectors = True
                        break
                
                if has_vectors:
                    valid_indices.append(idx)
            # Store original count for reporting
            original_count = h5f['labels'].shape[0]
            
            # Filter metadata to only include valid samples
            self.valid_indices = np.array(valid_indices)
            self.labels = self.labels[self.valid_indices]
            self.file_paths = [self.file_paths[i] for i in valid_indices]
            self.num_intervals = self.num_intervals[self.valid_indices]
            self.rotation_angles = self.rotation_angles[self.valid_indices]
            self.num_samples = len(self.valid_indices)
        
        print(f"Loaded {self.num_samples} non-empty samples from {self.h5_path}")
        print(f"  Filtered out {original_count - self.num_samples} empty samples")
        print(f"  Encoding dim: {self.encoding_dim}")
        if self.rotation_enabled:
            print(f"  Rotation augmentation: enabled with angles {self.rotation_angles_list}")
            print(f"  Rotation distribution: {dict(zip(*np.unique(self.rotation_angles, return_counts=True)))}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Load and return a precomputed sample in unified format.
        
        Returns:
            Dictionary containing:
                - 'segments_complex': List[Tensor] of [N_i, encoding_dim] for each segment
                - 'segments_coords': List[Tensor] of [N_i, 2] with [x, y] coords
                - 'label': Class label
                - 'num_segments': Number of segments
                - 'num_vectors_per_segment': List of vector counts
        """
        with h5py.File(self.h5_path, 'r') as h5f:
            # Map the index to the actual valid sample index
            actual_idx = self.valid_indices[idx]
            sample_group = h5f[f'sample_{actual_idx:06d}']
            num_intervals = self.num_intervals[idx]
            
            # Load all intervals
            all_vectors = []
            all_event_coords = []
            
            for interval_idx in range(num_intervals):
                interval_group = sample_group[f'interval_{interval_idx:03d}']
                
                # Load real and imaginary parts
                real_part = torch.from_numpy(interval_group['real'][:])
                imag_part = torch.from_numpy(interval_group['imag'][:])
                
                # Load event coordinates [num_vectors, 5] with columns [x, y, t, p, segment_id]
                event_coords = interval_group['event_coords'][:]
                
                # Reconstruct complex tensor
                vectors = torch.complex(real_part, imag_part)
                
                # Skip empty intervals to avoid concatenation issues
                if len(vectors) == 0:
                    continue
                
                # Removed second-stage downsampling (ratio_of_vectors) logic
                # We now rely on aug_drop_rate in Augmentor for training control
                
                all_vectors.append(vectors)
                all_event_coords.append(event_coords)
            
            # Concatenate all intervals into single tensors
            if len(all_vectors) > 0 and sum([len(v) for v in all_vectors]) > 0:
                vectors_concatenated = torch.cat(all_vectors, dim=0)
                event_coords_concatenated = np.concatenate(all_event_coords, axis=0)
                
                # Apply Robust Augmentations (Jitter, Drop, TimeScale)
                if self.augmentor is not None:
                    vectors_concatenated, event_coords_concatenated = self.augmentor(
                        vectors_concatenated, event_coords_concatenated
                    )
            else:
                # Handle empty case
                vectors_concatenated = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
                event_coords_concatenated = np.zeros((0, 5), dtype=np.float32)
        
        # Get label
        label = self.labels[idx]
        
        # ========================================================================
        # RESTRUCTURE BY SEGMENTS using segment_id column
        # ========================================================================
        if len(vectors_concatenated) > 0:
            # Extract segment IDs from column 4
            segment_ids = event_coords_concatenated[:, 4].astype(int)
            unique_segments = np.unique(segment_ids)
            
            segments_complex = []
            segments_coords = []
            num_vectors_per_segment = []
            
            for seg_id in sorted(unique_segments):
                # Get mask for this segment
                mask = segment_ids == seg_id
                
                # Extract vectors for this segment
                seg_vectors = vectors_concatenated[mask]
                seg_coords = event_coords_concatenated[mask, :2]  # Just x, y
                

                
                # Convert coords to tensor and normalize
                seg_coords_tensor = torch.from_numpy(seg_coords).float()
                seg_coords_tensor[:, 0] /= self.width  # Normalize x
                seg_coords_tensor[:, 1] /= self.height  # Normalize y
                
                segments_complex.append(seg_vectors)
                segments_coords.append(seg_coords_tensor)
                num_vectors_per_segment.append(len(seg_vectors))
            
            num_segments = len(segments_complex)
        else:
            # Empty sample
            segments_complex = []
            segments_coords = []
            num_vectors_per_segment = []
            num_segments = 0
        
        return {
            'segments_complex': segments_complex,  # List[Tensor] of [N_i, D]
            'segments_coords': segments_coords,     # List[Tensor] of [N_i, 2]
            'label': label,
            'num_segments': num_segments,
            'num_vectors_per_segment': num_vectors_per_segment,
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
    Custom collate function for batching precomputed samples in unified format.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary with segments and labels
    """
    # Extract components
    segments_complex_list = [sample['segments_complex'] for sample in batch]
    segments_coords_list = [sample['segments_coords'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.long)
    
    return {
        'segments_complex': segments_complex_list,  # List[List[Tensor]] - batch of segment lists
        'segments_coords': segments_coords_list,     # List[List[Tensor]] - batch of coord lists
        'labels': labels,  # [batch_size]
    }


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed_dir', type=str, required=True)
    parser.add_argument('--purpose', type=str, default='train')
    args = parser.parse_args()
    
    # Create dataset
    dataset = HMDB_DVS_Precomputed(
        dataset_dir=args.precomputed_dir,
        purpose=args.purpose,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Num segments: {sample['num_segments']}")
    print(f"  Label: {sample['label']}")
    print(f"  Num vectors per segment: {sample['num_vectors_per_segment']}")
    
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
        break
