"""
Dataloader for precomputed DVSGesture dataset stored in HDF5 format.

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
    Comprehensive Augmentation for Event data and VecKM embeddings.
    
    Augmentations are applied stochastically based on probability parameters.
    All spatial augmentations operate on the event_coords [x, y, t, p, segment_id].
    
    Augmentation categories:
    1. Spatial: flip, rotation, translation, scale, cutout
    2. Temporal: time scaling, time reversal, temporal crop
    3. Event-specific: jitter, drop, polarity flip
    """
    def __init__(
        self, 
        # Spatial augmentations
        flip_prob: float = 0.0,           # Probability of horizontal/vertical flip
        rotation_prob: float = 0.0,        # Probability of rotation
        rotation_angles: list = None,      # List of possible rotation angles in degrees
        translate_prob: float = 0.0,       # Probability of random translation
        translate_range: float = 0.1,      # Max translation as fraction of width/height
        scale_prob: float = 0.0,           # Probability of random scaling
        scale_range: tuple = (0.9, 1.1),   # Min/max scale factors
        cutout_prob: float = 0.0,          # Probability of cutout augmentation
        cutout_size: float = 0.2,          # Size of cutout region as fraction
        
        # Temporal augmentations
        time_reverse_prob: float = 0.0,    # Probability of reversing temporal order
        temporal_crop_prob: float = 0.0,   # Probability of temporal cropping
        temporal_crop_min: float = 0.7,    # Min fraction of time to keep
        
        # Event-specific augmentations
        jitter_std: float = 0.0,           # Std of coordinate jitter
        drop_rate: float = 0.0,            # Rate of random event dropping
        polarity_flip_prob: float = 0.0,   # Probability of flipping polarity
        
        # Spatial dimensions for normalization
        height: int = 128,
        width: int = 128,
    ):
        # Spatial
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angles = rotation_angles or [90, 180, 270]
        self.translate_prob = translate_prob
        self.translate_range = translate_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        
        # Temporal
        self.time_reverse_prob = time_reverse_prob
        self.temporal_crop_prob = temporal_crop_prob
        self.temporal_crop_min = temporal_crop_min
        
        # Event-specific
        self.jitter_std = jitter_std
        self.drop_rate = drop_rate
        self.polarity_flip_prob = polarity_flip_prob
        
        # Dimensions
        self.height = height
        self.width = width
        
    def __call__(self, vectors: torch.Tensor, event_coords: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply augmentations.
        
        Args:
            vectors: [N, C] complex tensor (VecKM embeddings)
            event_coords: [N, 5] numpy array [x, y, t, p, segment_id]
            
        Returns:
            Augmented vectors and coords
        """
        if len(vectors) == 0:
            return vectors, event_coords
        
        # Make a copy to avoid modifying original
        event_coords = event_coords.copy()
        
        # =====================================================================
        # SPATIAL AUGMENTATIONS
        # =====================================================================
        
        # 1. Random Flip (horizontal and/or vertical)
        if self.flip_prob > 0:
            if random.random() < self.flip_prob:
                # Horizontal flip
                event_coords[:, 0] = self.width - event_coords[:, 0]
            if random.random() < self.flip_prob:
                # Vertical flip
                event_coords[:, 1] = self.height - event_coords[:, 1]
        
        # 2. Random Rotation (90, 180, 270 degrees)
        if self.rotation_prob > 0 and random.random() < self.rotation_prob:
            angle = random.choice(self.rotation_angles)
            cx, cy = self.width / 2, self.height / 2
            x = event_coords[:, 0] - cx
            y = event_coords[:, 1] - cy
            
            if angle == 90:
                event_coords[:, 0] = -y + cx
                event_coords[:, 1] = x + cy
            elif angle == 180:
                event_coords[:, 0] = -x + cx
                event_coords[:, 1] = -y + cy
            elif angle == 270:
                event_coords[:, 0] = y + cx
                event_coords[:, 1] = -x + cy
        
        # 3. Random Translation
        if self.translate_prob > 0 and random.random() < self.translate_prob:
            tx = random.uniform(-self.translate_range, self.translate_range) * self.width
            ty = random.uniform(-self.translate_range, self.translate_range) * self.height
            event_coords[:, 0] += tx
            event_coords[:, 1] += ty
            # Clip to valid range
            event_coords[:, 0] = np.clip(event_coords[:, 0], 0, self.width)
            event_coords[:, 1] = np.clip(event_coords[:, 1], 0, self.height)
        
        # 4. Random Scale (zoom in/out)
        if self.scale_prob > 0 and random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)
            cx, cy = self.width / 2, self.height / 2
            event_coords[:, 0] = (event_coords[:, 0] - cx) * scale + cx
            event_coords[:, 1] = (event_coords[:, 1] - cy) * scale + cy
            # Clip to valid range
            event_coords[:, 0] = np.clip(event_coords[:, 0], 0, self.width)
            event_coords[:, 1] = np.clip(event_coords[:, 1], 0, self.height)
        
        # 5. Cutout (remove rectangular region)
        if self.cutout_prob > 0 and random.random() < self.cutout_prob:
            cut_w = int(self.cutout_size * self.width)
            cut_h = int(self.cutout_size * self.height)
            cut_x = random.randint(0, self.width - cut_w)
            cut_y = random.randint(0, self.height - cut_h)
            
            # Create mask for events outside cutout region
            mask = ~(
                (event_coords[:, 0] >= cut_x) & (event_coords[:, 0] < cut_x + cut_w) &
                (event_coords[:, 1] >= cut_y) & (event_coords[:, 1] < cut_y + cut_h)
            )
            
            if mask.sum() > 0:  # Ensure we don't remove everything
                vectors = vectors[torch.from_numpy(mask)]
                event_coords = event_coords[mask]
        
        # =====================================================================
        # TEMPORAL AUGMENTATIONS
        # =====================================================================
        
        # 6. Temporal Scaling (speed up/slow down)
        # REMOVED: Time scaling should be done during preprocessing
        
        # 7. Time Reversal (reverse temporal order)
        if self.time_reverse_prob > 0 and random.random() < self.time_reverse_prob:
            t_max = event_coords[:, 2].max()
            event_coords[:, 2] = t_max - event_coords[:, 2]
            # Reverse the order of vectors and coords
            vectors = torch.flip(vectors, dims=[0])
            event_coords = event_coords[::-1].copy()
        
        # 8. Temporal Crop (select a time window)
        if self.temporal_crop_prob > 0 and random.random() < self.temporal_crop_prob:
            t_min, t_max = event_coords[:, 2].min(), event_coords[:, 2].max()
            t_range = t_max - t_min
            if t_range > 0:
                # Select a random window
                keep_frac = random.uniform(self.temporal_crop_min, 1.0)
                window_size = t_range * keep_frac
                window_start = t_min + random.uniform(0, t_range - window_size)
                window_end = window_start + window_size
                
                # Filter events
                mask = (event_coords[:, 2] >= window_start) & (event_coords[:, 2] <= window_end)
                if mask.sum() > 0:
                    vectors = vectors[torch.from_numpy(mask)]
                    event_coords = event_coords[mask]
        
        # =====================================================================
        # EVENT-SPECIFIC AUGMENTATIONS
        # =====================================================================
        
        # 9. Coordinate Jittering (perturb x, y, t)
        if self.jitter_std > 0:
            noise = np.random.normal(0, self.jitter_std, event_coords[:, :3].shape).astype(event_coords.dtype)
            event_coords[:, :3] += noise
            # Clip spatial coordinates
            event_coords[:, 0] = np.clip(event_coords[:, 0], 0, self.width)
            event_coords[:, 1] = np.clip(event_coords[:, 1], 0, self.height)
        
        # 10. Polarity Flip
        if self.polarity_flip_prob > 0 and random.random() < self.polarity_flip_prob:
            event_coords[:, 3] = 1 - event_coords[:, 3]  # Flip 0<->1
        
        # 11. Random Event Drop
        if self.drop_rate > 0:
            num_events = len(vectors)
            keep_prob = 1.0 - self.drop_rate
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
        height: int = 128,
        width: int = 128,
        # Augmentation parameters (only applied during training)
        augmentation: dict = None,
        # Legacy parameters (for backward compatibility)
        aug_jitter_std: float = 0.0,
        aug_drop_rate: float = 0.0,
        # Skip samples with label 10
        skip_label_10: bool = False,
    ):
        """
        DVS Gesture Precomputed Dataset Loader.
        
        Args:
            precomputed_dir: Directory containing precomputed HDF5 files
            purpose: 'train', 'validation', or 'test'
            height: Image height
            width: Image width
            augmentation: Dict of augmentation parameters (see Augmentor class)
                Supported keys:
                - flip_prob: Probability of horizontal/vertical flip
                - rotation_prob: Probability of rotation
                - rotation_angles: List of possible rotation angles
                - translate_prob: Probability of random translation
                - translate_range: Max translation as fraction of width/height
                - scale_prob: Probability of random scaling
                - scale_range: (min, max) scale factors
                - cutout_prob: Probability of cutout augmentation
                - cutout_size: Size of cutout region as fraction
                - time_scale_min/max: Time scaling range
                - time_reverse_prob: Probability of reversing temporal order
                - temporal_crop_prob: Probability of temporal cropping
                - temporal_crop_min: Min fraction of time to keep
                - jitter_std: Std of coordinate jitter
                - drop_rate: Rate of random event dropping
                - polarity_flip_prob: Probability of flipping polarity
            skip_label_10: If True, skip samples with label 10
        """
        assert purpose in ('train', 'validation', 'test'), f"Invalid purpose: {purpose}"
        
        self.precomputed_dir = precomputed_dir
        self.purpose = purpose

        # Initialize augmenter for training
        self.augmentor = None
        if purpose == 'train':
            # Use new augmentation dict if provided, otherwise fall back to legacy params
            if augmentation is not None:
                aug_params = {
                    'height': height,
                    'width': width,
                    **augmentation
                }
            else:
                # Legacy parameter support
                aug_params = {
                    'height': height,
                    'width': width,
                    'flip_prob': 0.0, # Runtime flip disabled
                    'jitter_std': aug_jitter_std,
                    'drop_rate': aug_drop_rate,
                }
            self.augmentor = Augmentor(**aug_params)
            
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
        
        # Filter out samples with label 10 if requested
        self.skip_label_10 = skip_label_10
        if skip_label_10:
            # Create mapping from filtered indices to original indices
            self.valid_indices = np.where(self.labels != 10)[0]
            self.num_samples = len(self.valid_indices)
            print(f"Filtered out {len(self.labels) - self.num_samples} samples with label 10")
        else:
            # No filtering - use all indices
            self.valid_indices = np.arange(self.num_samples)
        
        print(f"Loaded {self.num_samples} samples from {self.h5_path}")
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
        # Map filtered index to original index
        original_idx = self.valid_indices[idx]
        
        with h5py.File(self.h5_path, 'r') as h5f:
            sample_group = h5f[f'sample_{original_idx:06d}']
            num_intervals = self.num_intervals[original_idx]
            
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
        
        # Get label using original index
        label = self.labels[original_idx]
        
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
        # Map filtered index to original index
        original_idx = self.valid_indices[idx]
        return {
            'label': self.labels[original_idx],
            'file_path': self.file_paths[original_idx],
            'num_intervals': self.num_intervals[original_idx],
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
    dataset = DVSGesturePrecomputed(
        precomputed_dir=args.precomputed_dir,
        purpose=args.purpose,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Sample info: {sample['label']}")
    exit()
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
