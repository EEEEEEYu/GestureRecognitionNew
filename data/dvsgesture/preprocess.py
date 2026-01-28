"""
Pre-computing script for DVSGesture dataset using SparseVKMEncoder.

This script:
1. Loads raw events from DVSGesture dataset
2. Encodes events into complex tensors using SparseVKMEncoder
3. Implements two-stage downsampling
4. Stores precomputed tensors in HDF5 format
5. Supports checkpointing for resume capability
"""

import os
import sys
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from data.dvsgesture.dataset import DVSGesture
from data.SparseVKMEncoderOptimized import VecKMSparseOptimized
from utils.event_augmentation import rotate_sliced_events
from utils.denoising_and_sampling import filter_noise_spatial


class DVSGesturePreprocessor:
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary containing PRECOMPUTING parameters
        """
        self.config = config
        precompute_cfg = config['PRECOMPUTING']
        
        self.dataset_dir = precompute_cfg['dataset_dir']
        self.output_dir = precompute_cfg['output_dir']
        self.accumulation_interval_ms = float(precompute_cfg['accumulation_interval_ms'])
        self.encoding_dim = int(precompute_cfg['encoding_dim'])
        self.temporal_length = float(precompute_cfg['temporal_length'])
        self.kernel_size = int(precompute_cfg['kernel_size'])
        self.T_scale = float(precompute_cfg['T_scale'])
        self.S_scale = float(precompute_cfg['S_scale'])
        self.height = int(precompute_cfg['height'])
        self.width = int(precompute_cfg['width'])
        self.checkpoint_every_n = int(precompute_cfg['checkpoint_every_n_samples'])
        
        # Rotation augmentation configuration (optional)
        rotation_cfg = OmegaConf.select(precompute_cfg, 'rotation_augmentation', default={})
        self.rotation_enabled = OmegaConf.select(rotation_cfg, 'enabled', default=False)
        self.rotation_angles = OmegaConf.select(rotation_cfg, 'angles', default=[0])
        self.rotation_mode = OmegaConf.select(rotation_cfg, 'mode', default='separate')
        self.augment_validation = OmegaConf.select(rotation_cfg, 'augment_validation', default=False)
        
        # Denoising configuration (applied BEFORE sampling)
        denoising_cfg = OmegaConf.select(precompute_cfg, 'denoising', default={})
        self.denoising_enabled = OmegaConf.select(denoising_cfg, 'enabled', default=False)
        self.denoising_grid_size = int(OmegaConf.select(denoising_cfg, 'grid_size', default=4))
        self.denoising_threshold = int(OmegaConf.select(denoising_cfg, 'threshold', default=2))
        
        # Sampling strategy configuration (applied AFTER denoising)
        sampling_cfg = OmegaConf.select(precompute_cfg, 'sampling', default={})
        self.sampling_method = OmegaConf.select(sampling_cfg, 'method', default='random')
        self.adaptive_sampling_cfg = OmegaConf.select(sampling_cfg, 'adaptive_striding', default={'kernel_size': 17, 'overlap_factor': 0.0})
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        print(f"Denoising: {'enabled' if self.denoising_enabled else 'disabled'}")
        if self.denoising_enabled:
            print(f"  Grid size: {self.denoising_grid_size}, Threshold: {self.denoising_threshold}")
        print(f"Sampling method: {self.sampling_method}")
        if self.sampling_method == 'random':
            print(f"  Random sampling selected (Keeping all events)")
        elif self.sampling_method == 'adaptive_striding':
            print(f"  Adaptive striding selected (Kernel size: {self.adaptive_sampling_cfg['kernel_size']}, Overlap factor: {self.adaptive_sampling_cfg['overlap_factor']})")
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        self.encoder = VecKMSparseOptimized(
            height=self.height,
            width=self.width,
            encoding_dim=self.encoding_dim,
            temporal_length=self.temporal_length,
            kernel_size=self.kernel_size,
            T_scale=self.T_scale,
            S_scale=self.S_scale,
        ).to(self.device)
        
        self.checkpoint_file = os.path.join(self.output_dir, 'checkpoint.json')
        
    def get_checkpoint_state(self, purpose: str) -> Dict:
        """Load checkpoint state for a specific purpose (train/validation)."""
        if not os.path.exists(self.checkpoint_file):
            return {'train': {'processed_samples': 0}, 'validation': {'processed_samples': 0}}
        
        with open(self.checkpoint_file, 'r') as f:
            state = json.load(f)
        
        if purpose not in state:
            state[purpose] = {'processed_samples': 0}
        
        return state
    
    def save_checkpoint_state(self, state: Dict):
        """Save checkpoint state."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def encode_events_for_interval(
        self,
        events_xy: np.ndarray,
        events_t: np.ndarray,
        events_p: np.ndarray,
        segment_id: int,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Encode events for a single time interval using the pipeline:
        1. Denoise (optional, pure numpy)
        2. Sample (pure numpy) - Now using Kernel-Aware Adaptive Sampling
        3. VecKM Encode (torch)
        
        Args:
            events_xy: Event coordinates [N, 2]
            events_t: Event timestamps [N]
            events_p: Event polarities [N]
            segment_id: Segment/interval index for this batch of events
        
        Returns:
            Tuple of:
                - Complex tensor of shape [num_vectors, encoding_dim]
                - Event coordinates array [num_vectors, 5] with columns [x, y, t, p, segment_id]
        """
        num_events = len(events_t)
        
        if num_events == 0:
            # Return empty tensors for empty intervals
            empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
            empty_coords = np.zeros((0, 5), dtype=np.float32)
            return empty_embeddings, empty_coords
        
        # Extract separate x, y coordinates
        events_y = events_xy[:, 1]
        events_x = events_xy[:, 0]
        
        # ===================================================================
        # BOUNDS CHECKING: Clip coordinates to valid range
        # ===================================================================
        events_x = np.clip(events_x, 0, self.width - 1)
        events_y = np.clip(events_y, 0, self.height - 1)
        
        # ===================================================================
        # STAGE 1: DENOISING (Optional, Pure Numpy)
        # ===================================================================
        if self.denoising_enabled:
            # Apply denoising (pure numpy, no conversion)
            events_t_clean, events_y_clean, events_x_clean, events_p_clean = filter_noise_spatial(
                events_t, events_y, events_x, events_p,
                self.height, self.width,
                self.denoising_grid_size,
                self.denoising_threshold
            )
            
            num_events_after_denoise = len(events_t_clean)
            
            if num_events_after_denoise == 0:
                # All events were filtered out
                empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
                empty_coords = np.zeros((0, 5), dtype=np.float32)
                return empty_embeddings, empty_coords
        else:
            # No denoising, use original events
            events_t_clean = events_t
            events_y_clean = events_y
            events_x_clean = events_x
            events_p_clean = events_p
            num_events_after_denoise = num_events
        
        # ===================================================================
        # STAGE 2: SAMPLING (Kernel-Aware Adaptive)
        # ===================================================================
        from utils.density_adaptive_spatial_striding import adaptive_spatial_sampling
        
        if self.sampling_method == 'adaptive_striding':
            # NEW: Kernel-Aware Adaptive Striding
            # Uses: Kernel Size and Overlap Factor
            
            kernel_size = self.adaptive_sampling_cfg.get('kernel_size', 17)
            overlap_factor = self.adaptive_sampling_cfg.get('overlap_factor', 0.2)
            
            query_indices = adaptive_spatial_sampling(
                events_t_clean, events_y_clean, events_x_clean, events_p_clean,
                height=self.height, width=self.width,
                kernel_size=kernel_size,
                overlap_factor=overlap_factor,
                sort_by_time=True  # Prefer newest events
            )
            
            num_vectors = len(query_indices)
            
        elif self.sampling_method == 'random':
            # FALLBACK: Random sampling (Pure Numpy) - defaults to keeping all if no ratio logic
            # Since ratio_of_vectors is removed, we keep all events or maybe implement a fixed cap if needed.
            # For now, let's just keep all events (identity).
            num_vectors = num_events_after_denoise
            query_indices = np.arange(num_vectors)
        
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        # If no queries selected, return empty
        if len(query_indices) == 0:
            empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
            empty_coords = np.zeros((0, 5), dtype=np.float32)
            return empty_embeddings, empty_coords
        
        # ===================================================================
        # STAGE 3: VecKM ENCODING (Convert to Torch here)
        # ===================================================================
        # Extract event coordinates for the sampled vectors
        # Format: [x, y, t, p, segment_id]
        event_coords = np.zeros((num_vectors, 5), dtype=np.float32)
        event_coords[:, 0] = events_x_clean[query_indices]  # x
        event_coords[:, 1] = events_y_clean[query_indices]  # y
        event_coords[:, 2] = events_t_clean[query_indices]  # t
        event_coords[:, 3] = events_p_clean[query_indices]  # p
        event_coords[:, 4] = segment_id  # segment_id
        
        # NOW convert to tensors for VecKM encoding (single conversion)
        # VecKM uses all denoised events as context, but only queries the sampled ones
        t = torch.from_numpy(events_t_clean).float().to(self.device)
        y = torch.from_numpy(events_y_clean).float().to(self.device)
        x = torch.from_numpy(events_x_clean).float().to(self.device)
        
        # Query points (sampled events)
        query_t = t[query_indices]
        query_y = y[query_indices]
        query_x = x[query_indices]
        
        # Encode using VecKMSparse
        # The encoder returns complex embeddings of shape [num_queries, encoding_dim]
        with torch.no_grad():
            embeddings = self.encoder(t, y, x, query_y, query_x, query_t)
        
        return embeddings.cpu(), event_coords
    
    def encode_sample(self, sample: Dict, rotation_angle: int = 0) -> Dict:
        """
        Encode a single sample from DVSGesture dataset with optional rotation.
        
        Args:
            sample: Dictionary containing sliced events and metadata
            rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        
        Returns:
            Dictionary containing encoded tensors, event coordinates, and metadata
        """
        events_xy_sliced = sample['events_xy_sliced']
        events_t_sliced = sample['events_t_sliced']
        events_p_sliced = sample['events_p_sliced']
        
        # Apply rotation if needed
        if rotation_angle != 0:
            events_xy_sliced, events_t_sliced, events_p_sliced = rotate_sliced_events(
                events_xy_sliced,
                events_t_sliced,
                events_p_sliced,
                rotation_angle,
                self.height,
                self.width
            )
        
        num_intervals = len(events_xy_sliced)
        encoded_intervals = []
        event_coords_intervals = []
        num_vectors_per_interval = []
        
        for i in range(num_intervals):
            encoded, event_coords = self.encode_events_for_interval(
                events_xy_sliced[i],
                events_t_sliced[i],
                events_p_sliced[i],
                segment_id=i,  # Pass segment ID
            )
            encoded_intervals.append(encoded)
            event_coords_intervals.append(event_coords)
            num_vectors_per_interval.append(encoded.shape[0])
        
        return {
            'encoded_intervals': encoded_intervals,  # List of tensors
            'event_coords_intervals': event_coords_intervals,  # List of arrays [num_vectors, 5]
            'num_vectors_per_interval': num_vectors_per_interval,  # List of ints
            'num_intervals': num_intervals,
            'label': sample['label'],
            'file_path': sample['file_path'],
            'augmentation_method': sample['augmentation_method'],
            'rotation_angle': rotation_angle,
        }
    
    def preprocess_split(self, purpose: str):
        """
        Preprocess a dataset split (train or validation).
        
        Args:
            purpose: 'train' or 'validation'
        """
        print(f"\n{'='*60}")
        print(f"Pre-computing {purpose} split")
        print(f"{'='*60}")
        
        # Load checkpoint
        checkpoint_state = self.get_checkpoint_state(purpose)
        processed_samples = checkpoint_state[purpose]['processed_samples']
        
        # Create dataset (without augmentation for precomputing)
        dataset = DVSGesture(
            dataset_dir=self.dataset_dir,
            purpose=purpose,
            height=self.height,
            width=self.width,
            use_flip_augmentation=False,  # No augmentation during precomputing
            accumulation_interval_ms=self.accumulation_interval_ms,
        )
        
        total_samples = len(dataset)
        print(f"Total samples: {total_samples}")
        print(f"Already processed: {processed_samples}")
        
        if processed_samples >= total_samples:
            print(f"Split {purpose} already fully processed. Skipping.")
            return
        
        # Open HDF5 file
        h5_path = os.path.join(self.output_dir, f'{purpose}.h5')
        
        # Determine mode: create new or append
        if processed_samples == 0:
            mode = 'w'
            print(f"Creating new HDF5 file: {h5_path}")
        else:
            mode = 'a'
            print(f"Appending to existing HDF5 file: {h5_path}")
        
        with h5py.File(h5_path, mode) as h5f:
            # Create datasets if new file
            if processed_samples == 0:
                # Create resizable datasets
                h5f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int32)
                h5f.create_dataset('file_paths', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
                h5f.create_dataset('augmentation_methods', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
                h5f.create_dataset('num_intervals', shape=(0,), maxshape=(None,), dtype=np.int32)
                h5f.create_dataset('rotation_angles', shape=(0,), maxshape=(None,), dtype=np.int32)
                
                # Store metadata
                h5f.attrs['accumulation_interval_ms'] = self.accumulation_interval_ms
                # h5f.attrs['ratio_of_vectors'] = self.ratio_of_vectors # Removed
                h5f.attrs['encoding_dim'] = self.encoding_dim
                h5f.attrs['temporal_length'] = self.temporal_length
                h5f.attrs['height'] = self.height
                h5f.attrs['width'] = self.width
                h5f.attrs['rotation_enabled'] = self.rotation_enabled
                h5f.attrs['rotation_angles_list'] = self.rotation_angles if self.rotation_enabled else [0]
            
            # Determine rotation angles to use
            if purpose == 'train' or self.augment_validation:
                angles_to_use = self.rotation_angles if self.rotation_enabled else [0]
            else:
                # Validation/test: only use 0° rotation
                angles_to_use = [0]
            
            # Process samples
            pbar = tqdm(range(processed_samples, total_samples), desc=f"Processing {purpose}")
            
            sample_counter = 0  # Track actual HDF5 sample index
            for idx in pbar:
                # Load sample once
                sample = dataset[idx]
                
                # Generate rotated versions
                for rotation_angle in angles_to_use:
                    # Encode sample with rotation
                    encoded_sample = self.encode_sample(sample, rotation_angle=rotation_angle)
                    
                    # Create a group for this sample
                    sample_group = h5f.create_group(f'sample_{sample_counter:06d}')
                    
                    # Store encoded intervals and event coordinates
                    for interval_idx, (encoded, event_coords) in enumerate(
                        zip(encoded_sample['encoded_intervals'], encoded_sample['event_coords_intervals'])
                    ):
                        # Store real and imaginary parts separately
                        real_part = encoded.real.numpy()
                        imag_part = encoded.imag.numpy()
                        
                        interval_group = sample_group.create_group(f'interval_{interval_idx:03d}')
                        interval_group.create_dataset('real', data=real_part, compression='gzip')
                        interval_group.create_dataset('imag', data=imag_part, compression='gzip')
                        
                        # Store event coordinates [num_vectors, 5] with columns [x, y, t, p, segment_id]
                        interval_group.create_dataset('event_coords', data=event_coords, compression='gzip')
                    
                    # Store metadata arrays - resize and append
                    current_size = h5f['labels'].shape[0]
                    new_size = current_size + 1
                    
                    h5f['labels'].resize((new_size,))
                    h5f['labels'][current_size] = encoded_sample['label']
                    
                    h5f['file_paths'].resize((new_size,))
                    h5f['file_paths'][current_size] = encoded_sample['file_path']
                    
                    h5f['augmentation_methods'].resize((new_size,))
                    h5f['augmentation_methods'][current_size] = encoded_sample['augmentation_method']
                    
                    h5f['num_intervals'].resize((new_size,))
                    h5f['num_intervals'][current_size] = encoded_sample['num_intervals']
                    
                    h5f['rotation_angles'].resize((new_size,))
                    h5f['rotation_angles'][current_size] = rotation_angle
                    
                    # Store num_vectors_per_interval in sample group
                    sample_group.create_dataset(
                        'num_vectors_per_interval',
                        data=np.array(encoded_sample['num_vectors_per_interval'], dtype=np.int32)
                    )
                    
                    sample_counter += 1
                
                # Update checkpoint every N original samples
                if (idx + 1) % self.checkpoint_every_n == 0:
                    checkpoint_state[purpose]['processed_samples'] = idx + 1
                    self.save_checkpoint_state(checkpoint_state)
                    h5f.flush()
                    pbar.set_postfix({'checkpoint': f'{idx + 1}/{total_samples}', 'total_encoded': sample_counter})
            
            # Final checkpoint
            checkpoint_state[purpose]['processed_samples'] = total_samples
            self.save_checkpoint_state(checkpoint_state)
        
        print(f"\n{purpose} split preprocessing complete!")
        print(f"Output saved to: {h5_path}")
        
        # Print statistics
        self.print_statistics(h5_path)
    
    def print_statistics(self, h5_path: str):
        """Print statistics about the preprocessed dataset."""
        with h5py.File(h5_path, 'r') as h5f:
            num_samples = h5f['labels'].shape[0]
            num_intervals_total = h5f['num_intervals'][:].sum()
            num_intervals_mean = h5f['num_intervals'][:].mean()
            
            # Calculate total vectors
            total_vectors = 0
            for idx in range(num_samples):
                sample_group = h5f[f'sample_{idx:06d}']
                total_vectors += sample_group['num_vectors_per_interval'][:].sum()
            
            avg_vectors_per_sample = total_vectors / num_samples if num_samples > 0 else 0
            avg_vectors_per_interval = total_vectors / num_intervals_total if num_intervals_total > 0 else 0
            
            print(f"\nDataset Statistics:")
            print(f"  Total samples: {num_samples}")
            print(f"  Total intervals: {num_intervals_total}")
            print(f"  Average intervals per sample: {num_intervals_mean:.2f}")
            print(f"  Total vectors: {total_vectors}")
            print(f"  Average vectors per sample: {avg_vectors_per_sample:.2f}")
            print(f"  Average vectors per interval: {avg_vectors_per_interval:.2f}")
            print(f"  Encoding dimension: {h5f.attrs['encoding_dim']}")
            print(f"  File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
    
    def run(self):
        """Run preprocessing for all splits."""
        print("Starting DVSGesture dataset preprocessing...")
        print(f"Output directory: {self.output_dir}")
        print(f"Accumulation interval: {self.accumulation_interval_ms} ms")
        print(f"Encoding dimension: {self.encoding_dim}")
        
        # Preprocess train split
        self.preprocess_split('train')
        
        # Preprocess validation split
        self.preprocess_split('validation')
        
        print("\n" + "="*60)
        print("All preprocessing complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess DVSGesture dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Load config with OmegaConf to resolve interpolations
    config = OmegaConf.load(args.config)
    
    # Create preprocessor and run
    preprocessor = DVSGesturePreprocessor(config)
    preprocessor.run()


if __name__ == '__main__':
    main()
