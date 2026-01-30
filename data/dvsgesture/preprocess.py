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
from utils.denoising_and_sampling import (
    filter_noise_spatial,
    filter_noise_spatial_temporal,
    filter_background_activity
)


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
        self.denoising_method = OmegaConf.select(denoising_cfg, 'method', default='spatial') # Default to spatial
        
        # Method 1: Spatial
        self.denoising_grid_size = int(OmegaConf.select(denoising_cfg, 'grid_size', default=4))
        self.denoising_threshold = int(OmegaConf.select(denoising_cfg, 'threshold', default=2))
        
        # Method 2: Spatial-Temporal
        st_cfg = OmegaConf.select(denoising_cfg, 'spatial_temporal', default={})
        self.st_time_window = int(OmegaConf.select(st_cfg, 'time_window', default=10000))
        self.st_threshold = int(OmegaConf.select(st_cfg, 'threshold', default=2))
        
        # Method 3: BAF
        baf_cfg = OmegaConf.select(denoising_cfg, 'baf', default={})
        self.baf_time_threshold = int(OmegaConf.select(baf_cfg, 'time_threshold', default=1000))

        
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
            print(f"  Method: {self.denoising_method}")
            if self.denoising_method == 'spatial':
                print(f"  Grid size: {self.denoising_grid_size}, Threshold: {self.denoising_threshold}")
            elif self.denoising_method == 'spatial_temporal':
                 print(f"  Time Window: {self.st_time_window}, Threshold: {self.st_threshold}")
            elif self.denoising_method == 'baf':
                 print(f"  Time Threshold: {self.baf_time_threshold}")
                 
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
    
    def _apply_denoising(self, t: np.ndarray, y: np.ndarray, x: np.ndarray, p: np.ndarray):
        """Apply the configured denoising method."""
        if self.denoising_method == 'spatial':
            return filter_noise_spatial(
                t, y, x, p,
                self.height, self.width,
                self.denoising_grid_size,
                self.denoising_threshold
            )
        elif self.denoising_method == 'spatial_temporal':
             return filter_noise_spatial_temporal(
                t, y, x, p,
                self.height, self.width,
                grid_size=self.denoising_grid_size,
                time_window_us=self.st_time_window,
                threshold=self.st_threshold
            )
        elif self.denoising_method == 'baf':
             return filter_background_activity(
                t, y, x, p,
                self.height, self.width,
                time_threshold=self.baf_time_threshold
            )
        return t, y, x, p

    def encode_events_for_interval(
        self,
        events_xy: np.ndarray,
        events_t: np.ndarray,
        events_p: np.ndarray,
        segment_id: int,
        apply_denoising: bool = True
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Encode events for a single time interval.
        
        Args:
            events_xy: Event coordinates [N, 2]
            events_t: Event timestamps [N]
            events_p: Event polarities [N]
            segment_id: Segment index
            apply_denoising: Whether to apply denoising (set False if already denoised)
        """
        num_events = len(events_t)
        
        if num_events == 0:
            empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
            empty_coords = np.zeros((0, 5), dtype=np.float32)
            return empty_embeddings, empty_coords
        
        # Extract separate x, y coordinates
        events_y = events_xy[:, 1]
        events_x = events_xy[:, 0]
        
        # BOUNDS CHECKING
        events_x = np.clip(events_x, 0, self.width - 1)
        events_y = np.clip(events_y, 0, self.height - 1)
        
        # STAGE 1: DENOISING (Optional)
        if self.denoising_enabled and apply_denoising:
            events_t_clean, events_y_clean, events_x_clean, events_p_clean = self._apply_denoising(
                events_t, events_y, events_x, events_p
            )
            
            num_events_after_denoise = len(events_t_clean)
            if num_events_after_denoise == 0:
                empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
                empty_coords = np.zeros((0, 5), dtype=np.float32)
                return empty_embeddings, empty_coords
        else:
            # Skip denoising (already done or disabled)
            events_t_clean = events_t
            events_y_clean = events_y
            events_x_clean = events_x
            events_p_clean = events_p
            num_events_after_denoise = num_events
        
        # STAGE 2: SAMPLING
        from utils.density_adaptive_spatial_striding import adaptive_spatial_sampling
        
        if self.sampling_method == 'adaptive_striding':
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
            num_vectors = num_events_after_denoise
            query_indices = np.arange(num_vectors)
        
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        if len(query_indices) == 0:
            empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
            empty_coords = np.zeros((0, 5), dtype=np.float32)
            return empty_embeddings, empty_coords
        
        # STAGE 3: VecKM ENCODING
        event_coords = np.zeros((num_vectors, 5), dtype=np.float32)
        event_coords[:, 0] = events_x_clean[query_indices]
        event_coords[:, 1] = events_y_clean[query_indices]
        event_coords[:, 2] = events_t_clean[query_indices]
        event_coords[:, 3] = events_p_clean[query_indices]
        event_coords[:, 4] = segment_id
        
        t = torch.from_numpy(events_t_clean).float().to(self.device)
        y = torch.from_numpy(events_y_clean).float().to(self.device)
        x = torch.from_numpy(events_x_clean).float().to(self.device)
        
        query_t = t[query_indices]
        query_y = y[query_indices]
        query_x = x[query_indices]
        
        with torch.no_grad():
            embeddings = self.encoder(t, y, x, query_y, query_x, query_t)
        
        return embeddings.cpu(), event_coords
    
    def encode_sample(self, sample: Dict, rotation_angle: int = 0) -> Dict:
        """
        Encode a single sample.
        OPTIMIZATION: Denoising is now handled BEFORE rotation to avoid redundancy.
        BUT: We can't easily denoise once IF the rotation is applied in the loop. 
        Actually, we can: Denoise -> Rotate -> Encode(skip_denoise=True)
        """
        events_xy_sliced = sample['events_xy_sliced']
        events_t_sliced = sample['events_t_sliced']
        events_p_sliced = sample['events_p_sliced']
        
        # 1. OPTIMIZATION: One-time Denoising (if enabled)
        # We process the raw slices first to remove noise
        if self.denoising_enabled:
             denoised_xy = []
             denoised_t = []
             denoised_p = []
             
             for i in range(len(events_xy_sliced)):
                 # Apply denoising on original coordinates
                 t_c, y_c, x_c, p_c = self._apply_denoising(
                     events_t_sliced[i], 
                     events_xy_sliced[i][:, 1], 
                     events_xy_sliced[i][:, 0], 
                     events_p_sliced[i]
                 )
                 # Reconstruct xy array
                 if len(t_c) > 0:
                     xy_c = np.stack([x_c, y_c], axis=1)
                     denoised_xy.append(xy_c)
                     denoised_t.append(t_c)
                     denoised_p.append(p_c)
                 else:
                     # Keep empty if filtered out
                     denoised_xy.append(np.zeros((0, 2), dtype=np.float32))
                     denoised_t.append(np.zeros(0, dtype=np.float32))
                     denoised_p.append(np.zeros(0, dtype=np.float32))
             
             # Replace original lists with denoised ones
             events_xy_sliced = denoised_xy
             events_t_sliced = denoised_t
             events_p_sliced = denoised_p
        
        # 2. Rotation (on already denoised data)
        if rotation_angle != 0:
            events_xy_sliced, events_t_sliced, events_p_sliced = rotate_sliced_events(
                events_xy_sliced,
                events_t_sliced,
                events_p_sliced,
                rotation_angle,
                self.height,
                self.width
            )
        
        # 3. Encoding (with skip_denoising=True because we just did it)
        num_intervals = len(events_xy_sliced)
        encoded_intervals = []
        event_coords_intervals = []
        num_vectors_per_interval = []
        
        for i in range(num_intervals):
            encoded, event_coords = self.encode_events_for_interval(
                events_xy_sliced[i],
                events_t_sliced[i],
                events_p_sliced[i],
                segment_id=i, 
                apply_denoising=False  # CRITICAL: Do not denoise again
            )
            encoded_intervals.append(encoded)
            event_coords_intervals.append(event_coords)
            num_vectors_per_interval.append(encoded.shape[0])
        
        return {
            'encoded_intervals': encoded_intervals,
            'event_coords_intervals': event_coords_intervals,
            'num_vectors_per_interval': num_vectors_per_interval,
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
                try:
                    sample = dataset[idx]
                except Exception as e:
                    print(f"Skipping sample {idx} due to error: {e}")
                    continue
                
                # --- OPTIMIZATION START ---
                # Pre-denoise locally if enabled, so we don't repeat for every rotation
                # Logic: We create a 'clean_sample' object
                clean_sample = sample.copy() # Shallow copy
                
                if self.denoising_enabled:
                     denoised_xy = []
                     denoised_t = []
                     denoised_p = []
                     
                     for i in range(len(sample['events_xy_sliced'])):
                         t_c, y_c, x_c, p_c = filter_noise_spatial(
                             sample['events_t_sliced'][i],
                             sample['events_xy_sliced'][i][:, 1],
                             sample['events_xy_sliced'][i][:, 0],
                             sample['events_p_sliced'][i],
                             self.height, self.width,
                             self.denoising_grid_size,
                             self.denoising_threshold
                         )
                         if len(t_c) > 0:
                             denoised_xy.append(np.stack([x_c, y_c], axis=1))
                             denoised_t.append(t_c)
                             denoised_p.append(p_c)
                         else:
                             denoised_xy.append(np.zeros((0, 2), dtype=np.float32))
                             denoised_t.append(np.zeros(0, dtype=np.float32))
                             denoised_p.append(np.zeros(0, dtype=np.float32))
                     
                     clean_sample['events_xy_sliced'] = denoised_xy
                     clean_sample['events_t_sliced'] = denoised_t
                     clean_sample['events_p_sliced'] = denoised_p
                # --- OPTIMIZATION END ---
                
                # Generate rotated versions from the CLEAN sample
                for rotation_angle in angles_to_use:
                    # Encode sample with rotation (SKIP internal denoising)
                    # We pass the already clean sample, and tell encode_sample to trust us
                    # (Note: I actually modified encode_sample to handle this flow implicitly by re-checking denoising 
                    # there, but to be safe and cleaner, let's USE the clean_sample and ensure encode_sample 
                    # doesn't redo it.
                    
                    # Wait, my previous edit to `encode_sample` ALREADY incorporated the denoising step at the top.
                    # So actually, I can just call `encode_sample` normally, but `encode_sample` itself was refactored
                    # to do the denoising FIRST. 
                    # However, calling `encode_sample` inside the loop means we denoise N times!
                    # I need to fix `encode_sample` to accept PRE-DENOISED data or just do the logic here.
                    
                    # Let's revert to calling a lower-level function or just doing the rotation here manually
                    # to fully realize the optimization.
                    
                    # 1. Rotate the CLEAN events
                    r_xy, r_t, r_p = clean_sample['events_xy_sliced'], clean_sample['events_t_sliced'], clean_sample['events_p_sliced']
                    
                    if rotation_angle != 0:
                        r_xy, r_t, r_p = rotate_sliced_events(r_xy, r_t, r_p, rotation_angle, self.height, self.width)
                    
                    # 2. Loop Intervals and Encode (Explicitly SKIP denoising)
                    encoded_intervals = []
                    event_coords_intervals = []
                    num_vectors_per_interval = []
                    
                    for i in range(len(r_xy)):
                         # Call with apply_denoising=False because clean_sample is already denoised
                         encoded, event_coords = self.encode_events_for_interval(
                            r_xy[i], r_t[i], r_p[i], segment_id=i, apply_denoising=False
                         )
                         encoded_intervals.append(encoded)
                         event_coords_intervals.append(event_coords)
                         num_vectors_per_interval.append(encoded.shape[0])
                    
                    # Create a group for this sample
                    sample_group = h5f.create_group(f'sample_{sample_counter:06d}')
                    
                    # Store encoded intervals and event coordinates
                    for interval_idx, (encoded, event_coords) in enumerate(zip(encoded_intervals, event_coords_intervals)):
                        real_part = encoded.real.numpy()
                        imag_part = encoded.imag.numpy()
                        interval_group = sample_group.create_group(f'interval_{interval_idx:03d}')
                        interval_group.create_dataset('real', data=real_part, compression='gzip')
                        interval_group.create_dataset('imag', data=imag_part, compression='gzip')
                        interval_group.create_dataset('event_coords', data=event_coords, compression='gzip')
                    
                    # Store metadata arrays - resize and append
                    current_size = h5f['labels'].shape[0]
                    new_size = current_size + 1
                    
                    h5f['labels'].resize((new_size,))
                    h5f['labels'][current_size] = sample['label']
                    
                    h5f['file_paths'].resize((new_size,))
                    h5f['file_paths'][current_size] = sample['file_path']
                    
                    h5f['augmentation_methods'].resize((new_size,))
                    h5f['augmentation_methods'][current_size] = sample['augmentation_method']
                    
                    h5f['num_intervals'].resize((new_size,))
                    h5f['num_intervals'][current_size] = len(encoded_intervals)
                    
                    h5f['rotation_angles'].resize((new_size,))
                    h5f['rotation_angles'][current_size] = rotation_angle
                    
                    sample_group.create_dataset(
                        'num_vectors_per_interval',
                        data=np.array(num_vectors_per_interval, dtype=np.int32)
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
