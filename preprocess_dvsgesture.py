"""
Pre-computing script for DVSGesture dataset using SparseVKMEncoder.

This script:
1. Loads raw events from DVSGesture dataset
2. Encodes events into complex tensors using SparseVKMEncoder
3. Implements two-stage downsampling with ratio_of_vectors
4. Stores precomputed tensors in HDF5 format
5. Supports checkpointing for resume capability
"""

import os
import sys
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.DVSGesture import DVSGesture
from data.SparseVKMEncoder import VecKMSparse


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
        self.accumulation_interval_ms = precompute_cfg['accumulation_interval_ms']
        self.ratio_of_vectors = precompute_cfg['ratio_of_vectors']
        self.encoding_dim = precompute_cfg['encoding_dim']
        self.temporal_length = precompute_cfg['temporal_length']
        self.kernel_size = precompute_cfg['kernel_size']
        self.T_scale = precompute_cfg['T_scale']
        self.S_scale = precompute_cfg['S_scale']
        self.height = precompute_cfg['height']
        self.width = precompute_cfg['width']
        self.checkpoint_every_n = precompute_cfg['checkpoint_every_n_samples']
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.encoder = VecKMSparse(
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
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Encode events for a single time interval using SparseVKMEncoder.
        
        Args:
            events_xy: Event coordinates [N, 2]
            events_t: Event timestamps [N]
            events_p: Event polarities [N]
        
        Returns:
            Tuple of:
                - Complex tensor of shape [num_vectors, encoding_dim]
                - Event coordinates array [num_vectors, 4] with columns [x, y, t, p]
        """
        num_events = len(events_t)
        
        if num_events == 0:
            # Return empty tensors for empty intervals
            empty_embeddings = torch.zeros(0, self.encoding_dim, dtype=torch.cfloat)
            empty_coords = np.zeros((0, 4), dtype=np.float32)
            return empty_embeddings, empty_coords
        
        # Calculate number of vectors to sample based on ratio_of_vectors
        num_vectors = max(1, int(num_events * self.ratio_of_vectors))
        
        # Randomly sample indices for query vectors
        if num_vectors >= num_events:
            # Use all events as queries
            query_indices = np.arange(num_events)
        else:
            # Randomly sample without replacement
            query_indices = np.random.choice(num_events, num_vectors, replace=False)
            query_indices = np.sort(query_indices)  # Sort for better memory access
        
        # Extract event coordinates for the sampled vectors
        # Format: [x, y, t, p]
        event_coords = np.zeros((num_vectors, 4), dtype=np.float32)
        event_coords[:, 0] = events_xy[query_indices, 0]  # x
        event_coords[:, 1] = events_xy[query_indices, 1]  # y
        event_coords[:, 2] = events_t[query_indices]      # t
        event_coords[:, 3] = events_p[query_indices]      # p
        
        # Convert to tensors
        t = torch.from_numpy(events_t).float().to(self.device)
        y = torch.from_numpy(events_xy[:, 1]).float().to(self.device)
        x = torch.from_numpy(events_xy[:, 0]).float().to(self.device)
        
        # Query points (sampled events)
        query_t = t[query_indices]
        query_y = y[query_indices]
        query_x = x[query_indices]
        
        # Encode using VecKMSparse
        # The encoder returns complex embeddings of shape [num_queries, encoding_dim]
        with torch.no_grad():
            embeddings = self.encoder(t, y, x, query_y, query_x, query_t)
        
        return embeddings.cpu(), event_coords
    
    def encode_sample(self, sample: Dict) -> Dict:
        """
        Encode a single sample from DVSGesture dataset.
        
        Args:
            sample: Dictionary containing sliced events and metadata
        
        Returns:
            Dictionary containing encoded tensors, event coordinates, and metadata
        """
        events_xy_sliced = sample['events_xy_sliced']
        events_t_sliced = sample['events_t_sliced']
        events_p_sliced = sample['events_p_sliced']
        
        num_intervals = len(events_xy_sliced)
        encoded_intervals = []
        event_coords_intervals = []
        num_vectors_per_interval = []
        
        for i in range(num_intervals):
            encoded, event_coords = self.encode_events_for_interval(
                events_xy_sliced[i],
                events_t_sliced[i],
                events_p_sliced[i],
            )
            encoded_intervals.append(encoded)
            event_coords_intervals.append(event_coords)
            num_vectors_per_interval.append(encoded.shape[0])
        
        return {
            'encoded_intervals': encoded_intervals,  # List of tensors
            'event_coords_intervals': event_coords_intervals,  # List of arrays [num_vectors, 4]
            'num_vectors_per_interval': num_vectors_per_interval,  # List of ints
            'num_intervals': num_intervals,
            'label': sample['label'],
            'file_path': sample['file_path'],
            'augmentation_method': sample['augmentation_method'],
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
                
                # Store metadata
                h5f.attrs['accumulation_interval_ms'] = self.accumulation_interval_ms
                h5f.attrs['ratio_of_vectors'] = self.ratio_of_vectors
                h5f.attrs['encoding_dim'] = self.encoding_dim
                h5f.attrs['temporal_length'] = self.temporal_length
                h5f.attrs['height'] = self.height
                h5f.attrs['width'] = self.width
            
            # Process samples
            pbar = tqdm(range(processed_samples, total_samples), desc=f"Processing {purpose}")
            
            for idx in pbar:
                # Load and encode sample
                sample = dataset[idx]
                encoded_sample = self.encode_sample(sample)
                
                # Create a group for this sample
                sample_group = h5f.create_group(f'sample_{idx:06d}')
                
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
                    
                    # Store event coordinates [num_vectors, 4] with columns [x, y, t, p]
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
                
                # Store num_vectors_per_interval in sample group
                sample_group.create_dataset(
                    'num_vectors_per_interval',
                    data=np.array(encoded_sample['num_vectors_per_interval'], dtype=np.int32)
                )
                
                # Update checkpoint every N samples
                if (idx + 1) % self.checkpoint_every_n == 0:
                    checkpoint_state[purpose]['processed_samples'] = idx + 1
                    self.save_checkpoint_state(checkpoint_state)
                    h5f.flush()
                    pbar.set_postfix({'checkpoint': f'{idx + 1}/{total_samples}'})
            
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
        print(f"Ratio of vectors (first stage): {self.ratio_of_vectors}")
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
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create preprocessor and run
    preprocessor = DVSGesturePreprocessor(config)
    preprocessor.run()


if __name__ == '__main__':
    main()
