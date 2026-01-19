"""
Preprocess Denoised Events Only

This script applies denoising (from config) and saves the denoised events to disk.
This creates intermediate data for the sampling benchmark to use.

Workflow:
1. Run benchmark_denoising.py to find optimal parameters
2. Update config.yaml with recommended denoising parameters
3. Run this script to save denoised events
4. Run benchmark_sampling.py on the denoised events

Usage:
    python preprocess_denoise_only.py --config configs/config.yaml
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataset_class
from utils.denoising_and_sampling import filter_noise_spatial


class DenoiseOnlyPreprocessor:
    """
    Preprocessor that applies ONLY denoising and saves results.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load settings from config
        precompute_cfg = config['PRECOMPUTING']
        benchmark_cfg = config['BENCHMARKING']
        
        self.dataset_dir = precompute_cfg['dataset_dir']
        self.height = int(precompute_cfg['height'])
        self.width = int(precompute_cfg['width'])
        self.accumulation_interval_ms = float(precompute_cfg['accumulation_interval_ms'])
        
        # Denoising settings (from config)
        denoising_cfg = OmegaConf.select(precompute_cfg, 'denoising', default={})
        self.denoising_enabled = OmegaConf.select(denoising_cfg, 'enabled', default=True)
        self.denoising_grid_size = int(OmegaConf.select(denoising_cfg, 'grid_size', default=4))
        self.denoising_threshold = int(OmegaConf.select(denoising_cfg, 'threshold', default=2))
        
        # Output directory (from benchmark config)
        self.output_dir = Path(benchmark_cfg['denoised_cache_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Denoising-Only Preprocessing")
        print(f"  Denoising: {'enabled' if self.denoising_enabled else 'disabled'}")
        if self.denoising_enabled:
            print(f"    Grid size: {self.denoising_grid_size}")
            print(f"    Threshold: {self.denoising_threshold}")
        print(f"  Output: {self.output_dir}")
    
    def denoise_interval(
        self,
        events_xy: np.ndarray,
        events_t: np.ndarray,
        events_p: np.ndarray
    ):
        """
        Apply denoising to a single interval.
        
        Returns:
            Denoised events (t, y, x, p) as numpy arrays
        """
        if len(events_t) == 0:
            return events_t, events_xy[:, 1], events_xy[:, 0], events_p
        
        if not self.denoising_enabled:
            return events_t, events_xy[:, 1], events_xy[:, 0], events_p
        
        # Apply denoising
        events_y = events_xy[:, 1]
        events_x = events_xy[:, 0]
        
        t_clean, y_clean, x_clean, p_clean = filter_noise_spatial(
            events_t, events_y, events_x, events_p,
            self.height, self.width,
            self.denoising_grid_size,
            self.denoising_threshold
        )
        
        return t_clean, y_clean, x_clean, p_clean
    
    def process_split(self, purpose: str):
        """
        Process a dataset split and save denoised events.
        
        Args:
            purpose: 'train' or 'validation'
        """
        print(f"\n{'='*60}")
        print(f"Processing {purpose} split")
        print(f"{'='*60}")
        
        # Load dataset
        # Load dataset
        dataset_cls = get_dataset_class(self.config.get('dataset', 'dvsgesture')) # or infer from config
        
        dataset = dataset_cls(
            dataset_dir=self.dataset_dir,
            purpose=purpose,
            height=self.height,
            width=self.width,
            use_flip_augmentation=False,
            accumulation_interval_ms=self.accumulation_interval_ms
        )
        
        total_samples = len(dataset)
        print(f"Total samples: {total_samples}")
        
        # Open HDF5 file for output
        h5_path = self.output_dir / f'{purpose}.h5'
        
        with h5py.File(h5_path, 'w') as h5f:
            # Store metadata
            h5f.attrs['height'] = self.height
            h5f.attrs['width'] = self.width
            h5f.attrs['accumulation_interval_ms'] = self.accumulation_interval_ms
            h5f.attrs['denoising_enabled'] = self.denoising_enabled
            h5f.attrs['denoising_grid_size'] = self.denoising_grid_size
            h5f.attrs['denoising_threshold'] = self.denoising_threshold
            h5f.attrs['num_samples'] = total_samples
            
            # Process each sample
            pbar = tqdm(range(total_samples), desc=f"Processing {purpose}")
            
            for sample_idx in pbar:
                sample = dataset[sample_idx]
                
                # Create group for this sample
                sample_group = h5f.create_group(f'sample_{sample_idx:06d}')
                
                # Store label
                sample_group.attrs['label'] = sample['label']
                sample_group.attrs['file_path'] = sample['file_path']
                
                # Process each interval
                num_intervals = len(sample['events_t_sliced'])
                sample_group.attrs['num_intervals'] = num_intervals
                
                for interval_idx in range(num_intervals):
                    events_xy = sample['events_xy_sliced'][interval_idx]
                    events_t = sample['events_t_sliced'][interval_idx]
                    events_p = sample['events_p_sliced'][interval_idx]
                    
                    # Denoise
                    t_clean, y_clean, x_clean, p_clean = self.denoise_interval(
                        events_xy, events_t, events_p
                    )
                    
                    # Save denoised events
                    interval_group = sample_group.create_group(f'interval_{interval_idx:03d}')
                    interval_group.create_dataset('t', data=t_clean, compression='gzip')
                    interval_group.create_dataset('y', data=y_clean, compression='gzip')
                    interval_group.create_dataset('x', data=x_clean, compression='gzip')
                    interval_group.create_dataset('p', data=p_clean, compression='gzip')
                    interval_group.attrs['num_events'] = len(t_clean)
        
        print(f"\n{purpose} split complete!")
        print(f"Output saved to: {h5_path}")
        
        # Print statistics
        self.print_statistics(h5_path)
    
    def print_statistics(self, h5_path: Path):
        """Print statistics about the denoised dataset."""
        with h5py.File(h5_path, 'r') as h5f:
            num_samples = h5f.attrs['num_samples']
            
            total_events_original = 0
            total_events_denoised = 0
            
            for sample_idx in range(num_samples):
                sample_group = h5f[f'sample_{sample_idx:06d}']
                num_intervals = sample_group.attrs['num_intervals']
                
                for interval_idx in range(num_intervals):
                    interval_group = sample_group[f'interval_{interval_idx:03d}']
                    total_events_denoised += interval_group.attrs['num_events']
            
            print(f"\nDataset Statistics:")
            print(f"  Total samples: {num_samples}")
            print(f"  Total denoised events: {total_events_denoised}")
            print(f"  File size: {os.path.getsize(h5_path) / (1024**3):.2f} GB")
    
    def run(self):
        """Run preprocessing for all splits."""
        print("Starting denoising-only preprocessing...")
        print(f"Output directory: {self.output_dir}")
        
        # Process train split
        self.process_split('train')
        
        # Process validation split
        self.process_split('validation')
        
        print("\n" + "="*60)
        print("All preprocessing complete!")
        print("="*60)
        print(f"\nDenoised events saved to: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Run: python benchmark_sampling.py --config configs/config.yaml")
        print("  2. Update config.yaml with recommended sampling parameters")
        print("  3. Run: python preprocess_dvsgesture.py --config configs/config.yaml")


def main():
    parser = argparse.ArgumentParser(description='Preprocess denoised events only')
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
    preprocessor = DenoiseOnlyPreprocessor(config)
    preprocessor.run()


if __name__ == '__main__':
    main()
