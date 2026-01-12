"""
Dry-Run Preprocessing Statistics

This script simulates the preprocessing pipeline (denoising + sampling) without
VecKM encoding to quickly estimate dataset statistics.

Purpose:
- Understand dataset statistics before expensive preprocessing
- Estimate memory requirements
- Validate denoising/sampling parameter choices

Usage:
    python data/dry_run_preprocessing.py --config configs/config.yaml --num_samples 100
"""

import os
import sys
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.DVSGesture import DVSGesture
from utils.denoising_and_sampling import filter_noise_spatial, sample_grid_decimation_fast


class DryRunPreprocessor:
    """Simulate preprocessing pipeline for statistics estimation."""
    
    def __init__(self, config: Dict):
        self.config = config
        precompute_cfg = config['PRECOMPUTING']
        
        # Dataset parameters
        self.dataset_dir = precompute_cfg['dataset_dir']
        self.height = int(precompute_cfg['height'])
        self.width = int(precompute_cfg['width'])
        self.accumulation_interval_ms = float(precompute_cfg['accumulation_interval_ms'])
        self.encoding_dim = int(precompute_cfg['encoding_dim'])
        self.ratio_of_vectors = float(precompute_cfg['ratio_of_vectors'])
        
        # Denoising config
        denoising_cfg = OmegaConf.select(precompute_cfg, 'denoising', default={})
        self.denoising_enabled = OmegaConf.select(denoising_cfg, 'enabled', default=False)
        self.denoising_grid_size = int(OmegaConf.select(denoising_cfg, 'grid_size', default=4))
        self.denoising_threshold = int(OmegaConf.select(denoising_cfg, 'threshold', default=2))
        
        # Sampling config
        sampling_cfg = OmegaConf.select(precompute_cfg, 'sampling', default={})
        self.sampling_method = OmegaConf.select(sampling_cfg, 'method', default='random')
        self.sampling_grid_size = int(OmegaConf.select(
            OmegaConf.select(sampling_cfg, 'grid_decimation', default={}),
            'grid_size',
            default=4
        ))
        
        print("Dry-Run Preprocessor Initialized")
        print(f"  Denoising: {'enabled' if self.denoising_enabled else 'disabled'}")
        if self.denoising_enabled:
            print(f"    Grid size: {self.denoising_grid_size}, Threshold: {self.denoising_threshold}")
        print(f"  Sampling: {self.sampling_method}")
        if self.sampling_method == 'grid_decimation':
            print(f"    Grid size: {self.sampling_grid_size}, Retention: {self.ratio_of_vectors}")
    
    def process_interval(
        self,
        events_xy: np.ndarray,
        events_t: np.ndarray,
        events_p: np.ndarray,
    ) -> Dict:
        """Process a single interval and return statistics."""
        num_raw = len(events_t)
        
        if num_raw == 0:
            return {
                'num_raw': 0,
                'num_denoised': 0,
                'num_sampled': 0,
                'denoising_retention': 0.0,
                'sampling_retention': 0.0,
                'total_retention': 0.0,
            }
        
        # Denoising
        if self.denoising_enabled:
            events_y = events_xy[:, 1]
            events_x = events_xy[:, 0]
            
            t_clean, y_clean, x_clean, p_clean = filter_noise_spatial(
                events_t, events_y, events_x, events_p,
                self.height, self.width,
                self.denoising_grid_size,
                self.denoising_threshold
            )
            num_denoised = len(t_clean)
        else:
            t_clean = events_t
            y_clean = events_xy[:, 1]
            x_clean = events_xy[:, 0]
            p_clean = events_p
            num_denoised = num_raw
        
        if num_denoised == 0:
            return {
                'num_raw': num_raw,
                'num_denoised': 0,
                'num_sampled': 0,
                'denoising_retention': 0.0,
                'sampling_retention': 0.0,
                'total_retention': 0.0,
            }
        
        # Sampling
        if self.sampling_method == 'grid_decimation':
            sampled_indices = sample_grid_decimation_fast(
                t_clean, y_clean, x_clean, p_clean,
                self.height, self.width,
                target_grid=self.sampling_grid_size,
                retention_ratio=self.ratio_of_vectors,
                return_indices=True
            )
            num_sampled = len(sampled_indices)
        else:  # random
            num_vectors = max(1, int(num_denoised * self.ratio_of_vectors))
            num_sampled = min(num_vectors, num_denoised)
        
        return {
            'num_raw': num_raw,
            'num_denoised': num_denoised,
            'num_sampled': num_sampled,
            'denoising_retention': num_denoised / num_raw if num_raw > 0 else 0.0,
            'sampling_retention': num_sampled / num_denoised if num_denoised > 0 else 0.0,
            'total_retention': num_sampled / num_raw if num_raw > 0 else 0.0,
        }
    
    def run_dry_run(self, purpose: str, num_samples: int = None) -> Dict:
        """Run dry-run on dataset split."""
        print(f"\n{'='*60}")
        print(f"Dry-Run: {purpose} split")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = DVSGesture(
            dataset_dir=self.dataset_dir,
            purpose=purpose,
            height=self.height,
            width=self.width,
            use_flip_augmentation=False,
            accumulation_interval_ms=self.accumulation_interval_ms,
        )
        
        total_samples = len(dataset)
        if num_samples is not None:
            total_samples = min(num_samples, total_samples)
        
        print(f"Processing {total_samples} samples...")
        
        # Statistics accumulators
        stats = {
            'total_raw_events': 0,
            'total_denoised_events': 0,
            'total_sampled_vectors': 0,
            'total_intervals': 0,
            'vectors_per_sample': [],
            'vectors_per_interval': [],
            'denoising_retentions': [],
            'sampling_retentions': [],
            'total_retentions': [],
        }
        
        for sample_idx in tqdm(range(total_samples), desc=f"{purpose}"):
            sample = dataset[sample_idx]
            
            sample_vectors = 0
            for interval_idx in range(len(sample['events_t_sliced'])):
                interval_stats = self.process_interval(
                    sample['events_xy_sliced'][interval_idx],
                    sample['events_t_sliced'][interval_idx],
                    sample['events_p_sliced'][interval_idx],
                )
                
                stats['total_raw_events'] += interval_stats['num_raw']
                stats['total_denoised_events'] += interval_stats['num_denoised']
                stats['total_sampled_vectors'] += interval_stats['num_sampled']
                stats['total_intervals'] += 1
                stats['vectors_per_interval'].append(interval_stats['num_sampled'])
                stats['denoising_retentions'].append(interval_stats['denoising_retention'])
                stats['sampling_retentions'].append(interval_stats['sampling_retention'])
                stats['total_retentions'].append(interval_stats['total_retention'])
                
                sample_vectors += interval_stats['num_sampled']
            
            stats['vectors_per_sample'].append(sample_vectors)
        
        # Compute aggregates
        stats['avg_vectors_per_sample'] = np.mean(stats['vectors_per_sample'])
        stats['avg_vectors_per_interval'] = np.mean(stats['vectors_per_interval'])
        stats['avg_denoising_retention'] = np.mean(stats['denoising_retentions'])
        stats['avg_sampling_retention'] = np.mean(stats['sampling_retentions'])
        stats['avg_total_retention'] = np.mean(stats['total_retentions'])
        
        # Estimate memory (complex64 = 8 bytes per value, 64 dims = 512 bytes per vector)
        bytes_per_vector = self.encoding_dim * 8 * 2  # complex64
        stats['estimated_memory_mb'] = (stats['total_sampled_vectors'] * bytes_per_vector) / (1024**2)
        
        return stats
    
    def print_report(self, train_stats: Dict, val_stats: Dict):
        """Print detailed statistics report."""
        print(f"\n{'='*60}")
        print("DRY-RUN PREPROCESSING STATISTICS REPORT")
        print(f"{'='*60}\n")
        
        print("CONFIGURATION:")
        print(f"  Encoding dim: {self.encoding_dim}")
        print(f"  Denoising: {'enabled' if self.denoising_enabled else 'disabled'}")
        if self.denoising_enabled:
            print(f"    Grid: {self.denoising_grid_size}, Threshold: {self.denoising_threshold}")
        print(f"  Sampling: {self.sampling_method}")
        if self.sampling_method == 'grid_decimation':
            print(f"    Grid: {self.sampling_grid_size}, Retention: {self.ratio_of_vectors}")
        
        for split_name, stats in [("TRAIN", train_stats), ("VALIDATION", val_stats)]:
            print(f"\n{split_name} SPLIT:")
            print(f"  Total intervals: {stats['total_intervals']:,}")
            print(f"  Total raw events: {stats['total_raw_events']:,}")
            print(f"  Total denoised events: {stats['total_denoised_events']:,}")
            print(f"  Total sampled vectors: {stats['total_sampled_vectors']:,}")
            print(f"\n  Average vectors per sample: {stats['avg_vectors_per_sample']:.1f}")
            print(f"  Average vectors per interval: {stats['avg_vectors_per_interval']:.1f}")
            print(f"\n  Denoising retention: {stats['avg_denoising_retention']*100:.2f}%")
            print(f"  Sampling retention: {stats['avg_sampling_retention']*100:.2f}%")
            print(f"  Total retention: {stats['avg_total_retention']*100:.2f}%")
            print(f"\n  Estimated memory: {stats['estimated_memory_mb']:.2f} MB")
        
        total_memory = train_stats['estimated_memory_mb'] + val_stats['estimated_memory_mb']
        print(f"\n{'='*60}")
        print(f"TOTAL ESTIMATED MEMORY: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
        print(f"{'='*60}\n")
    
    def save_report(self, train_stats: Dict, val_stats: Dict, output_dir: Path):
        """Save report to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dry_run_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DRY-RUN PREPROCESSING STATISTICS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Encoding dim: {self.encoding_dim}\n")
            f.write(f"  Denoising: {'enabled' if self.denoising_enabled else 'disabled'}\n")
            if self.denoising_enabled:
                f.write(f"    Grid: {self.denoising_grid_size}, Threshold: {self.denoising_threshold}\n")
            f.write(f"  Sampling: {self.sampling_method}\n")
            if self.sampling_method == 'grid_decimation':
                f.write(f"    Grid: {self.sampling_grid_size}, Retention: {self.ratio_of_vectors}\n")
            
            for split_name, stats in [("TRAIN", train_stats), ("VALIDATION", val_stats)]:
                f.write(f"\n{split_name} SPLIT:\n")
                f.write(f"  Total intervals: {stats['total_intervals']:,}\n")
                f.write(f"  Total raw events: {stats['total_raw_events']:,}\n")
                f.write(f"  Total denoised events: {stats['total_denoised_events']:,}\n")
                f.write(f"  Total sampled vectors: {stats['total_sampled_vectors']:,}\n")
                f.write(f"\n  Average vectors per sample: {stats['avg_vectors_per_sample']:.1f}\n")
                f.write(f"  Average vectors per interval: {stats['avg_vectors_per_interval']:.1f}\n")
                f.write(f"\n  Denoising retention: {stats['avg_denoising_retention']*100:.2f}%\n")
                f.write(f"  Sampling retention: {stats['avg_sampling_retention']*100:.2f}%\n")
                f.write(f"  Total retention: {stats['avg_total_retention']*100:.2f}%\n")
                f.write(f"\n  Estimated memory: {stats['estimated_memory_mb']:.2f} MB\n")
            
            total_memory = train_stats['estimated_memory_mb'] + val_stats['estimated_memory_mb']
            f.write(f"\n{'='*60}\n")
            f.write(f"TOTAL ESTIMATED MEMORY: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)\n")
            f.write(f"{'='*60}\n")
        
        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Dry-run preprocessing statistics')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process per split (default: all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save report (default: precomputed_data/dry_run)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create preprocessor
    preprocessor = DryRunPreprocessor(config)
    
    # Run dry-run
    train_stats = preprocessor.run_dry_run('train', args.num_samples)
    val_stats = preprocessor.run_dry_run('validation', args.num_samples)
    
    # Print report
    preprocessor.print_report(train_stats, val_stats)
    
    # Save report
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['PRECOMPUTING']['output_dir']).parent / 'dry_run'
    
    preprocessor.save_report(train_stats, val_stats, output_dir)


if __name__ == '__main__':
    main()
