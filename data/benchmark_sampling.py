"""
Automated Sampling Parameter Benchmark

This script implements three quantitative evaluation metrics for sampling:
1. Active Area Coverage (AAC) - Completeness score
2. Redundancy Factor (RF) - Efficiency score  
3. Density Alignment Score (DAS) - Focus score

The benchmark automatically searches for optimal sampling parameters for each dataset.

Usage:
    python benchmark_sampling.py --dataset dvsgesture --config configs/config.yaml
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import h5py
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.DVSGesture import DVSGesture
from utils.denoising_and_sampling import sample_grid_decimation_fast


class SamplingBenchmark:
    """
    Automated benchmark for finding optimal sampling parameters.
    """
    
    def __init__(self, config: Dict, dataset_name: str = 'dvsgesture'):
        self.config = config
        self.dataset_name = dataset_name
        
        # Load dataset parameters
        precompute_cfg = config['PRECOMPUTING']
        benchmark_cfg = config['BENCHMARKING']
        
        self.dataset_dir = precompute_cfg['dataset_dir']
        self.height = precompute_cfg['height']
        self.width = precompute_cfg['width']
        self.accumulation_interval_ms = precompute_cfg['accumulation_interval_ms']
        self.kernel_size = precompute_cfg['kernel_size']  # VecKM kernel for overlap
        
        # Denoised cache directory (where preprocess_denoise_only.py saves data)
        self.denoised_cache_dir = Path(benchmark_cfg['denoised_cache_dir'])
        
        # Kernel radius for VecKM receptive field
        self.kernel_radius = self.kernel_size // 2
        
        # Benchmark output directory
        self.output_dir = Path(config['PRECOMPUTING']['output_dir']).parent / 'sampling_benchmark'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Sampling Benchmark Initialized")
        print(f"Dataset: {dataset_name}")
        print(f"VecKM kernel size: {self.kernel_size} (radius: {self.kernel_radius})")
        print(f"Loading denoised events from: {self.denoised_cache_dir}")
        print(f"Output: {self.output_dir}")
    
    def create_kernel_mask(self, radius: int) -> np.ndarray:
        """Create a circular kernel mask for the VecKM receptive field."""
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = (x**2 + y**2 <= radius**2).astype(np.float32)
        return mask
    
    def compute_aac_score(
        self,
        events_y_raw: np.ndarray,
        events_x_raw: np.ndarray,
        events_y_sampled: np.ndarray,
        events_x_sampled: np.ndarray
    ) -> float:
        """
        Metric 1: Active Area Coverage (AAC) - Completeness
        
        Measures what percentage of the ground truth structure is covered
        by at least one sampled kernel.
        
        Goal: Maximize (Target > 95%)
        
        Args:
            events_*_raw: Raw event coordinates (ground truth structure)
            events_*_sampled: Sampled event coordinates
        
        Returns:
            AAC score (0 to 1, higher is better)
        """
        if len(events_y_raw) == 0:
            return 0.0
        
        if len(events_y_sampled) == 0:
            return 0.0
        
        # Create ground truth mask (M_GT)
        M_GT = np.zeros((self.height, self.width), dtype=np.bool_)
        y_int = np.clip(events_y_raw.astype(np.int32), 0, self.height - 1)
        x_int = np.clip(events_x_raw.astype(np.int32), 0, self.width - 1)
        M_GT[y_int, x_int] = True
        
        # Create sample coverage mask (M_Sample)
        # For each sampled point, draw a filled circle of radius = kernel_radius
        M_Sample = np.zeros((self.height, self.width), dtype=np.bool_)
        kernel_mask = self.create_kernel_mask(self.kernel_radius)
        
        for y, x in zip(events_y_sampled, events_x_sampled):
            y_c, x_c = int(y), int(x)
            
            # Calculate bounds for the kernel
            y_min = max(0, y_c - self.kernel_radius)
            y_max = min(self.height, y_c + self.kernel_radius + 1)
            x_min = max(0, x_c - self.kernel_radius)
            x_max = min(self.width, x_c + self.kernel_radius + 1)
            
            # Calculate kernel slice
            ky_min = max(0, self.kernel_radius - y_c)
            ky_max = ky_min + (y_max - y_min)
            kx_min = max(0, self.kernel_radius - x_c)
            kx_max = kx_min + (x_max - x_min)
            
            # Apply kernel
            M_Sample[y_min:y_max, x_min:x_max] |= kernel_mask[ky_min:ky_max, kx_min:kx_max].astype(np.bool_)
        
        # Compute AAC = |M_GT ∩ M_Sample| / |M_GT|
        intersection = np.logical_and(M_GT, M_Sample).sum()
        gt_size = M_GT.sum()
        
        aac = intersection / gt_size if gt_size > 0 else 0.0
        
        return float(aac)
    
    def compute_rf_score(
        self,
        events_y_raw: np.ndarray,
        events_x_raw: np.ndarray,
        events_y_sampled: np.ndarray,
        events_x_sampled: np.ndarray
    ) -> float:
        """
        Metric 2: Redundancy Factor (RF) - Efficiency
        
        Measures on average how many sample kernels cover a single active pixel.
        
        Goal: Optimize (Target ~2.0 - 4.0 for smooth convolution)
        
        Args:
            events_*_raw: Raw event coordinates (ground truth structure)
            events_*_sampled: Sampled event coordinates
        
        Returns:
            RF score (average kernel overlap, lower is more efficient)
        """
        if len(events_y_raw) == 0 or len(events_y_sampled) == 0:
            return 0.0
        
        # Create ground truth mask
        M_GT = np.zeros((self.height, self.width), dtype=np.bool_)
        y_int = np.clip(events_y_raw.astype(np.int32), 0, self.height - 1)
        x_int = np.clip(events_x_raw.astype(np.int32), 0, self.width - 1)
        M_GT[y_int, x_int] = True
        
        # Create accumulator grid (counts overlaps)
        accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        kernel_mask = self.create_kernel_mask(self.kernel_radius)
        
        for y, x in zip(events_y_sampled, events_x_sampled):
            y_c, x_c = int(y), int(x)
            
            # Calculate bounds
            y_min = max(0, y_c - self.kernel_radius)
            y_max = min(self.height, y_c + self.kernel_radius + 1)
            x_min = max(0, x_c - self.kernel_radius)
            x_max = min(self.width, x_c + self.kernel_radius + 1)
            
            # Calculate kernel slice
            ky_min = max(0, self.kernel_radius - y_c)
            ky_max = ky_min + (y_max - y_min)
            kx_min = max(0, self.kernel_radius - x_c)
            kx_max = kx_min + (x_max - x_min)
            
            # Add kernel contribution
            accumulator[y_min:y_max, x_min:x_max] += kernel_mask[ky_min:ky_max, kx_min:kx_max]
        
        # Compute RF = Mean(Accumulator[M_GT])
        # Only look at pixels where ground truth exists
        rf = accumulator[M_GT].mean() if M_GT.sum() > 0 else 0.0
        
        return float(rf)
    
    def compute_das_score(
        self,
        events_y_raw: np.ndarray,
        events_x_raw: np.ndarray,
        events_y_sampled: np.ndarray,
        events_x_sampled: np.ndarray
    ) -> float:
        """
        Metric 3: Density Alignment Score (DAS) - Focus
        
        Measures correlation between raw event density and sampled event density.
        High score means sampling adapts to scene complexity.
        
        Goal: Maximize (Target > 0.8)
        
        Args:
            events_*_raw: Raw event coordinates
            events_*_sampled: Sampled event coordinates
        
        Returns:
            DAS score (Pearson correlation, -1 to 1, higher is better)
        """
        if len(events_y_raw) < 2 or len(events_y_sampled) < 2:
            return 0.0
        
        # Compute density maps with Gaussian smoothing
        sigma = max(2.0, self.kernel_radius / 2)  # Smooth at half-kernel scale
        
        # Raw density map
        D_raw = np.zeros((self.height, self.width), dtype=np.float32)
        y_int = np.clip(events_y_raw.astype(np.int32), 0, self.height - 1)
        x_int = np.clip(events_x_raw.astype(np.int32), 0, self.width - 1)
        np.add.at(D_raw, (y_int, x_int), 1)
        D_raw = gaussian_filter(D_raw, sigma=sigma)
        
        # Sampled density map
        D_sample = np.zeros((self.height, self.width), dtype=np.float32)
        y_int_s = np.clip(events_y_sampled.astype(np.int32), 0, self.height - 1)
        x_int_s = np.clip(events_x_sampled.astype(np.int32), 0, self.width - 1)
        np.add.at(D_sample, (y_int_s, x_int_s), 1)
        D_sample = gaussian_filter(D_sample, sigma=sigma)
        
        # Flatten and compute Pearson correlation
        D_raw_flat = D_raw.flatten()
        D_sample_flat = D_sample.flatten()
        
        # Remove zero regions (no events at all)
        mask = (D_raw_flat > 0) | (D_sample_flat > 0)
        
        if mask.sum() < 2:
            return 0.0
        
        D_raw_flat = D_raw_flat[mask]
        D_sample_flat = D_sample_flat[mask]
        
        # Compute Pearson correlation
        try:
            corr, _ = pearsonr(D_raw_flat, D_sample_flat)
            das = corr if not np.isnan(corr) else 0.0
        except:
            das = 0.0
        
        return float(das)
    
    def compute_combined_score(
        self,
        aac: float,
        rf: float,
        das: float,
        target_rf: float = 3.0,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3
    ) -> float:
        """
        Combine metrics into a single score.
        
        Args:
            aac: Active Area Coverage (0-1, higher is better)
            rf: Redundancy Factor (target ~3.0)
            das: Density Alignment Score (0-1, higher is better)
            target_rf: Optimal redundancy (default 3.0 for smooth convolution)
            alpha, beta, gamma: Weights
        
        Returns:
            Combined score (0-1, higher is better)
        """
        # AAC: Already 0-1, maximize
        aac_score = aac
        
        # RF: Penalize deviation from target
        # Best if RF is close to target_rf (e.g., 3.0)
        # Convert to 0-1 score where 1.0 = perfect
        rf_deviation = abs(rf - target_rf) / max(target_rf, 1e-6)
        rf_score = max(0.0, 1.0 - rf_deviation)
        
        # DAS: Already 0-1 (shifted from -1 to 1)
        # Shift to 0-1 range
        das_score = (das + 1.0) / 2.0
        
        combined = alpha * aac_score + beta * rf_score + gamma * das_score
        
        return combined
    
    def evaluate_sampling_params(
        self,
        events_y: np.ndarray,
        events_x: np.ndarray,
        events_t: np.ndarray,
        events_p: np.ndarray,
        grid_size: int,
        retention_ratio: float
    ) -> Dict:
        """
        Evaluate a specific set of sampling parameters.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Apply sampling
        t_sampled, y_sampled, x_sampled, p_sampled = sample_grid_decimation_fast(
            events_t, events_y, events_x, events_p,
            self.height, self.width,
            target_grid=grid_size,
            retention_ratio=retention_ratio,
            return_indices=False
        )
        
        # Compute metrics
        aac = self.compute_aac_score(events_y, events_x, y_sampled, x_sampled)
        rf = self.compute_rf_score(events_y, events_x, y_sampled, x_sampled)
        das = self.compute_das_score(events_y, events_x, y_sampled, x_sampled)
        
        # Retention rate
        num_original = len(events_t)
        num_sampled = len(t_sampled)
        actual_retention = num_sampled / num_original if num_original > 0 else 0.0
        
        return {
            'grid_size': grid_size,
            'retention_ratio': retention_ratio,
            'aac': aac,
            'rf': rf,
            'das': das,
            'actual_retention': actual_retention,
            'num_original': num_original,
            'num_sampled': num_sampled
        }
    
    def benchmark_dataset(
        self,
        purpose: str = 'train'
    ) -> Dict:
        """
        Run benchmark on pre-denoised events loaded from cache.
        
        Args:
            purpose: 'train' or 'validation'
        
        Returns:
            Benchmark results dictionary
        """
        # Read benchmark parameters from config
        benchmark_cfg = self.config['BENCHMARKING']
        num_samples = benchmark_cfg['num_samples']
        grid_sizes = benchmark_cfg['sampling']['grid_sizes']
        retention_ratios = benchmark_cfg['sampling']['retention_ratios']
        
        print(f"\n{'='*60}")
        print(f"Starting Sampling Parameter Search")
        print(f"{'='*60}")
        print(f"Data split: {purpose}")
        print(f"Grid sizes: {grid_sizes}")
        print(f"Retention ratios: {retention_ratios}")
        print(f"Number of samples: {num_samples}")
        
        # Load denoised events from cache
        h5_path = self.denoised_cache_dir / f'{purpose}.h5'
        
        if not h5_path.exists():
            raise FileNotFoundError(
                f"Denoised cache not found: {h5_path}\n"
                f"Please run: python preprocess_denoise_only.py --config configs/config.yaml"
            )
        
        # Storage for results
        all_results = []
        
        # Iterate over parameter combinations
        total_configs = len(grid_sizes) * len(retention_ratios)
        
        with h5py.File(h5_path, 'r') as h5f:
            total_samples = h5f.attrs['num_samples']
            num_samples = min(num_samples, total_samples)
            
            print(f"Total samples in cache: {total_samples}")
            print(f"Using: {num_samples} samples")
            
            pbar = tqdm(total=total_configs * num_samples, desc="Benchmarking")
            
            for grid_size in grid_sizes:
                for retention_ratio in retention_ratios:
                    config_results = []
                    
                    # Test on multiple samples
                    for sample_idx in range(num_samples):
                        sample_group = h5f[f'sample_{sample_idx:06d}']
                        num_intervals = sample_group.attrs['num_intervals']
                        
                        # Process each interval
                        for interval_idx in range(num_intervals):
                            interval_group = sample_group[f'interval_{interval_idx:03d}']
                            
                            # Load denoised events
                            events_t = interval_group['t'][:]
                            events_y = interval_group['y'][:]
                            events_x = interval_group['x'][:]
                            events_p = interval_group['p'][:]
                            
                            if len(events_t) == 0:
                                continue
                            
                            # Evaluate sampling on denoised events
                            result = self.evaluate_sampling_params(
                                events_y,
                                events_x,
                                events_t,
                                events_p,
                                grid_size,
                                retention_ratio
                            )
                            
                            config_results.append(result)
                        
                        pbar.update(1)
                    
                    # Aggregate results for this configuration
                    if len(config_results) > 0:
                        avg_aac = np.mean([r['aac'] for r in config_results])
                        avg_rf = np.mean([r['rf'] for r in config_results])
                        avg_das = np.mean([r['das'] for r in config_results])
                        avg_retention = np.mean([r['actual_retention'] for r in config_results])
                        
                        all_results.append({
                            'grid_size': grid_size,
                            'retention_ratio': retention_ratio,
                            'avg_aac': avg_aac,
                            'avg_rf': avg_rf,
                            'avg_das': avg_das,
                            'avg_actual_retention': avg_retention,
                            'num_intervals': len(config_results)
                        })
            
            pbar.close()
        
        # Compute combined scores
        for r in all_results:
            r['combined_score'] = self.compute_combined_score(
                r['avg_aac'],
                r['avg_rf'],
                r['avg_das']
            )
        
        # Sort by combined score
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'dataset': self.dataset_name,
            'purpose': purpose,
            'num_samples': num_samples,
            'kernel_size': self.kernel_size,
            'denoised_cache': str(h5_path),
            'results': all_results,
            'best_params': all_results[0] if all_results else None
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"sampling_benchmark_{self.dataset_name}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def save_results_txt(self, results: Dict, filename: str = None):
        """Save human-readable benchmark results to text file."""
        if filename is None:
            filename = f"sampling_results_{self.dataset_name}.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Sampling Benchmark Results - {self.dataset_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Data split: {results['purpose']}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
            f.write(f"VecKM kernel size: {results['kernel_size']}\n")
            f.write(f"Denoised cache: {results['denoised_cache']}\n")
            f.write(f"Total configurations tested: {len(results['results'])}\n\n")
            
            if results['best_params']:
                best = results['best_params']
                f.write("="*80 + "\n")
                f.write("RECOMMENDED PARAMETERS:\n")
                f.write("="*80 + "\n")
                f.write(f"  grid_size: {best['grid_size']}\n")
                f.write(f"  retention_ratio: {best['retention_ratio']:.2f}\n")
                f.write(f"\n")
                f.write(f"METRICS:\n")
                f.write(f"  AAC (Coverage):    {best['avg_aac']:.4f} (target > 0.95)\n")
                f.write(f"  RF (Redundancy):   {best['avg_rf']:.2f} (target ~2-4)\n")
                f.write(f"  DAS (Alignment):   {best['avg_das']:.4f} (target > 0.8)\n")
                f.write(f"  Actual Retention:  {best['avg_actual_retention']:.4f}\n")
                f.write(f"  Combined Score:    {best['combined_score']:.4f}\n")
                f.write("="*80 + "\n\n")
            
            f.write("ALL CONFIGURATIONS (sorted by combined score):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Grid':<6} {'Ratio':<8} {'AAC':<8} {'RF':<8} {'DAS':<8} {'Retention':<10} {'Combined':<10}\n")
            f.write("-"*80 + "\n")
            
            for idx, r in enumerate(results['results']):
                f.write(f"{idx+1:<6} "
                       f"{r['grid_size']:<6} "
                       f"{r['retention_ratio']:<8.2f} "
                       f"{r['avg_aac']:<8.4f} "
                       f"{r['avg_rf']:<8.2f} "
                       f"{r['avg_das']:<8.4f} "
                       f"{r['avg_actual_retention']:<10.4f} "
                       f"{r['combined_score']:<10.4f}\n")
        
        print(f"Text results saved to: {output_path}")
    
    def print_results(self, results: Dict, top_k: int = 10):
        """Print benchmark results in a readable format."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results - Top {top_k} Configurations")
        print(f"{'='*60}")
        
        print(f"\n{'Rank':<6} {'Grid':<6} {'Ratio':<8} {'AAC':<8} {'RF':<8} {'DAS':<8} {'Retention':<10} {'Combined':<10}")
        print("-" * 80)
        
        for idx, r in enumerate(results['results'][:top_k]):
            print(f"{idx+1:<6} "
                  f"{r['grid_size']:<6} "
                  f"{r['retention_ratio']:<8.2f} "
                  f"{r['avg_aac']:<8.4f} "
                  f"{r['avg_rf']:<8.2f} "
                  f"{r['avg_das']:<8.4f} "
                  f"{r['avg_actual_retention']:<10.4f} "
                  f"{r['combined_score']:<10.4f}")
        
        if results['best_params']:
            best = results['best_params']
            print(f"\n{'='*60}")
            print(f"RECOMMENDED PARAMETERS:")
            print(f"  grid_size: {best['grid_size']}")
            print(f"  retention_ratio: {best['retention_ratio']:.2f}")
            print(f"")
            print(f"METRICS:")
            print(f"  AAC (Coverage):    {best['avg_aac']:.4f} (target > 0.95)")
            print(f"  RF (Redundancy):   {best['avg_rf']:.2f} (target ~2-4)")
            print(f"  DAS (Alignment):   {best['avg_das']:.4f} (target > 0.8)")
            print(f"{'='*60}")
    
    def run(
        self,
        purpose: str = 'train',
        save: bool = True
    ):
        """Run the complete benchmark pipeline."""
        # Run benchmark (reads parameters from config)
        results = self.benchmark_dataset(purpose)
        
        # Print results
        self.print_results(results)
        
        # Save results
        if save:
            self.save_results(results)
            self.save_results_txt(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark sampling parameters on denoised events')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dvsgesture',
        help='Dataset name'
    )
    parser.add_argument(
        '--purpose',
        type=str,
        default='train',
        choices=['train', 'validation'],
        help='Data split to use (train or validation)'
    )
    
    args = parser.parse_args()
    
    # Load config with OmegaConf to resolve interpolations
    config = OmegaConf.load(args.config)
    
    # Create benchmark
    benchmark = SamplingBenchmark(config, args.dataset)
    
    # Run benchmark
    benchmark.run(
        purpose=args.purpose,
        save=True
    )


if __name__ == '__main__':
    main()
