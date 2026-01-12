"""
Automated Denoising Parameter Benchmark

This script implements two quantitative evaluation metrics for event denoising:
1. Contrast Maximization (Variance of IWE) - Gallego et al., CVPR 2019
2. Event Structural Ratio (ESR) - Ding et al., E-MLB 2023

The benchmark automatically searches for optimal denoising parameters for each dataset.

Usage:
    python benchmark_denoising.py --dataset dvsgesture --config configs/config.yaml
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.DVSGesture import DVSGesture
from utils.denoising_and_sampling import filter_noise_spatial


class DenoisingBenchmark:
    """
    Automated benchmark for finding optimal denoising parameters.
    """
    
    def __init__(self, config: Dict, dataset_name: str = 'dvsgesture'):
        self.config = config
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load dataset parameters
        precompute_cfg = config['PRECOMPUTING']
        self.dataset_dir = precompute_cfg['dataset_dir']
        self.height = precompute_cfg['height']
        self.width = precompute_cfg['width']
        self.accumulation_interval_ms = precompute_cfg['accumulation_interval_ms']
        
        # Benchmark output directory
        self.output_dir = Path(config['PRECOMPUTING']['output_dir']).parent / 'denoising_benchmark'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Denoising Benchmark Initialized")
        print(f"Dataset: {dataset_name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
    
    def compute_variance_score(
        self, 
        events_t: np.ndarray,
        events_y: np.ndarray,
        events_x: np.ndarray,
        normalize: bool = True
    ) -> float:
        """
        Metric 1: Contrast Maximization (Variance of IWE)
        
        Computes the variance of the Image of Warped Events (IWE).
        High variance indicates sharp edges (good signal), low variance indicates noise.
        
        Args:
            events_t, events_y, events_x: Event data
            normalize: If True, normalize by event count to prevent "empty set" solution
        
        Returns:
            Variance score (higher is better)
        """
        if len(events_t) == 0:
            return 0.0
        
        # Create a 2D histogram (Image of Warped Events)
        H = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Accumulate events
        y_int = events_y.astype(np.int32)
        x_int = events_x.astype(np.int32)
        
        # Clip to valid range
        y_int = np.clip(y_int, 0, self.height - 1)
        x_int = np.clip(x_int, 0, self.width - 1)
        
        # Count events per pixel
        np.add.at(H, (y_int, x_int), 1)
        
        # Compute variance
        variance = np.var(H)
        
        # Normalize by event count to prevent trivial solutions
        if normalize and len(events_t) > 0:
            variance = variance / (len(events_t) + 1e-6)
        
        return float(variance)
    
    def compute_esr_score(
        self,
        events_t_raw: np.ndarray,
        events_y_raw: np.ndarray,
        events_x_raw: np.ndarray,
        events_t_denoised: np.ndarray,
        events_y_denoised: np.ndarray,
        events_x_denoised: np.ndarray
    ) -> float:
        """
        Metric 2: Event Structural Ratio (ESR)
        
        Measures the "structural intensity" of the event stream.
        ESR = sum(H_denoised^2) / sum(H_raw^2)
        
        This balances signal preservation against noise removal.
        
        Args:
            events_*_raw: Raw (noisy) event data
            events_*_denoised: Denoised event data
        
        Returns:
            ESR score (0 to 1, higher indicates better structure preservation)
        """
        if len(events_t_raw) == 0 or len(events_t_denoised) == 0:
            return 0.0
        
        # Create histograms
        H_raw = np.zeros((self.height, self.width), dtype=np.float32)
        H_denoised = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Accumulate raw events
        y_raw_int = np.clip(events_y_raw.astype(np.int32), 0, self.height - 1)
        x_raw_int = np.clip(events_x_raw.astype(np.int32), 0, self.width - 1)
        np.add.at(H_raw, (y_raw_int, x_raw_int), 1)
        
        # Accumulate denoised events
        y_den_int = np.clip(events_y_denoised.astype(np.int32), 0, self.height - 1)
        x_den_int = np.clip(events_x_denoised.astype(np.int32), 0, self.width - 1)
        np.add.at(H_denoised, (y_den_int, x_den_int), 1)
        
        # Compute ESR
        energy_raw = np.sum(H_raw ** 2)
        energy_denoised = np.sum(H_denoised ** 2)
        
        if energy_raw == 0:
            return 0.0
        
        esr = energy_denoised / energy_raw
        
        return float(esr)
    
    def compute_combined_score(
        self,
        variance_score: float,
        esr_score: float,
        retention_rate: float,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ) -> float:
        """
        Combine multiple metrics into a single score.
        
        Args:
            variance_score: Normalized variance (0-1)
            esr_score: Event Structural Ratio (0-1)
            retention_rate: Percentage of events retained (0-1)
            alpha, beta, gamma: Weights (should sum to 1.0)
        
        Returns:
            Combined score (0-1, higher is better)
        """
        # We want high variance and high ESR
        # We prefer moderate retention (not too aggressive, not too lenient)
        # Penalize very low retention (< 10%) and very high retention (> 95%)
        retention_penalty = 1.0
        if retention_rate < 0.1:
            retention_penalty = retention_rate / 0.1
        elif retention_rate > 0.95:
            retention_penalty = (1.0 - retention_rate) / 0.05
        
        combined = (
            alpha * variance_score + 
            beta * esr_score + 
            gamma * retention_penalty
        )
        
        return combined
    
    def evaluate_denoising_params(
        self,
        events_t: np.ndarray,
        events_y: np.ndarray,
        events_x: np.ndarray,
        events_p: np.ndarray,
        grid_size: int,
        threshold: int
    ) -> Dict:
        """
        Evaluate a specific set of denoising parameters.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Apply denoising (pure numpy)
        t_clean, y_clean, x_clean, p_clean = filter_noise_spatial(
            events_t, events_y, events_x, events_p,
            self.height, self.width,
            grid_size, threshold
        )
        
        # Compute metrics
        variance_score = self.compute_variance_score(t_clean, y_clean, x_clean)
        esr_score = self.compute_esr_score(
            events_t, events_y, events_x,
            t_clean, y_clean, x_clean
        )
        
        # Retention rate
        num_original = len(events_t)
        num_retained = len(t_clean)
        retention_rate = num_retained / num_original if num_original > 0 else 0.0
        
        # Normalize variance score for combined metric (approximate normalization)
        # We'll track max variance across all trials for proper normalization
        variance_normalized = variance_score  # Will normalize later
        
        return {
            'grid_size': grid_size,
            'threshold': threshold,
            'variance_score': variance_score,
            'esr_score': esr_score,
            'retention_rate': retention_rate,
            'num_original': num_original,
            'num_retained': num_retained
        }
    
    def benchmark_dataset(
        self,
        num_samples: int = 100,
        grid_sizes: List[int] = [2, 3, 4, 5, 6, 8],
        thresholds: List[int] = [1, 2, 3, 4, 5]
    ) -> Dict:
        """
        Run benchmark on a subset of the dataset.
        
        Args:
            num_samples: Number of samples to evaluate
            grid_sizes: List of grid sizes to test
            thresholds: List of thresholds to test
        
        Returns:
            Benchmark results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting Denoising Parameter Search")
        print(f"{'='*60}")
        print(f"Grid sizes: {grid_sizes}")
        print(f"Thresholds: {thresholds}")
        print(f"Number of samples: {num_samples}")
        
        # Load dataset
        dataset = DVSGesture(
            dataset_dir=self.dataset_dir,
            purpose='train',
            height=self.height,
            width=self.width,
            use_flip_augmentation=False,
            accumulation_interval_ms=self.accumulation_interval_ms
        )
        
        # Limit number of samples
        num_samples = min(num_samples, len(dataset))
        
        # Storage for results
        all_results = []
        
        # Iterate over parameter combinations
        total_configs = len(grid_sizes) * len(thresholds)
        pbar = tqdm(total=total_configs * num_samples, desc="Benchmarking")
        
        for grid_size in grid_sizes:
            for threshold in thresholds:
                config_results = []
                
                # Test on multiple samples
                for sample_idx in range(num_samples):
                    sample = dataset[sample_idx]
                    
                    # Process each interval
                    for interval_idx in range(len(sample['events_t_sliced'])):
                        events_t = sample['events_t_sliced'][interval_idx]
                        events_xy = sample['events_xy_sliced'][interval_idx]
                        events_p = sample['events_p_sliced'][interval_idx]
                        
                        if len(events_t) == 0:
                            continue
                        
                        result = self.evaluate_denoising_params(
                            events_t,
                            events_xy[:, 1],  # y
                            events_xy[:, 0],  # x
                            events_p,
                            grid_size,
                            threshold
                        )
                        
                        config_results.append(result)
                    
                    pbar.update(1)
                
                # Aggregate results for this configuration
                if len(config_results) > 0:
                    avg_variance = np.mean([r['variance_score'] for r in config_results])
                    avg_esr = np.mean([r['esr_score'] for r in config_results])
                    avg_retention = np.mean([r['retention_rate'] for r in config_results])
                    
                    all_results.append({
                        'grid_size': grid_size,
                        'threshold': threshold,
                        'avg_variance': avg_variance,
                        'avg_esr': avg_esr,
                        'avg_retention': avg_retention,
                        'num_intervals': len(config_results)
                    })
        
        pbar.close()
        
        # Normalize variance scores
        if len(all_results) > 0:
            max_variance = max(r['avg_variance'] for r in all_results)
            if max_variance > 0:
                for r in all_results:
                    r['variance_normalized'] = r['avg_variance'] / max_variance
            else:
                for r in all_results:
                    r['variance_normalized'] = 0.0
            
            # Compute combined scores
            for r in all_results:
                r['combined_score'] = self.compute_combined_score(
                    r['variance_normalized'],
                    r['avg_esr'],
                    r['avg_retention']
                )
        
        # Sort by combined score
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'dataset': self.dataset_name,
            'num_samples': num_samples,
            'results': all_results,
            'best_params': all_results[0] if all_results else None
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"denoising_benchmark_{self.dataset_name}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def save_results_txt(self, results: Dict, filename: str = None):
        """Save human-readable benchmark results to text file."""
        if filename is None:
            filename = f"denoising_results_{self.dataset_name}.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Denoising Benchmark Results - {self.dataset_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
            f.write(f"Total configurations tested: {len(results['results'])}\n\n")
            
            if results['best_params']:
                best = results['best_params']
                f.write("="*80 + "\n")
                f.write("RECOMMENDED PARAMETERS:\n")
                f.write("="*80 + "\n")
                f.write(f"  grid_size: {best['grid_size']}\n")
                f.write(f"  threshold: {best['threshold']}\n")
                f.write(f"\n")
                f.write(f"METRICS:\n")
                f.write(f"  Variance (normalized): {best['variance_normalized']:.6f}\n")
                f.write(f"  ESR:                   {best['avg_esr']:.4f}\n")
                f.write(f"  Retention:             {best['avg_retention']:.4f}\n")
                f.write(f"  Combined Score:        {best['combined_score']:.4f}\n")
                f.write("="*80 + "\n\n")
            
            f.write("ALL CONFIGURATIONS (sorted by combined score):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Grid':<6} {'Thresh':<7} {'Variance':<12} {'ESR':<8} {'Retention':<10} {'Combined':<10}\n")
            f.write("-"*80 + "\n")
            
            for idx, r in enumerate(results['results']):
                f.write(f"{idx+1:<6} "
                       f"{r['grid_size']:<6} "
                       f"{r['threshold']:<7} "
                       f"{r['avg_variance']:<12.6f} "
                       f"{r['avg_esr']:<8.4f} "
                       f"{r['avg_retention']:<10.4f} "
                       f"{r['combined_score']:<10.4f}\n")
        
        print(f"Text results saved to: {output_path}")
    
    def print_results(self, results: Dict, top_k: int = 10):
        """Print benchmark results in a readable format."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results - Top {top_k} Configurations")
        print(f"{'='*60}")
        
        print(f"\n{'Rank':<6} {'Grid':<6} {'Thresh':<7} {'Variance':<12} {'ESR':<8} {'Retention':<10} {'Combined':<10}")
        print("-" * 75)
        
        for idx, r in enumerate(results['results'][:top_k]):
            print(f"{idx+1:<6} "
                  f"{r['grid_size']:<6} "
                  f"{r['threshold']:<7} "
                  f"{r['avg_variance']:<12.6f} "
                  f"{r['avg_esr']:<8.4f} "
                  f"{r['avg_retention']:<10.4f} "
                  f"{r['combined_score']:<10.4f}")
        
        if results['best_params']:
            best = results['best_params']
            print(f"\n{'='*60}")
            print(f"RECOMMENDED PARAMETERS:")
            print(f"  grid_size: {best['grid_size']}")
            print(f"  threshold: {best['threshold']}")
            print(f"{'='*60}")
    
    def run(
        self,
        num_samples: int = 100,
        grid_sizes: List[int] = None,
        thresholds: List[int] = None,
        save: bool = True
    ):
        """Run the complete benchmark pipeline."""
        if grid_sizes is None:
            grid_sizes = [2, 3, 4, 5, 6, 8, 10]
        
        if thresholds is None:
            thresholds = [1, 2, 3, 4, 5, 6]
        
        # Run benchmark
        results = self.benchmark_dataset(num_samples, grid_sizes, thresholds)
        
        # Print results
        self.print_results(results)
        
        # Save results
        if save:
            self.save_results(results)
            self.save_results_txt(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark denoising parameters')
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
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to use for benchmarking'
    )
    parser.add_argument(
        '--grid_sizes',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5, 6, 8, 10],
        help='Grid sizes to test'
    )
    parser.add_argument(
        '--thresholds',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4, 5, 6],
        help='Thresholds to test'
    )
    
    args = parser.parse_args()
    
    # Load config with OmegaConf to resolve interpolations
    config = OmegaConf.load(args.config)
    
    # Create benchmark
    benchmark = DenoisingBenchmark(config, args.dataset)
    
    # Run benchmark
    benchmark.run(
        num_samples=args.num_samples,
        grid_sizes=args.grid_sizes,
        thresholds=args.thresholds,
        save=True
    )


if __name__ == '__main__':
    main()
