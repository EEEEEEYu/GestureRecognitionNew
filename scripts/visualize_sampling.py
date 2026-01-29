import os
import sys
import numpy as np
import argparse
import imageio
import cv2  # Use OpenCV for video writing
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Add project root to path
# Assuming this script is in scripts/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.HMDB_DVS.dataset import HMDB_DVS
from data.UCF101_DVS.dataset import UCF101_DVS
from data.dvsgesture.dataset import DVSGesture
from utils.density_adaptive_spatial_striding import adaptive_spatial_sampling
from utils.denoising_and_sampling import filter_noise_spatial

def get_dataset_class_and_name(config_path):
    if 'hmdb' in config_path.lower():
        return HMDB_DVS, "HMDB_DVS"
    elif 'ucf101' in config_path.lower():
        return UCF101_DVS, "UCF101_DVS"
    elif 'dvsgesture' in config_path.lower():
        return DVSGesture, "DVSGesture"
    else:
        # Fallback: check content (simplified)
        return HMDB_DVS, "HMDB_DVS" # Default or raise error

def draw_hollow_square(img, center_y, center_x, size, color):
    """Draw a hollow square on the image."""
    h, w, _ = img.shape
    half_size = size // 2
    
    y_min = max(0, center_y - half_size)
    y_max = min(h - 1, center_y + half_size)
    x_min = max(0, center_x - half_size)
    x_max = min(w - 1, center_x + half_size)
    
    # Top and Bottom
    img[y_min, x_min:x_max+1] = color
    img[y_max, x_min:x_max+1] = color
    
    # Left and Right
    img[y_min:y_max+1, x_min] = color
    img[y_min:y_max+1, x_max] = color

def visualize_sampling(config_path, sample_idx, output_file=None, max_intervals=None):
    # Load config
    config = OmegaConf.load(config_path)
    precompute_cfg = config['PRECOMPUTING']
    
    # Identify Dataset
    DatasetClass, dataset_name_detected = get_dataset_class_and_name(config_path)
    
    # Parameters
    dataset_dir = precompute_cfg['dataset_dir']
    accumulation_interval_ms = float(precompute_cfg['accumulation_interval_ms'])
    height = int(precompute_cfg['height'])
    width = int(precompute_cfg['width'])
    
    # Calculate FPS
    fps = 1000.0 / accumulation_interval_ms
    print(f"Calculated FPS: {fps} (Interval: {accumulation_interval_ms}ms)")
    
    # Denoising
    denoising_cfg = OmegaConf.select(precompute_cfg, 'denoising', default={})
    denoising_enabled = OmegaConf.select(denoising_cfg, 'enabled', default=False)
    denoise_grid_size = int(OmegaConf.select(denoising_cfg, 'grid_size', default=4))
    denoise_threshold = int(OmegaConf.select(denoising_cfg, 'threshold', default=2))
    
    # Sampling
    sampling_cfg = OmegaConf.select(precompute_cfg, 'sampling', default={})
    sampling_method = OmegaConf.select(sampling_cfg, 'method', default='random')
    adaptive_cfg = OmegaConf.select(sampling_cfg, 'adaptive_striding', default={})
    kernel_size = int(OmegaConf.select(adaptive_cfg, 'kernel_size', default=17))
    overlap_factor = float(OmegaConf.select(adaptive_cfg, 'overlap_factor', default=0.0))

    # Construct dynamic filename
    if output_file is None or output_file == 'hmdb_sampling_comparison.mp4' or output_file == 'hmdb_sampling_vis.gif':
        denoise_str = f"denoise({denoise_grid_size},{denoise_threshold})" if denoising_enabled else "no_denoise"
        overlap_str = f"overlap{overlap_factor}"
        output_file = f"{dataset_name_detected}_sample{sample_idx}_{denoise_str}_{overlap_str}.mp4"
    
    print(f"Loading {dataset_name_detected} Dataset from {dataset_dir}...")
    
    # Instantiate Dataset (Handle differences in __init__)
    
    init_args = {
        'dataset_dir': dataset_dir,
        'purpose': 'train',
        'height': height,
        'width': width,
        'accumulation_interval_ms': accumulation_interval_ms,
        'use_flip_augmentation': False
    }
    
    # Add optional args if supported
    if DatasetClass == UCF101_DVS or DatasetClass == HMDB_DVS:
        init_args['train_split'] = 0.8 
        
    # Check if DVSGesture accepts use_flip_augmentation (it does, based on file reading)
    # Check if DVSGesture accepts accumulation_interval_ms (it does)
    # DVSGesture does NOT accept train_split
     
    dataset = DatasetClass(**init_args)
    
    if sample_idx >= len(dataset):
        print(f"Error: Sample index {sample_idx} out of bounds (max {len(dataset)-1})")
        return

    print(f"Processing sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    events_xy_sliced = sample['events_xy_sliced']
    events_t_sliced = sample['events_t_sliced']
    events_p_sliced = sample['events_p_sliced']
    
    num_intervals = len(events_xy_sliced)
    if max_intervals is not None:
        num_intervals = min(num_intervals, max_intervals)
    
    print(f"Total intervals: {len(events_xy_sliced)}, visualizing first {num_intervals}")
    
    # Setup Video Writer
    # Side-by-side view: 2 * width
    vis_width = width * 2
    vis_height = height
    
    # Try using cv2.VideoWriter for MP4
    # Codec: 'mp4v' or 'avc1' usually works
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_file, fourcc, fps, (vis_width, vis_height))
    
    if not out.isOpened():
        print("Error: Could not create video writer with cv2. Trying 'avc1'...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_file, fourcc, fps, (vis_width, vis_height))
        
        if not out.isOpened():
            print("Error: Could not create video writer. Fallback to MJPG (avi)...")
            output_file = output_file.replace('.mp4', '.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_file, fourcc, fps, (vis_width, vis_height))
            
    print(f"Writing video to {output_file}...")
    
    # Statistics Collection
    stats_vectors_per_interval = []
    
    for i in tqdm(range(num_intervals), desc="Generating Frames"):
        # Setup Frame canvas
        # Left: Clean, Right: Sampled
        full_frame = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Left frame ROI
        left_frame = full_frame[:, :width]
        # Right frame ROI
        right_frame = full_frame[:, width:]
        
        events_xy = events_xy_sliced[i]
        events_t = events_t_sliced[i]
        events_p = events_p_sliced[i]
        
        if len(events_t) == 0:
            out.write(full_frame)
            stats_vectors_per_interval.append(0)
            continue
            
        # Separate X/Y
        events_x = events_xy[:, 0]
        events_y = events_xy[:, 1]
        
        # Clip
        events_x = np.clip(events_x, 0, width - 1)
        events_y = np.clip(events_y, 0, height - 1)
        
        # 1. Denoising
        if denoising_enabled:
            events_t_clean, events_y_clean, events_x_clean, events_p_clean = filter_noise_spatial(
                events_t, events_y, events_x, events_p,
                height, width,
                denoise_grid_size,
                denoise_threshold
            )
        else:
            events_t_clean, events_y_clean, events_x_clean, events_p_clean = events_t, events_y, events_x, events_p
            
        # 2. Draw Events on BOTH frames
        # Use BGR for OpenCV
        # Red: [0, 0, 255], Blue: [255, 0, 0]
        
        # NOTE: The "base" event visualization is identical for both left and right frames.
        # We draw the exact same set of 'clean' events on both sides.
        # The right frame only differs by the addition of the green overlay boxes.
        
        for j in range(len(events_x_clean)):
            x, y, p = int(events_x_clean[j]), int(events_y_clean[j]), int(events_p_clean[j])
            color = [0, 0, 255] if p == 1 else [255, 0, 0] # Red if ON, Blue if OFF
            
            # Draw on Left (Clean)
            left_frame[y, x] = color
            # Draw on Right (Sampled base)
            right_frame[y, x] = color
                
        # 3. Sampling Logic
        num_sampled_vectors = 0
        if sampling_method == 'adaptive_striding':
            query_indices = adaptive_spatial_sampling(
                events_t_clean, events_y_clean, events_x_clean, events_p_clean,
                height=height, width=width,
                kernel_size=kernel_size,
                overlap_factor=overlap_factor,
                sort_by_time=True
            )
            
            num_sampled_vectors = len(query_indices)
            
            # 4. Mark Sampled Areas on RIGHT frame
            # Green: [0, 255, 0]
            sampled_x = events_x_clean[query_indices]
            sampled_y = events_y_clean[query_indices]
            
            green = [0, 255, 0]
            for j in range(len(sampled_x)):
                sx, sy = int(sampled_x[j]), int(sampled_y[j])
                draw_hollow_square(right_frame, sy, sx, kernel_size, green)
        
        elif sampling_method == 'random':
             # If random, num_vectors might be all clean events (as per current preprocess logic fallback)
             # But let's assume we aren't using random for this visualization task
             num_sampled_vectors = len(events_x_clean)

        stats_vectors_per_interval.append(num_sampled_vectors)
        
        # Write frame
        out.write(full_frame)

    out.release()
    print("\n" + "="*50)
    print("STATISTICS REPORT")
    print("="*50)
    print(f"Sample Index: {sample_idx}")
    print(f"Total Intervals (Processed): {len(stats_vectors_per_interval)}")
    
    if len(stats_vectors_per_interval) > 0:
        total_vecs = sum(stats_vectors_per_interval)
        avg_vecs = np.mean(stats_vectors_per_interval)
        max_vecs = np.max(stats_vectors_per_interval)
        min_vecs = np.min(stats_vectors_per_interval)
        
        print(f"Total Vectors: {total_vecs}")
        print(f"Avg Vectors per Interval: {avg_vecs:.2f}")
        print(f"Max Vectors per Interval: {max_vecs}")
        print(f"Min Vectors per Interval: {min_vecs}")
    else:
        print("No intervals processed.")
    print("="*50)
    print(f"Video saved to: {output_file}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Event Sampling')
    parser.add_argument('--config', type=str, default='configs/config_hmdb.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default=None, help='Output video file')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--duration', type=int, default=100, help='Max number of intervals to visualize')
    
    args = parser.parse_args()
    
    visualize_sampling(args.config, args.sample_idx, args.output, args.duration)
