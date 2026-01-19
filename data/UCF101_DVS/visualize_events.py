"""
Self-contained visualization script for UCF101-DVS dataset.
Visualizes raw event data without requiring external configs.

Usage:
    python visualize_events.py --sample_idx 0 --output_dir ./visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add parent directory to path to import dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import UCF101_DVS, UCF101_CLASSES


def visualize_event_frames(events_xy_sliced, events_p_sliced, height, width, num_frames=10, min_events=100):
    """
    Visualize event frames as accumulated images.
    
    Args:
        events_xy_sliced: List of event coordinates for each interval
        events_p_sliced: List of event polarities for each interval  
        height, width: Sensor dimensions
        num_frames: Number of frames to visualize
        min_events: Minimum events per interval to consider (default: 100)
    """
    # Find intervals with enough events
    valid_intervals = []
    for i in range(len(events_xy_sliced)):
        if len(events_xy_sliced[i]) >= min_events:
            valid_intervals.append(i)
    
    if len(valid_intervals) == 0:
        print(f"Warning: No intervals with >={min_events} events found!")
        # Fall back to showing all intervals
        valid_intervals = list(range(len(events_xy_sliced)))
    
    # Select evenly spaced intervals from valid ones
    if len(valid_intervals) > num_frames:
        step = len(valid_intervals) // num_frames
        selected_intervals = [valid_intervals[i * step] for i in range(num_frames)]
    else:
        selected_intervals = valid_intervals[:num_frames]
    
    num_intervals = len(selected_intervals)
    
    fig, axes = plt.subplots(2, (num_intervals + 1) // 2, figsize=(15, 6))
    axes = axes.flatten() if num_intervals > 1 else [axes]
    
    for plot_idx, interval_idx in enumerate(selected_intervals):
        events_xy = events_xy_sliced[interval_idx]
        events_p = events_p_sliced[interval_idx]
        
        # Create accumulation frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(events_xy) > 0:
            x = events_xy[:, 0].astype(int)
            y = events_xy[:, 1].astype(int)
            
            # Clip to valid range
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            
            # ON events (polarity=1) in red, OFF events (polarity=0) in blue
            for j in range(len(x)):
                if events_p[j] > 0.5:  # ON event
                    frame[y[j], x[j], 0] = 255  # Red channel
                else:  # OFF event
                    frame[y[j], x[j], 2] = 255  # Blue channel
        
        axes[plot_idx].imshow(frame)
        axes[plot_idx].set_title(f'Interval {interval_idx} ({len(events_xy)} events)')
        axes[plot_idx].axis('off')
    
    # Hide unused subplots
    for i in range(num_intervals, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize UCF101-DVS events')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='~/Downloads/UCF101_DVS',
        help='Path to UCF101_DVS dataset'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Sample index to visualize'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./visualizations/ucf101_dvs',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=10,
        help='Number of time intervals to visualize'
    )
    parser.add_argument(
        '--purpose',
        type=str,
        default='train',
        choices=['train', 'validation', 'all'],
        help='Dataset split to use'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading UCF101-DVS dataset from {args.dataset_dir}...")
    dataset = UCF101_DVS(
        dataset_dir=args.dataset_dir,
        purpose=args.purpose,
        height=180,
        width=240,
        accumulation_interval_ms=50.0,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    if args.sample_idx >= len(dataset):
        print(f"Error: Sample index {args.sample_idx} out of range (0-{len(dataset)-1})")
        return
    
    # Load sample
    print(f"Loading sample {args.sample_idx}...")
    sample = dataset[args.sample_idx]
    
    label = sample['label']
    class_name = UCF101_CLASSES[label]
    file_path = sample['file_path']
    
    print(f"Class: {class_name} (label {label})")
    print(f"File: {file_path}")
    print(f"Number of time intervals: {len(sample['events_xy_sliced'])}")
    
    # Visualize
    print("Creating visualization...")
    fig = visualize_event_frames(
        sample['events_xy_sliced'],
        sample['events_p_sliced'],
        height=180,
        width=240,
        num_frames=args.num_frames
    )
    
    # Save
    output_path = output_dir / f'ucf101_dvs_sample_{args.sample_idx}_{class_name}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
