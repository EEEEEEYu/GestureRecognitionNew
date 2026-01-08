"""
Test script to verify sequence filtering configuration.

This script helps preview which sequences will be selected based on your
filter configuration WITHOUT running the full preprocessing pipeline.
"""

import os
import sys
import yaml
from collections import Counter, defaultdict
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_sequence_name(seq_name: str) -> Dict[str, str]:
    """Parse sequence name to extract metadata."""
    parts = seq_name.split('_')
    person_id = parts[1]
    view = parts[2]
    background = parts[3]
    lighting = parts[4]
    class_parts = parts[5:]
    class_name = '_'.join(class_parts)
    
    return {
        'person_id': person_id,
        'view': view,
        'background': background,
        'lighting': lighting,
        'class_name': class_name,
    }


def filter_sequences(sequences: List[str], filter_config: Dict) -> List[str]:
    """
    Filter sequences based on view, lighting, and background conditions.
    
    Args:
        sequences: List of sequence names to filter
        filter_config: Dictionary with 'view', 'lighting', 'background' keys
        
    Returns:
        Filtered list of sequences matching the filter criteria
    """
    def matches_filter(value: str, filter_value) -> bool:
        """Check if a value matches the filter."""
        if filter_value in ['both', 'all']:
            return True
        elif isinstance(filter_value, list):
            return value in filter_value
        else:
            return value == filter_value
    
    filtered = []
    for seq in sequences:
        metadata = parse_sequence_name(seq)
        
        # Check each filter condition
        if not matches_filter(metadata['view'], filter_config['view']):
            continue
        if not matches_filter(metadata['lighting'], filter_config['lighting']):
            continue
        if not matches_filter(metadata['background'], filter_config['background']):
            continue
        
        filtered.append(seq)
    
    return filtered


def main():
    # Load config
    config_path = 'configs/custom_gesture_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    precompute_cfg = config['PRECOMPUTING']
    dataset_dir = precompute_cfg['dataset_dir']
    filter_config = precompute_cfg.get('filter', {
        'view': 'both',
        'lighting': 'both',
        'background': 'both'
    })
    
    # Get all sequences
    all_sequences = [d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('sequence_')]
    
    print("=" * 80)
    print("SEQUENCE FILTERING TEST")
    print("=" * 80)
    print(f"\nDataset directory: {dataset_dir}")
    print(f"Total sequences found: {len(all_sequences)}")
    
    # Show current filter settings
    print(f"\nCurrent filter settings (from {config_path}):")
    print(f"  View:       {filter_config['view']}")
    print(f"  Lighting:   {filter_config['lighting']}")
    print(f"  Background: {filter_config['background']}")
    
    # Apply filters
    filtered_sequences = filter_sequences(all_sequences, filter_config)
    
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Sequences after filtering: {len(filtered_sequences)} / {len(all_sequences)}")
    print(f"Reduction: {100 * (1 - len(filtered_sequences)/len(all_sequences)):.1f}%")
    
    # Analyze filtered sequences
    view_counts = Counter()
    background_counts = Counter()
    lighting_counts = Counter()
    person_counts = Counter()
    class_counts = Counter()
    
    for seq in filtered_sequences:
        metadata = parse_sequence_name(seq)
        view_counts[metadata['view']] += 1
        background_counts[metadata['background']] += 1
        lighting_counts[metadata['lighting']] += 1
        person_counts[metadata['person_id']] += 1
        class_counts[metadata['class_name']] += 1
    
    print(f"\nDistribution of filtered sequences:")
    print(f"\n  By View:")
    for view, count in sorted(view_counts.items()):
        pct = 100 * count / len(filtered_sequences)
        print(f"    {view:10s}: {count:4d} sequences ({pct:.1f}%)")
    
    print(f"\n  By Background:")
    for bg, count in sorted(background_counts.items()):
        pct = 100 * count / len(filtered_sequences)
        print(f"    {bg:10s}: {count:4d} sequences ({pct:.1f}%)")
    
    print(f"\n  By Lighting:")
    for light, count in sorted(lighting_counts.items()):
        pct = 100 * count / len(filtered_sequences)
        print(f"    {light:10s}: {count:4d} sequences ({pct:.1f}%)")
    
    print(f"\n  By Person:")
    for person, count in sorted(person_counts.items()):
        pct = 100 * count / len(filtered_sequences)
        print(f"    {person:10s}: {count:4d} sequences ({pct:.1f}%)")
    
    print(f"\n  By Class (top 10):")
    for class_name, count in class_counts.most_common(10):
        pct = 100 * count / len(filtered_sequences)
        print(f"    {class_name:20s}: {count:4d} sequences ({pct:.1f}%)")
    
    # Show example sequences
    print(f"\n{'='*80}")
    print(f"SAMPLE SEQUENCES (first 10):")
    print(f"{'='*80}")
    for i, seq in enumerate(filtered_sequences[:10], 1):
        metadata = parse_sequence_name(seq)
        print(f"{i:2d}. {seq}")
        print(f"    → Person: {metadata['person_id']}, View: {metadata['view']}, "
              f"BG: {metadata['background']}, Light: {metadata['lighting']}, Class: {metadata['class_name']}")
    
    if len(filtered_sequences) > 10:
        print(f"\n    ... and {len(filtered_sequences) - 10} more sequences")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if len(filtered_sequences) == 0:
        print("⚠️  WARNING: No sequences match your filter criteria!")
        print("   Please adjust the filter settings in the config file.")
    else:
        print(f"✓ {len(filtered_sequences)} sequences will be preprocessed")
        print(f"✓ Configuration looks good!")
    
    print("\nTo change filter settings, edit: configs/custom_gesture_config.yaml")
    print("Look for the PRECOMPUTING.filter section.\n")


if __name__ == '__main__':
    main()
